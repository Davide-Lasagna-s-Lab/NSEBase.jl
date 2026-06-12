# Concepts & Conventions

This page documents the core assumptions, conventions, and constraints that
NSEBase builds on.  Understanding these is essential both for using the package
correctly and for implementing a new downstream grid.

## The grid interface

Everything in NSEBase is parameterised on a **grid** — a concrete subtype of
`AbstractGrid{T, D, AXES, FFT_DIMS_ORDER, DECOMPOSITION}`.  The five compile-time type parameters
encode all structural information about the domain:

| Parameter | Meaning |
|-----------|---------|
| `T <: Real` | Scalar real type (typically `Float64`) |
| `D` | Number of array dimensions |
| `AXES` | Four-tuple `(x_dim, y_dim, z_dim, t_dim)` mapping logical Cartesian coordinates to array dimensions; use `nothing` for absent coordinates |
| `FFT_DIMS_ORDER` | Ordered tuple of the array dimensions that are FFT-transformed; `FFT_DIMS_ORDER[1]` is **always** the rfft dimension |
| `DECOMPOSITION` | Storage-partition tag: `Undecomposed` for a complete local domain, or `Decomposed{DIMS}` for a grid split along the storage dimensions in `DIMS` |

Because these parameters are part of the type, the compiler can fully specialise
every loop, index mapping, and generated function at compile time — there is no
runtime dispatch and no runtime allocation in the hot paths.

NSEBase records decomposition metadata but does not prescribe communication or
halo storage. Downstream packages use [`decomposition_dims`](@ref) to inspect the
partitioned storage dimensions and provide the corresponding distributed
operations.

### Required methods

A concrete grid must implement exactly four methods:

```julia
Base.size(grid)                       # → NTuple{D, Int}
points(grid; dealias=false)           # → one array per dimension
wavenumber_scale(grid, dim::Int)      # → Real (2π/L for spatial, 1 for temporal)
weights(grid)                         # → AbstractArray (quadrature weights)
```

`weights` must have one axis per entry in `inhomogeneous_dims(grid)` in ascending
array-dimension order.

### Homogeneous vs. inhomogeneous dimensions

NSEBase draws a sharp distinction between two kinds of dimensions:

- **Homogeneous** (a.k.a. FFT-transformed) dimensions — entries of
  `FFT_DIMS_ORDER`.  These are statistically periodic and are represented in
  spectral space as Fourier coefficients.  Derivatives are exact multiplications
  by `im · n · wavenumber_scale`.
- **Inhomogeneous** dimensions — the complement of `FFT_DIMS_ORDER` in `1:D`.
  These are non-periodic (e.g. the wall-normal direction in channel flow).
  NSEBase provides no derivative for them; downstream packages must extend `ddx!`
  with their own matrix-multiply or spectral-element method.

---

## Field types

NSEBase provides four field types.  All are subtypes of `AbstractArray` and store
a reference to their grid alongside their data.

### `Field`

A physical-space scalar field.  The underlying array has shape `size(grid)` and
real element type `T`.  Used as a scratch buffer for de-aliased nonlinear
products.

```julia
u = Field(grid)                       # zero field
u = Field(grid, (x, y) -> sin(x)*y)  # initialise from a function
```

### `FTField`

A spectral-space scalar field — the rfft of a `Field`.  The underlying array has
complex element type and shape `transform_size(grid)`, which is `size(grid)` with
dimension `FFT_DIMS_ORDER[1]` halved to `(N÷2)+1`.

The remaining transformed dimensions keep their full size in **FFTW wrap-around
order**: indices `1:n÷2+1` hold non-negative wavenumbers and indices `n÷2+2:n`
hold negative wavenumbers in reverse order.

Two invariants are **enforced on construction** (when given a plain `Array`):

1. **Hermitian symmetry** — in the zero-rfft-wavenumber plane every coefficient
   at signed wavenumber `(0, k₂, k₃, …)` must satisfy
   `û(0, -k₂, -k₃, …) = conj(û(0, k₂, k₃, …))`.
2. **Real mean** — the fully-zero-wavenumber mode `û(0, 0, …)` must be real.

### `VectorField`

A fixed-length tuple of `FTField`s (or `Field`s) representing a vector-valued
quantity such as a velocity field.  `VectorField{N}` holds exactly `N` scalar
components.  Broadcasting, `copy`, `zero`, `similar`, etc. are all
component-wise.

### `ProjectedField`

A Galerkin-reduced representation of a vector field.  Instead of storing full
spectral coefficients for every wavenumber and every wall-normal point, it stores
`Nm` scalar coefficients per wavenumber:

```
parent(a)[m, i_H1, i_H2, …]   m ∈ 1:Nm, spectral storage indices
```

The mode index occupies axis 1 and the spectral axes follow in `FFT_DIMS_ORDER`
order.  Non-FFT (inhomogeneous) dimensions are **absent** — the basis functions
absorb the wall-normal dependence.  The same Hermitian-symmetry and real-mean
invariants as `FTField` apply.

---

## Wavenumber conventions

### Storage order and signed wavenumbers

The rfft dimension (`FFT_DIMS_ORDER[1]`) stores only non-negative wavenumbers
`0, 1, …, N÷2`.  All other transformed dimensions use FFTW's wrap-around order.
The helper type `WaveNumberVector{N}` converts between storage indices and signed
wavenumber integers, handling the conjugate-symmetry lookup for the rfft axis.

### Physical wavenumber scale

The actual physical wavenumber associated with integer mode `n` in an
`FFT_DIMS_ORDER[j]` dimension is

```
k_physical = n × wavenumber_scale(grid, FFT_DIMS_ORDER[j])
```

For a spatial dimension with period `L` the convention is `wavenumber_scale = 2π/L`.
For a temporal direction with unit period the convention is `wavenumber_scale = 1`.

Downstream grids must implement `wavenumber_scale` for every dimension in
`fft_dims(grid)`.

---

## Spectral derivative convention

`ddx!(out, u, Val{DIM})` computes the in-place spectral derivative along array
dimension `DIM`:

```
out[k] = +im · n · wavenumber_scale(grid, DIM) · u[k]     (adjoint=false)
out[k] = -im · n · wavenumber_scale(grid, DIM) · u[k]     (adjoint=true)
```

The `adjoint=true` form is the L² adjoint of the spectral derivative operator.
`DIM` must lie in `FFT_DIMS_ORDER`; passing an inhomogeneous dimension throws
`NotImplementedError`.

Four named wrappers pick the array dimension from `AXES`:

| Wrapper | Logical coordinate |
|---------|--------------------|
| `ddx_1!` | x (`AXES[1]`) |
| `ddx_2!` | y (`AXES[2]`) |
| `ddx_3!` | z (`AXES[3]`) |
| `ddx_4!` | t (`AXES[4]`) |

When the coordinate is absent (`AXES[j] === nothing`) the wrapper is a compile-time
no-op.

---

## Inner product and norm conventions

The L² inner product is defined as

```math
\langle u, v \rangle
  = \frac{1}{2} \sum_{\mathbf{k}} c_{k_1}\, w(\mathbf{j})\,
    \operatorname{Re}\!\bigl(\bar{u}_{\mathbf{k},\mathbf{j}}\, v_{\mathbf{k},\mathbf{j}}\bigr),
```

where:

- The sum runs over all stored spectral wavenumbers **k** and inhomogeneous
  indices **j**.
- `c_{k₁}` is **1** for the zero rfft wavenumber and **2** for all others,
  accounting for Hermitian symmetry.
- `w(j)` are the quadrature weights returned by `weights(grid)`.
- The `1/2` factor combined with the `c_{k₁}` multiplier ensures equivalence with
  the continuous L² norm: the rfft discards the negative-wavenumber half of the
  spectrum, which carries equal energy.

`dot(u, v)` implements this for `FTField`, `VectorField{<:FTField}`, and
`ProjectedField`.

---

## Phase shifts

A continuous shift by displacement `s` in a homogeneous direction is an **exact**
spectral operation — no interpolation is needed.  The phase factor applied to mode
`n` is

```
exp(im · n · s · wavenumber_scale(g, dim))
```

`shift!(u, shifts)` applies the product of phase factors over all homogeneous
dimensions in `FFT_DIMS_ORDER` order.  Shifts are given in **physical units**
(same units as the period `L`).

---

## De-aliasing

Physical-space nonlinear products are computed on a padded grid using the **3/2
rule**: the physical-space arrays in `pcache` are allocated with `dealias=true`,
which asks `growto(grid, target_size)` to extend each homogeneous dimension to
`3N/2`.  NSEBase zero-pads the spectral coefficients before transforming to the
padded grid and symmetrically truncates after transforming back, eliminating
aliasing errors from quadratic nonlinearities.

Downstream grids must implement `growto` to support de-aliasing.

---

## NSE Operators

NSEBase includes a single Cartesian primitive-variable operator family,
`CartesianPrimitiveNSE{NDIM,NCOMP}`. `NDIM` is the number of spatial derivative
directions, `NCOMP` is the number of advected components, and `MODE <: Mode`
selects whether the object applies the nonlinear equation or a linearised
variant. The common cases have short aliases:

| Operator alias | Components |
|---------------|------------|
| `CartesianPrimitive2DNSE` | u, v on a 2D grid |
| `CartesianPrimitive2D3CNSE` | u, v, w on a 2D grid |
| `CartesianPrimitive3DNSE` | u, v, w on a 3D grid |

Each operator is parameterised on a `MODE <: Mode` tag and an advection
`FORM <: AdvectionForm`:

| `MODE` | Operator computed |
|--------|------------------|
| `NonLinear` | Nonlinear NSE |
| `Forward` | Standard linearised NSE around base flow |
| `AdjointContinuous` | Continuous adjoint of the linearised operator |
| `AdjointDiscrete` | Discrete (numerical) adjoint — exact transpose of `Forward` |

---

## `construct_equations` and `ProjectedNSE`

`construct_equations` is the recommended entry point for building a solver.  It:

1. Allocates two shared scratch pools — `scache` (spectral) and `pcache`
   (physical), sized for the linearised operator.
2. Builds FFTW plans for the grid.
3. Instantiates the nonlinear and linearised operators, wiring them to the same
   scratch pools.
4. Wraps everything in a `ProjectedNSE` struct.

`ProjectedNSE` is callable:

```julia
obj(out, a)       # nonlinear NSE: out = P(Δu/Re − (u·∇)u + force)
obj(out, a, b)    # linearised NSE: out = P(L_a · b)
```

where `a` and `b` are `ProjectedField`s.

---

## Memory and allocation policy

Every hot-path operation in NSEBase is **allocation-free** at steady state.
Cache arrays are pre-allocated once inside the operator structs and reused on
every call.  The `@generated` loop generators (`for_each_homogeneous_index`,
`for_each_index`) emit fully unrolled code specialised to the grid type — no
closures, no dynamic dispatch, no heap allocations.

When writing downstream code, prefer the in-place (`!`) variants of all
operations.  The allocating wrappers (`FFT`, `IFFT`, `shift`, `project`,
`expand`) are convenience forms intended for interactive exploration and tests.
