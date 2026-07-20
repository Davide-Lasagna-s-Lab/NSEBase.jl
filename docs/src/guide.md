# Concepts and conventions

## Grid structure

Every field carries an `AbstractGrid{T,D,AXES,FFT_DIMS_ORDER}`. The type
parameters encode structural information that must remain fixed for a concrete
grid type:

| Parameter | Meaning |
|:--|:--|
| `T` | Real scalar type |
| `D` | Number of storage dimensions |
| `AXES` | Mapping `(x,y,z,t)` to storage dimensions; `nothing` means absent |
| `FFT_DIMS_ORDER` | Homogeneous storage dimensions; the first uses rFFT |

`RectangularGrid{NI}` is the built-in tensor-product implementation. `NI` is
the number of inhomogeneous FDGrids directions. Case factories select fixed
layout contracts, so users normally specify only resolutions, domain scales,
and optional finite-difference settings.

## Physical order and storage order

Physical directions always mean `(x,y,z,t)`. Storage order is the order of
array dimensions and may differ for performance. A channel is stored as
`(y,x,z,t)`, placing the wall-normal matrix product on the leading dimension:

| Physical direction | Storage dimension | Discretisation |
|:--|:--:|:--|
| `x` | 2 | rFFT |
| `y` | 1 | FDGrids |
| `z` | 3 | FFT |
| `t` | 4 | FFT |

Use `storage_dim`, `physical_dim`, and the `*_physical_dims`/
`*_storage_dims` queries instead of hard-coding this mapping in generic code.
Functions passed to `Field(grid, func)` and `VectorField(grid, funcs...)` always
receive four arguments in physical `(x,y,z,t)` order; absent coordinates are
`nothing`, and the constructors perform the storage permutation internally.
There are no coordinate-name exceptions. The streamwise-invariant 2D3C channel
uses axis map `(nothing,1,2,3)`: physical `x` is absent, `y` is the wall-normal
storage dimension 1, and `z` is the spanwise storage dimension 2. Its active
derivatives are therefore `ddy!` and `ddz!`, and a function-valued field
receives `(nothing,y,z,t)`. `TwoDimensionalChannelGrid` represents only the physical `(x,y)`
channel, stores it as `(y,x,t)`, and has no spanwise coordinate. Hydrodynamic cases use velocity
state `(u,v)`; two-dimensional Boussinesq flow adds temperature as `(u,v,θ)`.

## Homogeneous directions and wavenumbers

The first entry of `fft_storage_dims(g)` is stored by a real-to-complex FFT and
contains nonnegative wavenumbers. Later entries use FFTW wrap-around order.
`WaveNumberVector` provides signed modal indexing without exposing storage
indices.

For a periodic coordinate of length `L`, integer mode `n` represents physical
wavenumber

```math
k = n\,\frac{2\pi}{L}.
```

Consequently `wavenumber_scale(g, dim) == 2π/L`. Every bundled temporal
coordinate covers `[0,1)`, so its scale is `2π`, not `1`.

## Inhomogeneous directions

Rectangular grids store, for each inhomogeneous direction:

- collocation points `xs[i]`;
- first and second FDGrids operators `D₁[i]` and `D₂[i]`;
- caller-supplied weighted adjoints `D₁⁺[i]` and `D₂⁺[i]`; and
- one-dimensional quadrature weights `ws[i]`.

The tuples follow `inhomogeneous_storage_dims(g)` order. `weights(g)` returns
their tensor product. Matrix products unwrap `parent(field)` before calling
FDGrids' dimension-aware `mul!`, ensuring the specialised stencil kernels are
used instead of a generic array wrapper path.

## Fields and transforms

`Field` stores real physical values with shape `size(g)`. `FTField` stores the
Fourier representation, with the rFFT dimension reduced to `N÷2+1`.
`VectorField` groups scalar fields into a fixed component tuple, and
`ProjectedField` stores coefficients in a supplied Galerkin basis.

`FFTPlans` owns compatible forward and inverse plans. The allocating `FFT` and
`IFFT` helpers are convenient for setup and examples; hot paths should use the
in-place plan calls.

Because physical fields are real, the zero-rFFT plane must be Hermitian across
the remaining Fourier dimensions and the all-zero mode must be real. `FTField`
and `ProjectedField` constructors sanitise ordinary array inputs to enforce
these invariants. Modal indexing through `WaveNumberVector` applies the same
conjugate-symmetry convention when a negative rFFT wavenumber is requested.

## Derivatives and Laplacian

`ddx!`, `ddy!`, `ddz!`, and `ddt!` map physical names through `AXES`.
Homogeneous derivatives multiply each Fourier coefficient by
`im*n*wavenumber_scale`. Inhomogeneous derivatives apply `D₁` along the
corresponding storage dimension.

The spatial Laplacian deliberately excludes the logical time direction.
`inhomogeneous_laplacian!` applies the FD contributions and
`add_homogeneous_laplacian!` adds spatial Fourier contributions. Explicit
implementations for `NI=1,2,3` keep the FD sequence statically specialised.

Passing `adjoint=true` selects the stored weighted FD adjoints and reverses the
sign of Fourier first derivatives. These are the operators used by the discrete
adjoint formulations.

## Inner products

The spectral inner product combines:

- quadrature weights in every inhomogeneous direction;
- FFT normalisation in homogeneous directions; and
- the factor-of-two multiplicity of nonzero rFFT modes.

This makes `dot(FFT(u), FFT(v))` agree with the corresponding quadrature
integral over inhomogeneous directions and average over homogeneous directions.
The inhomogeneous measure is not normalised: a constant field on a channel of
height two has squared norm two. Each supplied `D⁺` satisfies the discrete
adjoint identity under this inner product.

## Phase shifts

A continuous displacement in a homogeneous direction is exact in spectral
space: integer mode `n` is multiplied by
`exp(im*n*shift*wavenumber_scale(g,dim))`. `shift!` accepts one physical-unit
displacement per entry of `fft_storage_dims(g)`. `normdiff` and `minnormdiff`
use the same convention to compare fields modulo continuous translations.

## De-aliasing

Quadratic nonlinear products use the 3/2 rule. With `dealias=true`, physical
caches are created on enlarged homogeneous dimensions. `growto` changes only
those sizes and retains the inhomogeneous points, one-dimensional weights, and
FD operators.

## Equation construction

`construct_equations` combines a grid, Reynolds number, base state, formulation
tag, body force, FFT plans, and shared caches into `ProjectedNSE`. Bundled case
constructors select the appropriate formulation:

| Case | Formulation | State order |
|:--|:--|:--|
| Full channel and square duct | `CartesianPrimitive3D` | `(u,v,w)` |
| Streamwise-invariant channel | `CartesianPrimitive2D3C` | `(v,w,u)` |
| Two-dimensional channel | `CartesianPrimitive2D` | `(u,v)` |
| 2D cavity | `CartesianPrimitive2D` | `(u,v)` |
| 3D cavity | `CartesianPrimitive3D` | `(u,v,w)` |
| 2D Rayleigh–Bénard | `CartesianPrimitive2DBoussinesq` | `(u,v,θ)` |
| 3D Rayleigh–Bénard | `CartesianPrimitive3DBoussinesq` | `(u,v,w,θ)` |

The default linearised mode is `AdjointDiscrete`, appropriate when an exact
adjoint of the discretised operator is required. `AdjointContinuous` remains
available when the continuously derived adjoint is desired.

The returned object operates on `ProjectedField` coefficients:

```julia
equations(out, a)       # nonlinear action
equations(out, a, b)    # linearised or adjoint action on b
```

Both calls expand into cached full fields, apply the primitive-variable
operator, and project back into the basis stored by the inputs. Call the
nonlinear form before the three-argument form when reusing the cached base-flow
gradients, as required by the current `ProjectedNSE` interface. The underlying
full operators remain available as `equations.nl` and `equations.ln`.

To add a formulation handled by the generic factory, define a tag and implement `ncomp`, both
`cache_length` methods, `nonlinear_operator`, and `linearised_operator`. A parameterised
formulation whose operator constructors need additional coefficients can instead overload
`construct_equations`, as the two Boussinesq formulations do. Bundled case constructors select
the appropriate factory method and physical defaults.

## Allocation policy

Constructors and convenience operations such as `FFT`, `IFFT`, `project`, and
`expand` allocate their results. Mutating derivatives, transforms, forcing,
projection, and warmed equation actions reuse caller-owned arrays or operator
caches and are covered by allocation regressions. Prefer the `!` forms in
iterative solvers.

## Boundary-condition contract

FDGrids distributions determine whether wall points are present, but a grid
does not impose boundary values. Bases or residual formulations must encode
no-slip walls and fixed thermal values. In a lid-driven cavity, the required
base lifting carries the inhomogeneous lid values while the perturbation basis
or residual enforces homogeneous wall conditions. Body forces are not
substitutes for boundary conditions.

## MPI decomposition

Loading MPI, FDGrids, and HaloArrays activates NSEBase's MPI extension.
`distributed` wraps a rectangular grid and partitions selected inhomogeneous
physical directions. The wrapper forwards derivative-matrix access to the
parent grid and uses halo-aware FDGrids kernels. Homogeneous decomposition is
not currently supported.

```julia
using MPI, FDGrids, HaloArrays, NSEBase

MPI.Init()
nranks = MPI.Comm_size(MPI.COMM_WORLD)
parent_grid = ChannelGrid(31, 16 * nranks, 31; Nt=1, α=0.5, β=0.5, width=5)
grid = distributed(parent_grid, MPI.COMM_WORLD;
                   decomposed_physical_dims=(:y,),
                   nprocesses=(nranks,), nhalo=(2,))
```

Every decomposed parent size must be divisible by its corresponding process
count; deriving `Ny` from `nranks` makes the example valid for any launch size.
