# Migrating from case packages

The former channel, square-duct, cavity, rotating-Couette, and
Rayleigh–Bénard packages duplicated grid storage and interface methods. Their
case constructors now use NSEBase's shared `RectangularGrid` implementation.

## Package mapping

| Former package | NSEBase API |
|:--|:--|
| ReSolver-ChannelFlow.jl | `ChannelGrid`, `PlaneCouetteFlow`, `PlanePoiseuilleFlow` |
| ReSolver-SquareDuct.jl | `SquareDuctGrid`, `SquareDuctFlow` |
| Resolver-LidDrivenCavity.jl | `LidDrivenCavityGrid`, `LidDrivenCavityFlow` |
| ReSolver-RPCF.jl | `StreamwiseInvariantChannelGrid`, `PlaneCouetteFlow` |
| ReSolver-RayleighBenard.jl | `TwoDimensionalChannelGrid` or 3D `ChannelGrid`, `RayleighBenardFlow` |

Replace package imports with `using NSEBase`. ReSolver-RPCF.jl maps to an
explicitly named streamwise-invariant constructor. Rayleigh–Bénard can use the
physical two-dimensional channel or the original full channel layout.

## Channel constructors

The preferred constructors name the reduced physical formulation explicitly:

```julia
StreamwiseInvariantChannelGrid(Ny, Nz; Nt=1, β=1, ...) # wall-normal–spanwise 2D3C
TwoDimensionalChannelGrid(Nx, Ny; Nt=1, α=1, ...)      # planar flow / Rayleigh–Bénard
ChannelGrid(Nx, Ny, Nz; Nt=1, α=1, β=1, ...)           # full 3D / Rayleigh–Bénard
```

This removes low-level axis and FFT parameters from user code without making
the meaning of a two-resolution call depend on argument position. Compact
forms place every resolution first, followed by the stencil width and Fourier
scales:

```julia
StreamwiseInvariantChannelGrid(Ny, Nz, Nt, width, β)
TwoDimensionalChannelGrid(Nx, Ny, Nt, width, α)
ChannelGrid(Nx, Ny, Nz, Nt, width, α, β)
```

These use physical resolution order and place `width` after every resolution.
Historical width-second and wall-normal-first forms are intentionally not
forwarded.

The explicit-adjoint low-level constructor remains
`ChannelGrid(y,Nx,Nz,Nt,α,β,Dy,Dy2,Dya,Dy2a,wy[,T])`. The original
ReSolver-ChannelFlow form is also restored:
`ChannelGrid(y,Nx,Nz,Nt,α,β,Dy,Dy2,wy[,T]; adjoint_diff=true)`.

## Channel layout dispatch

Use `AbstractStreamwiseInvariantChannelGrid` for the `(y,z,t)` 2D3C layout,
`AbstractTwoDimensionalChannelGrid` for the `(y,x,t)` planar layout,
`AbstractChannel3DGrid` for the full `(y,x,z,t)` layout, and
`AbstractChannelGrid` when all channel layouts are valid. These structural
contracts accept the bundled serial grid, MPI-decomposed grids, and compatible
downstream implementations. The high-level names are factory functions, not
concrete types, so type checks should use the appropriate abstract contract.

`PlaneCouetteFlow` and `PlanePoiseuilleFlow` dispatch on all three layouts.
Full-channel state ordering is `(u,v,w)` with the streamwise base and force in
component one; the streamwise-invariant 2D3C ordering is `(v,w,u)` with them
in component three. The two-dimensional layout uses state `(u,v)` and base `(U,nothing)`.
`RayleighBenardFlow` accepts the two-dimensional and full 3D layouts. Their
states are `(u,v,θ)` and `(u,v,w,θ)`, respectively; the streamwise-invariant
`(y,z)` layout is intentionally not a Rayleigh–Bénard case.

The streamwise-invariant 2D3C axis map is `(nothing,1,2,3)`: physical
streamwise `x` is absent, wall-normal `y` occupies storage dimension 1,
spanwise `z` occupies dimension 2, and `t` occupies dimension 3. Thus `ddy!`
is the wall-normal derivative, `ddz!` is the spanwise derivative, and MPI
wall-normal decomposition uses physical symbol `:y`, just as it does for the two-dimensional and
full layouts.

## Function-valued field construction

Functions passed to `Field` and `VectorField` no longer follow array storage
order. They always receive four arguments in physical `(x,y,z,t)`
order, with `nothing` for an absent coordinate. In particular, migrate a full
channel initializer from

```julia
u(y, x, z, t) = (1 - y^2) * cos(x) * sin(z)
```

to

```julia
u(x, y, z, t) = (1 - y^2) * cos(x) * sin(z)
```

A 2D-cavity or two-dimensional channel callable receives
`(x,y,nothing,t)`. A streamwise-invariant 2D3C callable receives
`(nothing,y,z,t)`: the first argument records the absent physical streamwise
coordinate rather than relabelling the active wall-normal and spanwise
coordinates. `points(grid)` itself remains in storage order; only the
function-valued field constructors perform this permutation.

## Other grid factories

`SquareDuctGrid(N,Nz,Nt,α; width=...)` is the sole duct case constructor; the
historical `SquareDuctGrid(N,width,Nz,Nt,α)` form forwards to it.

`LidDrivenCavityGrid(Nx,Ny; Nt=...)` constructs a 2D cavity, while
`LidDrivenCavityGrid(Nx,Ny,Nz; Nt=...,spanwise=...)` constructs a 3D cavity.
`LidDrivenCavityFlow` requires an explicit base lifting so moving-lid values
cannot be silently omitted.

## Field mapping

Direction-specific rectangular-grid data is tuple-valued:

| Former field | Unified field |
|:--|:--|
| `g.y` or scalar `g.xs` | `g.xs[1]` |
| scalar `g.Dy`, `g.Dy2` | `g.D₁[1]`, `g.D₂[1]` |
| scalar `g.Dya`, `g.Dy2a` | `g.D₁⁺[1]`, `g.D₂⁺[1]` |
| scalar `g.ws` | `g.ws[1]` |
| cavity `g.x` | `g.xs[1]`, `g.xs[2]` |
| cavity `g.D1`, `g.D2` | `g.D₁[i]`, `g.D₂[i]` |
| cavity `g.D1adj`, `g.D2adj` | `g.D₁⁺[i]`, `g.D₂⁺[i]` |
| cavity or square-duct product weights | `weights(g)` or `g.w` |
| `g.α`, `g.β` | `wavenumber_scale(g, storage_dim(g,...))` |

`SquareDuctGrid` deliberately shares its two cross-section objects. Cavity
directions may have different sizes, distributions, and stencil types.

## Shared numerical documentation

The old per-package grid hooks now have one authoritative implementation and
docstring:

| Former per-case API | Unified binding |
|:--|:--|
| grid fields and ownership | `RectangularGrid` |
| physical coordinates | `points(::RectangularGrid)` |
| Fourier scales | `wavenumber_scale(::RectangularGrid,::Int)` |
| product quadrature | `weights(::RectangularGrid)` |
| homogeneous growth | `growto(::RectangularGrid,::Tuple)` |
| precision conversion | `Base.convert(::Type,::RectangularGrid)` |
| FD derivatives and weighted adjoints | `dd!`, `inhomogeneous_dd!` |
| finite-difference Laplacian | `inhomogeneous_laplacian!` |
| serial/MPI stencil lookup | `derivative_matrix` |

Case docstrings retain geometry, state ordering, normalisation, boundary
ownership, and flow-constructor semantics.

## Profiles and forcing

Couette, Poiseuille, and Rayleigh–Bénard conduction profiles now share the
global normalized wall coordinate `η∈[-1,1]`. This gives canonical wall values
for arbitrary `lim` and avoids independently normalizing each MPI slab.

The old case packages used implicit forcing components. NSEBase makes them
explicit:

```julia
ConstantBodyForce(value; component=1)       # full-channel streamwise force
CoriolisForce(Ro)                           # full-channel pair (1,2)
CoriolisForce(Ro; components=(3,1))         # streamwise-invariant 2D3C pair
```

The flow constructors select these conventions automatically.

Every bundled temporal coordinate covers `[0,1)` and therefore uses Fourier
scale `2π`. Wall values remain the responsibility of a basis or residual. For
lid-driven cavities the required base lifting carries the inhomogeneous lid
values while the perturbation representation enforces homogeneous conditions.
