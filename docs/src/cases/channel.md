# Channel grids and flows

NSEBase provides three channel layouts. They share the same wall-normal
FDGrids discretisation but distinguish exactly which physical plane is represented. The
reduced-grid constructors make the `(y,z)` 2D3C and `(x,y)` two-dimensional layouts unambiguous.

| Layout | Constructor | Storage | Fourier directions | Exact contract |
|:--|:--|:--|:--|:--|
| Streamwise invariant, 2D3C | `StreamwiseInvariantChannelGrid(Ny,Nz; Nt,β)` | `(y,z,t)` | `z,t` | `AbstractStreamwiseInvariantChannelGrid` |
| Two-dimensional | `TwoDimensionalChannelGrid(Nx,Ny; Nt,α)` | `(y,x,t)` | `x,t` | `AbstractTwoDimensionalChannelGrid` |
| Full 3D | `ChannelGrid(Nx,Ny,Nz; Nt,α,β)` | `(y,x,z,t)` | `x,z,t` | `AbstractChannel3DGrid` |

All three satisfy `AbstractChannelGrid` and return `RectangularGrid{1}`
objects containing one-tuples of wall-normal points, derivatives, weighted
adjoints, and quadrature weights.

## Grid construction

```julia
using NSEBase

streamwise_invariant = StreamwiseInvariantChannelGrid(65, 63; Nt=1, β=1, width=7)
two_dimensional = TwoDimensionalChannelGrid(63, 65; Nt=1, α=0.5, width=7)
full = ChannelGrid(63, 65, 63; Nt=1, α=0.5, β=1, width=7)
```

Compact positional forms put every resolution first, followed by the stencil
width and Fourier scales:

```julia
StreamwiseInvariantChannelGrid(Ny, Nz, Nt, width, β)
TwoDimensionalChannelGrid(Nx, Ny, Nt, width, α)
ChannelGrid(Nx, Ny, Nz, Nt, width, α, β)
```

The default Gauss–Lobatto distribution includes both walls and supplies
Clenshaw–Curtis quadrature. `width=5` is the default stencil width. Every
Fourier resolution must be positive and odd; `width` must be odd and at least
three, with `Ny > 2width` for the compact weighted adjoints. `lim=(y₋,y₊)`
sets the wall locations and `T` selects the real scalar type.

### Coordinate conventions

Every layout keeps `x`, `y`, and `z` tied to the physical streamwise,
wall-normal, and spanwise directions. The full and two-dimensional layouts therefore use `ddx!`
for the streamwise Fourier derivative and `ddy!` for the wall-normal finite-difference derivative.

The streamwise-invariant layout has axis map `(nothing,1,2,3)`: physical `x` is
absent, `y` occupies storage dimension 1, `z` occupies dimension 2, and `t`
occupies dimension 3. Consequently `ddy!` is wall-normal and `ddz!` is
spanwise; `ddx!` refers only to the absent streamwise direction.
`CartesianPrimitive2D3C` is restricted to this physical `(y,z)` plane and uses those derivatives
without relabelling either coordinate. Functions passed to `Field` or
`VectorField` receive `(nothing,y,z,t)` in physical coordinate order even
though arrays remain stored as `(y,z,t)`.

For precomputed full-channel data, both the migrated explicit-adjoint form and
the original ReSolver-ChannelFlow form are available:

```julia
g = ChannelGrid(y, Nx, Nz, Nt, α, β, Dy, Dy2, Dya, Dy2a, wy)
g = ChannelGrid(y, Nx, Nz, Nt, α, β, Dy, Dy2, wy; adjoint_diff=true)
```

## Plane Couette flow

`PlaneCouetteFlow` supports all three layouts:

```julia
full_equations = PlaneCouetteFlow(full, 500; Ro=0.1,
                                  fftw_flags=FFTW.ESTIMATE, dealias=false)
reduced_equations = PlaneCouetteFlow(streamwise_invariant, 500; Ro=0.2,
                                     fftw_flags=FFTW.ESTIMATE, dealias=false)
planar_equations = PlaneCouetteFlow(two_dimensional, 500; Ro=0.1,
                                    fftw_flags=FFTW.ESTIMATE, dealias=false)
```

The full state is `(u,v,w)` with base `(U,nothing,nothing)` and Coriolis pair
`(1,2)`. The streamwise-invariant state is `(v,w,u)` with base
`(nothing,nothing,U)` and Coriolis pair `(3,1)`. The two-dimensional state is `(u,v)` with base
`(U,nothing)` and Coriolis pair `(1,2)`. In every layout
`U(η)=η`, where the global wall interval is normalized to `η∈[-1,1]`.

See [Streamwise-invariant rotating plane Couette flow](streamwise_invariant_channel.md)
for the 2D3C component and rotation conventions.

## Plane Poiseuille flow

`PlanePoiseuilleFlow` also supports all three layouts:

```julia
full_poiseuille = PlanePoiseuilleFlow(full, 500; f=1,
                                      fftw_flags=FFTW.ESTIMATE, dealias=false)
planar_poiseuille = PlanePoiseuilleFlow(two_dimensional, 500; f=1,
                                        fftw_flags=FFTW.ESTIMATE, dealias=false)
streamwise_invariant_poiseuille = PlanePoiseuilleFlow(streamwise_invariant, 500; f=1,
                                                      fftw_flags=FFTW.ESTIMATE, dealias=false)
```

All use the normalized streamwise base `U(η)=1-η²` and a constant streamwise
pressure-gradient force `f` at the zero Fourier mode. The full state and base
are `(u,v,w)` and `(U,nothing,nothing)`; the two-dimensional state and base are `(u,v)` and
`(U,nothing)`. In the streamwise-invariant `(v,w,u)` order,
the base and pressure force occupy component three.

See [Two-dimensional plane Poiseuille flow](two_dimensional_channel.md)
for the planar layout, derivative mapping, and exact laminar balance.

## Rayleigh–Bénard convection

`RayleighBenardFlow` supports the two-dimensional and full layouts. The planar state is `(u,v,θ)`
and selects `CartesianPrimitive2DBoussinesq`; the full state is `(u,v,w,θ)` and selects
`CartesianPrimitive3DBoussinesq`. Both use the conduction base `Θ(η)=(1-η)/2`, wall-normal
buoyancy, and `Ri=Ra/(Re²Pr)`:

```julia
planar_convection = RayleighBenardFlow(two_dimensional, 1, 0.71, 1e4;
                                       fftw_flags=FFTW.ESTIMATE, dealias=false)
full_convection = RayleighBenardFlow(full, 1, 0.71, 1e4;
                                     fftw_flags=FFTW.ESTIMATE, dealias=false)
```

See [Rayleigh–Bénard convection](rayleigh_benard.md) for the equations, state order, linearised
thermal couplings, and hydrostatic-pressure convention.

## Shared contracts

The canonical Couette, Poiseuille, and Rayleigh–Bénard conduction profiles use
the same global normalized wall coordinate. On MPI-decomposed grids each rank
evaluates its local slice using the undecomposed parent's wall limits. All flow
constructors default to `AdjointDiscrete()`, `FFTW.EXHAUSTIVE`, and 3/2
de-aliasing.

The grids supply wall points and numerical operators but do not impose
velocity or thermal boundary values. A basis or residual formulation must
enforce the wall conditions.
