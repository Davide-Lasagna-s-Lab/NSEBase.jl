# Channel grids and flows

`ChannelGrid` is one factory for every wall-bounded case with one
finite-difference wall-normal direction. The number of positional spatial
resolutions selects the layout, while time and Fourier scales are keywords.

| Layout | Constructor | Storage | Fourier directions | Exact contract |
|:--|:--|:--|:--|:--|
| Streamwise-independent 2D3C | `ChannelGrid(Ny,Nz; Nt,β)` | `(y,z,t)` | spanwise, `t` | `AbstractChannel2D3CGrid` |
| Full 3D | `ChannelGrid(Nx,Ny,Nz; Nt,α,β)` | `(y,x,z,t)` | streamwise, spanwise, `t` | `AbstractChannel3DGrid` |

Both satisfy `AbstractChannelGrid` and contain the same one-tuples of
wall-normal points, derivatives, weighted adjoints, and quadrature weights.

## Grid construction

```julia
using NSEBase

reduced = ChannelGrid(65, 63; Nt=1, β=1, width=7)
full = ChannelGrid(63, 65, 63; Nt=1, α=0.5, β=1, width=7)
```

For compact positional calls, put every resolution first, then the stencil
width, then the Fourier scales: `ChannelGrid(Ny,Nz,Nt,width,β)` and
`ChannelGrid(Nx,Ny,Nz,Nt,width,α,β)`.

The default Gauss–Lobatto distribution includes both walls and supplies
Clenshaw–Curtis quadrature. `width=5` is the default stencil width. Every
Fourier resolution must be positive and odd; `width` must be odd and at least
three, with `Ny > 2width` for the compact weighted adjoints.

The 2D3C layout uses the computational coordinates required by
`CartesianPrimitive2D3C`: `ddx!` is the physical wall-normal derivative and
`ddy!` is the physical spanwise derivative. This is why its axis map is
`CHANNEL_2D3C_AXES == (1,2,nothing,3)` rather than the physical 3D channel
map with a missing streamwise entry.

For precomputed full-channel data, both the migrated explicit-adjoint form and
the original ReSolver-ChannelFlow form are available:

```julia
g = ChannelGrid(y, Nx, Nz, Nt, α, β, Dy, Dy2, Dya, Dy2a, wy)
g = ChannelGrid(y, Nx, Nz, Nt, α, β, Dy, Dy2, wy; adjoint_diff=true)
```

## Plane Couette flow

`PlaneCouetteFlow` dispatches on the layout:

```julia
full_equations = PlaneCouetteFlow(full, 500; Ro=0.1, fftw_flags=FFTW.ESTIMATE, dealias=false)
reduced_equations = PlaneCouetteFlow(reduced, 500; Ro=0.2, fftw_flags=FFTW.ESTIMATE, dealias=false)
```

The full state is `(u,v,w)` with base `(U,nothing,nothing)` and Coriolis pair
`(1,2)`. The streamwise-independent state is `(v,w,u)` with base
`(nothing,nothing,U)` and Coriolis pair `(3,1)`. In both cases
`U(η)=η`, where the global wall interval is mapped to `η∈[-1,1]`.

## Plane Poiseuille flow

```julia
equations = PlanePoiseuilleFlow(full, 500; f=1, Ro=0,
                                fftw_flags=FFTW.ESTIMATE, dealias=false)
```

Poiseuille flow uses the full 3D layout. Its default base is
`U(η)=1-η²`; `f` is a constant streamwise pressure-gradient force at the zero
Fourier mode. Nonzero `Ro` combines pressure and Coriolis terms.

## Shared contracts

The canonical Couette, Poiseuille, and conduction profiles use the same global
normalized wall coordinate. On MPI-decomposed grids each rank evaluates its
local slice using the undecomposed parent's wall limits. All flow constructors
default to `AdjointDiscrete()`, `FFTW.EXHAUSTIVE`, and 3/2 de-aliasing.

The grid supplies wall points and operators but does not impose boundary
values. A basis or residual formulation must enforce the wall conditions.
