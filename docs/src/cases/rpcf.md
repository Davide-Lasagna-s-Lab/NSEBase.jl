# Streamwise-independent rotating plane Couette flow

RPCF is the 2D3C `ChannelGrid` layout: the solution is independent of
streamwise position but retains all three velocity components.

```julia
using NSEBase

grid = ChannelGrid(17, 9; Nt=1, β=0.5, width=5)
equations = PlaneCouetteFlow(grid, 500; Ro=0.2,
                             fftw_flags=FFTW.ESTIMATE, dealias=false)
```

The grid arguments are `(Ny,Nz)`. `Nt`, `β=2π/Lz`, and the FD stencil
`width` are keywords, following the same dimensional constructor pattern as
`LidDrivenCavityGrid`. The equivalent compact form is
`ChannelGrid(Ny,Nz,Nt,width,β)`: every resolution precedes the stencil width.

There is no separate RPCF grid type or equation constructor: the common
`ChannelGrid` factory and its streamwise-independent `PlaneCouetteFlow` method
provide the complete case implementation.

## Layout and state order

Arrays are stored as `(y,z,t)`: wall-normal finite differences, spanwise rFFT,
and temporal FFT. The 2D3C formulation calls physical wall-normal position its
computational `x` coordinate and physical spanwise position its computational
`y` coordinate. Consequently `ddx!` is wall-normal and `ddy!` is spanwise.

The state ordering is `(v,w,u)`: wall-normal and spanwise in-plane velocities,
followed by streamwise velocity. The default base is
`(nothing,nothing,U(η))`, where `U(η)=η` reaches `-1` and `1` at the walls for
any valid `lim`. `rpcf_base` is a compatibility name for that shared profile.

## Rotation

`Ro=2Ωh/Uw` is the rotation number about the physical spanwise axis. The
forward Coriolis block is

```text
out[1] -= Ro*u[3]
out[3] += Ro*u[1]
```

and both signs reverse for continuous and discrete adjoints. `Ro=0` installs
`NoForce`; otherwise the shared `CoriolisForce` uses `components=(3,1)`.
