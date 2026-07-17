# Rayleigh–Bénard convection

Rayleigh–Bénard convection uses the full 3D `ChannelGrid` layout; temperature
and buoyancy distinguish the formulation rather than a separate grid type.

```julia
using NSEBase

grid = ChannelGrid(9, 17, 9; Nt=1, α=0.5, β=0.5, width=5)
equations = RayleighBenardFlow(grid, 1.0, 0.71, 1000.0;
                               fftw_flags=FFTW.ESTIMATE, dealias=false)
```

The spatial resolutions are `(Nx,Ny,Nz)`. Temporal resolution, Fourier
scales, and FD stencil width are keywords, as in the other channel layouts.
The equivalent compact form is
`ChannelGrid(Nx,Ny,Nz,Nt,width,α,β)`.

## Physical model

The state is `(u,v,w,θ)`. `RayleighBenardFlow` computes

```math
Ri = \frac{Ra}{Re^2 Pr}
```

and selects `CartesianPrimitive3DBoussinesq(Pr,Ri; grav=2)`. For natural
convection without an imposed velocity scale, `Re=1` gives `Ri=Ra/Pr`.
`Ra=0` reduces temperature to a passive scalar.

The default base is motionless conduction with
`Θ(η)=(1-η)/2`, returned by `rbc_base_temperature`. The shared normalized wall
coordinate gives lower and upper plate values one and zero for any valid
channel `lim`, including MPI-decomposed grids.

Velocity and temperature boundary conditions must be supplied by the basis or
residual formulation.
