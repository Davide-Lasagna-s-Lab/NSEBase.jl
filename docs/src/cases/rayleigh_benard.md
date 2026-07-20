# Rayleigh–Bénard convection

Rayleigh–Bénard convection is available on the physical two-dimensional `(x,y)` channel and the
full `(x,y,z)` channel. Both cases use the same wall-normal FDGrids discretisation and conduction
profile; the state and Boussinesq formulation change with dimension.

```julia
using NSEBase

grid_2d = TwoDimensionalChannelGrid(9, 17; Nt=1, α=0.5, width=5)
equations_2d = RayleighBenardFlow(grid_2d, 1.0, 0.71, 1000.0;
                                  fftw_flags=FFTW.ESTIMATE, dealias=false)

grid_3d = ChannelGrid(9, 17, 9; Nt=1, α=0.5, β=0.5, width=5)
equations_3d = RayleighBenardFlow(grid_3d, 1.0, 0.71, 1000.0;
                                  fftw_flags=FFTW.ESTIMATE, dealias=false)
```

The 2D grid takes `(Nx,Ny)` and stores fields as `(y,x,t)`. The 3D grid takes `(Nx,Ny,Nz)` and
stores them as `(y,x,z,t)`. In both layouts `x` remains horizontal/streamwise, `y` remains
vertical/wall-normal, `ddx!` is the streamwise Fourier derivative, and `ddy!` is the wall-normal
finite-difference derivative.

## Equations and state order

For velocity `U`, temperature `θ`, and gravity in the positive wall-normal component, both
formulations solve

```math
\frac{\partial U}{\partial t} +(U\cdot\nabla)U
=-\nabla p +\frac{1}{Re}\nabla^2U + Ri\,\theta\,\mathbf e_y + f,
```

```math
\frac{\partial\theta}{\partial t} +(U\cdot\nabla)\theta
=\frac{1}{Re\,Pr}\nabla^2\theta,
\qquad \nabla\cdot U=0,
\qquad Ri=\frac{Ra}{Re^2Pr}.
```

The raw full-field operators evaluate the non-pressure terms shown on the right-hand side. Pressure
is eliminated by a solenoidal basis/projection or handled by the downstream residual formulation;
the grid itself does not solve a pressure equation.

The layout-specific state conventions are:

| Grid | Formulation | State | Base tuple |
|:--|:--|:--|:--|
| `TwoDimensionalChannelGrid` | `CartesianPrimitive2DBoussinesq` | `(u,v,θ)` | `(U,V,Θ)` |
| `ChannelGrid` | `CartesianPrimitive3DBoussinesq` | `(u,v,w,θ)` | `(U,V,W,Θ)` |

`RayleighBenardFlow` always sets `grav=2`, so buoyancy acts on physical wall-normal velocity in
both states. For natural convection without an imposed velocity scale, `Re=1` gives `Ri=Ra/Pr`.
Setting `Ra=0` retains temperature as a passive scalar.

The forward linearisation includes both thermal couplings: a temperature perturbation produces
buoyancy, and a velocity perturbation advects the base temperature. The discrete adjoint transposes
both terms exactly. The concrete 2D operators can also be constructed directly when a forward
linearisation is needed:

```julia
formulation = CartesianPrimitive2DBoussinesq(0.71, 1000 / 0.71)
forward = CartesianPrimitive2DBoussinesqLNSE(grid_2d, 1.0, formulation;
                                             mode=Forward(), flags=FFTW.ESTIMATE)
```

`RayleighBenardFlow` returns the adjoint-oriented `ProjectedNSE` bundle and accepts
`mode=AdjointDiscrete()` or `mode=AdjointContinuous()`. Use the direct LNSE constructor above to
request `Forward()`.

## Conduction base and hydrostatic balance

The default base is motionless conduction with

```math
\Theta(\eta)=\frac{1-\eta}{2},
\qquad \eta=2\frac{y-y_-}{y_+-y_-}-1.
```

[`rbc_base_temperature`](@ref) constructs this profile, giving lower- and upper-plate values one
and zero for any channel `lim`. The default tuples are `(nothing,nothing,Θ)` in 2D and
`(nothing,nothing,nothing,Θ)` in 3D.

The raw primitive-variable operator evaluated at conduction contains the wall-normal buoyancy
`Ri*Θ`; it is not a zero vector by itself. In the physical equilibrium that gradient force is
cancelled by hydrostatic pressure. A test or downstream residual should account for pressure or
projection before treating conduction as a zero residual.

## Boundary conditions

The grids include wall points but neither the grid nor `RayleighBenardFlow` imposes no-slip or
fixed-temperature values. The basis or residual formulation must enforce velocity and thermal
boundary conditions. `StreamwiseInvariantChannelGrid` is deliberately not accepted by this case:
its physical plane is `(y,z)` and its three-component state `(v,w,u)` is the streamwise-invariant
hydrodynamic formulation, not the two-dimensional `(u,v,θ)` Boussinesq state.
