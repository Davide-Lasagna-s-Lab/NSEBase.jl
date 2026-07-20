# Streamwise-invariant rotating plane Couette flow

A streamwise-invariant channel suppresses dependence on the physical
streamwise coordinate, `∂/∂x=0`, while retaining all three velocity
components. The resulting flow evolves in the wall-normal–spanwise `(y,z)`
plane and uses a 2D3C formulation: `(v,w)` advects the in-plane motion and the
streamwise velocity `u` is an active out-of-plane component.

```julia
using NSEBase

grid = StreamwiseInvariantChannelGrid(17, 9; Nt=1, β=0.5, width=5)
equations = PlaneCouetteFlow(grid, 500; Ro=0.2,
                             fftw_flags=FFTW.ESTIMATE, dealias=false)
```

The grid arguments are `(Ny,Nz)`. `β=2π/Lz` sets the spanwise period, `Nt`
sets the unit-period temporal or phase resolution, and `width` controls the
wall-normal FDGrids stencil. The default Gauss–Lobatto distribution includes
both walls and supplies the quadrature used by discrete-adjoint derivatives.
The equivalent compact positional form is
`StreamwiseInvariantChannelGrid(Ny,Nz,Nt,width,β)`.

## Layout and state order

Arrays are stored as `(y,z,t)`: wall-normal finite differences, spanwise rFFT,
and temporal FFT. The physical axis map is `(nothing,1,2,3)`: streamwise `x` is
absent, wall-normal `y` occupies storage dimension 1, spanwise `z` occupies
dimension 2, and `t` occupies dimension 3. Coordinate names never change
meaning, so `ddy!` applies the wall-normal finite-difference operator, `ddz!`
applies the spanwise Fourier derivative, and `ddx!` refers to the absent
streamwise direction. `CartesianPrimitive2D3C` is defined specifically on this physical `(y,z)`
plane.

Functions passed to `Field` or `VectorField` receive four arguments in physical
`(x,y,z,t)` order. The absent streamwise coordinate is `nothing`:

```julia
profile = Field(grid, (::Nothing, y, z, t) ->
                      (1 - y^2) * cos(0.5z) * cos(2π*t))
```

The state ordering is `(v,w,u)`: the wall-normal and spanwise in-plane
velocities followed by the streamwise velocity. The default Couette base is
`(nothing,nothing,U)`, where

```math
U(\eta)=\eta, \qquad
\eta=2\frac{y-y_-}{y_+-y_-}-1.
```

Thus `U` reaches `-1` and `1` at the walls for any valid `lim=(y₋,y₊)`.
[`plane_couette_base`](@ref) returns this normalized profile.

## Rotation

`Ro=2Ωh/Uw` is the rotation number about the physical spanwise axis. The
forward Coriolis block is

```text
out[1] -= Ro*u[3]
out[3] += Ro*u[1]
```

and both signs reverse in continuous- and discrete-adjoint modes. `Ro=0`
installs `NoForce`; otherwise the shared `CoriolisForce` uses
`components=(3,1)` to match `(v,w,u)` state order.

The same grid also accepts `PlanePoiseuilleFlow`. Its Poiseuille base and
constant pressure-gradient force occupy streamwise component three, while the
layout and Coriolis pair remain unchanged.

For MPI decomposition, the finite-difference direction is the physical
wall-normal direction and is therefore selected with
`decomposed_physical_dims=(:y,)`.

## Boundary conditions

The grid includes wall points but does not prescribe the wall velocities.
The Couette base carries the moving-wall values, while the perturbation basis
or residual must impose homogeneous no-slip conditions at both walls.
