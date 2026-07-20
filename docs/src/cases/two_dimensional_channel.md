# Two-dimensional plane Poiseuille flow

A two-dimensional channel contains only the physical streamwise–wall-normal `(x,y)` plane;
physical `z` and the spanwise velocity component are absent. Hydrodynamic Couette and Poiseuille
cases use `CartesianPrimitive2D` with velocity state `(u,v)`. Rayleigh–Bénard convection uses the
same grid with `CartesianPrimitive2DBoussinesq` and state `(u,v,θ)`.

```julia
using NSEBase

grid = TwoDimensionalChannelGrid(9, 17; Nt=1, α=0.5, width=5)
equations = PlanePoiseuilleFlow(grid, 500; f=1,
                                fftw_flags=FFTW.ESTIMATE, dealias=false)
```

The grid arguments are `(Nx,Ny)`. `α=2π/Lx` sets the streamwise period, `Nt`
sets the unit-period temporal or phase resolution, and `width` controls the
wall-normal FDGrids stencil. As in every channel layout, `lim=(y₋,y₊)` changes
the wall locations without changing the normalized base profile.
The equivalent compact positional form is
`TwoDimensionalChannelGrid(Nx,Ny,Nt,width,α)`.

## Layout and derivatives

Arrays are stored as `(y,x,t)`: wall-normal finite differences, streamwise
rFFT, and temporal FFT. Coordinate names retain their physical meanings in this
layout: `ddx!` is the streamwise Fourier derivative, `ddy!` is the wall-normal
finite-difference derivative, and `ddz!` refers to the absent spanwise
direction.

Functions passed to `Field` or `VectorField` receive four arguments in physical
`(x,y,z,t)` order. The absent spanwise coordinate is `nothing`:

```julia
profile = Field(grid, (x, y, ::Nothing, t) ->
                      (1 - y^2) * cos(α*x) * cos(2π*t))
```

## Pressure-driven equations

The default base is `(U,nothing)`, with

```math
U(\eta)=1-\eta^2, \qquad
\eta=2\frac{y-y_-}{y_+-y_-}-1.
```

The profile is zero at both walls and reaches one at the channel centre.
[`plane_poiseuille_base`](@ref) constructs it on the grid. `f` is a constant
streamwise body force applied to velocity component one at the zero `(x,t)`
Fourier mode; it represents the imposed pressure gradient in the periodic
streamwise direction.

`PlaneCouetteFlow` is available on the same grid when wall motion rather than
a pressure gradient drives the flow. It uses base `(U,nothing)` with
`U(η)=η`; the `(y,x,t)` layout and `(u,v)` state order are unchanged.

For the nondimensional equations used here, `U=1-η²` is an exact steady
balance on the canonical interval `[-1,1]` when `f=2/Re`. This relation is
useful when checking a basis or residual implementation.

`RayleighBenardFlow(grid,Re,Pr,Ra)` is also available on this layout. It uses base
`(nothing,nothing,Θ)` and places wall-normal buoyancy in component two; see
[Rayleigh–Bénard convection](rayleigh_benard.md).

## Boundary conditions

`TwoDimensionalChannelGrid` supplies wall points, quadrature, derivatives,
and weighted adjoints but does not impose no slip. The basis or residual must
enforce `u=v=0` at the walls. The constant body force acts in the domain and
does not replace those boundary conditions.
