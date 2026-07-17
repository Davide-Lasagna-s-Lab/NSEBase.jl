# Lid-driven cavities

`LidDrivenCavityGrid` constructs both 2D and 3D rectangular cavities. The
number of positional resolutions is the number of spatial dimensions; optional
time or phase always uses the `Nt` keyword.

## Two-dimensional cavity

```julia
using NSEBase

grid = LidDrivenCavityGrid(65, 49; Nt=1, xlim=(-1,1), ylim=(0,1), width=7)
```

The storage order is `(x,y,t)`. Both spatial directions use FDGrids, while
unit-period time is the only Fourier direction. `Nt=1` gives a steady grid.

## Three-dimensional cavity

```julia
bounded = LidDrivenCavityGrid(65, 49, 33; spanwise=:bounded, width=7)
periodic = LidDrivenCavityGrid(65, 49, 31; spanwise=:periodic, Lz=2, width=7)
```

Both layouts are stored as `(x,y,z,t)`. With `spanwise=:bounded`, `z` has its
own FDGrids points and operators. With `spanwise=:periodic`, `z` is Fourier
with physical period `Lz`. The word “bounded” describes the discretisation;
the grid itself does not impose no-slip values.

## Required base lifting

`LidDrivenCavityFlow` requires an explicit `base`: `(U,V)` in 2D or `(U,V,W)`
in 3D. This prevents the equation constructor from silently choosing a zero
reference state that does not carry the moving-lid boundary values.

For example, the following streamfunction produces a divergence-free 2D
lifting with a smooth lid velocity that vanishes at the upper corners:

```julia
X, Y, _ = points(grid)
U = @. 16X^2 * (1-X)^2 * (3Y^2 - 2Y)
V = @. -32X * (1-X) * (1-2X) * Y^2 * (Y-1)
equations = LidDrivenCavityFlow(grid, 1000; base=(U,V), fftw_flags=FFTW.ESTIMATE)
```

The lifting carries the inhomogeneous lid values. The perturbation basis or
residual must enforce homogeneous conditions on the walls. For MPI-decomposed
grids, base arrays use the local inhomogeneous shape owned by each rank.

All bounded directions may independently select their interval, distribution,
and stencil width. Product quadrature and weighted adjoints follow the standard
[`RectangularGrid`](@ref) contract.
