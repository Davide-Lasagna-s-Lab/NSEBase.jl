# NSEBase.jl

NSEBase is the shared numerical foundation for wall-bounded Navier–Stokes
solvers. It combines Fourier directions with FDGrids collocation directions,
provides physical and spectral field types, and assembles nonlinear, linearised,
continuous-adjoint, and discrete-adjoint primitive-variable equations.

The package now includes concrete rectangular grids and complete case
constructors. Users can build full, streamwise-invariant, and two-dimensional channels, square
ducts, cavities, and Rayleigh–Bénard systems without installing a geometry-specific grid package.

## First example

```julia
using NSEBase

grid = ChannelGrid(9, 17, 9; Nt=1, α=0.5, β=0.5, width=5)
equations = PlanePoiseuilleFlow(grid, 500; f=1, fftw_flags=FFTW.ESTIMATE, dealias=false)
```

This creates a `(y,x,z,t)` grid with Gauss–Lobatto wall-normal points,
quadrature-weighted finite-difference adjoints, and Fourier domains of length
`4π` in both homogeneous spatial directions. The unit-period temporal
coordinate uses wavenumber scale `2π`.

## What to read next

- [Concepts and conventions](guide.md) explains storage order, transforms,
  derivatives, inner products, de-aliasing, and equation assembly.
- [Rectangular grids](rectangular_grids.md) documents the shared concrete grid,
  its ownership contract, and custom construction.
- The case manual gives geometry-specific defaults, examples, and physical
  conventions.
- [API reference](api.md) contains every public docstring.

```@contents
Depth = 2
```
