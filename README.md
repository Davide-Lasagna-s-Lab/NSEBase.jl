# NSEBase.jl

NSEBase provides the grids, spectral/finite-difference fields, transforms,
derivatives, Galerkin projection, and primitive-variable Navier–Stokes operators
used by the ReSolver ecosystem.

Rectangular wall-bounded cases are included directly. Separate geometry
packages are no longer required for:

- plane Couette and plane Poiseuille flow;
- streamwise-invariant rotating plane Couette flow;
- two-dimensional plane Poiseuille flow;
- square-duct flow;
- lid-driven cavities;
- Rayleigh–Bénard convection.

All of these cases use one concrete `RectangularGrid{NI}` backed by FDGrids.
Case factories and structural aliases hide axis mappings, FFT order, differentiation
matrices, quadrature weights, and weighted adjoints behind physical parameters
and documented defaults.

## Installation

NSEBase is not yet registered. Install it directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/Davide-Lasagna-s-Lab/FDGrids.jl")
Pkg.add(url="https://github.com/Davide-Lasagna-s-Lab/NSEBase.jl")
```

FDGrids is installed explicitly because it is also currently unregistered.

## Quick start

```julia
using NSEBase

# (Nx, Ny, Nz), with Nt, α=2π/Lx, and β=2π/Lz as keywords.
grid = ChannelGrid(9, 17, 9; Nt=1, α=0.5, β=0.5, width=5)

couette = PlaneCouetteFlow(grid, 500; Ro=0.1, fftw_flags=FFTW.ESTIMATE, dealias=false)
poiseuille = PlanePoiseuilleFlow(grid, 500; f=1, fftw_flags=FFTW.ESTIMATE, dealias=false)
```

`grid` stores arrays as `(y,x,z,t)`, builds the wall-normal FDGrids operators
and quadrature-weighted adjoints, and uses Fourier periods `2π/α`, `2π/β`,
and `1`. The flow constructors return `ProjectedNSE` operator bundles.

Dimensionally reduced channels use constructors that state their physical formulation explicitly:

```julia
streamwise_invariant = StreamwiseInvariantChannelGrid(17, 9; β=0.5, width=5)
two_dimensional = TwoDimensionalChannelGrid(9, 17; α=0.5, width=5)
```

The first is a wall-normal–spanwise 2D3C grid with state `(v,w,u)`. The second
is a streamwise–wall-normal 2D grid: hydrodynamic cases use `(u,v)`, while
Rayleigh–Bénard convection uses `(u,v,θ)`.

See [`examples/`](examples/) for runnable constructors for every bundled case
and [`docs/`](docs/) for conventions, custom-grid construction, boundary-condition
contracts, MPI decomposition, and the complete API.

## Boundary conditions

Grid distributions include boundary points when the selected FDGrids rule does,
but NSEBase does not impose wall values. No-slip and thermal conditions belong
to the projection basis or residual assembly. For lid-driven cavities, the
required base lifting carries the inhomogeneous moving-lid values while the
perturbation basis or residual enforces homogeneous wall conditions. Case
constructors do not silently alter boundary degrees of freedom.

## Custom geometries

For a new Cartesian geometry, construct `RectangularGrid` from tuples of
collocation vectors, FDGrids matrices, weighted adjoints, and quadrature weights.
For a non-rectangular discretisation, subtype
`AbstractGrid{T,D,AXES,FFT_DIMS_ORDER}` and implement the documented grid
interface.
