# NSEBase.jl

[![codecov](https://codecov.io/gh/Davide-Lasagna-s-Lab/NSEBase.jl/branch/dev/graph/badge.svg)](https://codecov.io/gh/Davide-Lasagna-s-Lab/NSEBase.jl)

**NSEBase** is a Julia package that provides the shared spectral-field infrastructure
for Navier-Stokes solvers targeting wall-bounded flows.  It defines the grid
interface, all field types, FFT transforms, spectral derivative operators,
Galerkin projection utilities, and concrete Cartesian primitive-variable NSE
operators.  Downstream packages (e.g. *ChannelFlow.jl*) contribute a concrete
`AbstractGrid` subtype and obtain a fully-featured, allocation-free spectral
solver without reimplementing any of the above.

## Installation

NSEBase is not registered.  Install it directly from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/Davide-Lasagna-s-Lab/NSEBase.jl")
```

## Quick start

```julia
using NSEBase

# A concrete grid is provided by a downstream package, e.g. ChannelFlow.jl.
# Assuming `grid` is already constructed and `U` holds the laminar base flow:
obj = construct_equations(grid, Re, (U, nothing, nothing); flags = FFTW.MEASURE)

# Evaluate the nonlinear NSE residual projected onto a Galerkin basis
a   = ProjectedField(grid, randn(ComplexF64, M, Nk), modes)
rhs = obj(zero(a), a)          # rhs = P(Δu/Re − (u·∇)u)

# Evaluate the linearised operator around a given base state `b`
rhs_lin = obj(similar(a), a, b)
```

## Documentation

The [online documentation](https://Davide-Lasagna-s-Lab.github.io/NSEBase.jl/dev/)
contains the concepts and conventions guide together with the complete API
reference.

## Extending NSEBase

Implement a new grid by subtyping
`AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}` and defining the required interface
methods.

| Method | Purpose |
|--------|---------|
| `Base.size(grid)` | Physical-space array size |
| `points(grid; dealias)` | Collocation coordinate arrays |
| `wavenumber_scale(grid, dim)` | `2π/L` for a periodic coordinate of period `L` |
| `weights(grid)` | Quadrature weights for inhomogeneous dimensions |
| `derivative_matrix(grid, dim, Val(order), mode)` | Inhomogeneous first- and second-derivative operators |

See the [Concepts & Conventions](docs/src/guide.md) page for the full interface
contract and the assumptions NSEBase makes about grid geometry.
