using Test

using FFTW
using LinearAlgebra
using NSEBase

# Shared test fixtures: `fake.jl` defines `FakeGrid` (a 2-D grid with one
# inhomogeneous and one rfft dimension), `test_grids.jl` defines
# `TripleGrid` (a 3-D grid with one inhomogeneous, one rfft and one signed
# FFT dimension) for tests that need multiple homogeneous directions.
include("fake.jl")
include("test_grids.jl")

# Generic interface / utility tests — exercise every public function in
# NSEBase against the documented contract, not its implementation.
include("test_notimplementederror.jl")
include("test_abstractgrid.jl")
include("test_wavenumbervector.jl")
include("test_broadcasting.jl")
include("test_field.jl")
include("test_ftfield.jl")
include("test_vectorfield.jl")
include("test_projectedfield.jl")
include("test_fields.jl")
include("test_fftplans.jl")
include("test_derivatives.jl")
include("test_shifts.jl")
include("test_norms.jl")
include("test_weighting.jl")
include("test_galerkin.jl")
include("test_io.jl")

# Integration tests for the bundled equations module.
include("test_operators.jl")

# Allocation tests — check that no unexpected allocations occur
# include("test_allocations.jl")