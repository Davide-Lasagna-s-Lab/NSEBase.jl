using Test

using FFTW
using HCubature
using LinearAlgebra
using NSEBase
import FDGrids

# Test support contains constructor functions and numerical references only.
# Production-grid behavior is never supplied by a test-only grid subtype.
include("helpers/rectangular.jl")
include("helpers/case_contracts.jl")

@testset verbose=true "Implementation contracts                                    " begin
    include("implementation/errors/notimplementederror.jl")
    include("implementation/grids/abstractgrid.jl")
    include("implementation/grids/rectangulargrid.jl")
    include("implementation/grids/wavenumbervector.jl")
    include("implementation/grids/axis_utils.jl")

    include("implementation/fields/broadcasting.jl")
    include("implementation/fields/field.jl")
    include("implementation/fields/ftfield.jl")
    include("implementation/fields/vectorfield.jl")
    include("implementation/fields/projectedfield.jl")
    include("implementation/fields/contracts.jl")

    include("implementation/transforms/fftplans.jl")
    include("implementation/transforms/shifts.jl")
    include("implementation/operators/derivatives.jl")
    include("implementation/operators/norms.jl")
    include("implementation/operators/weighting.jl")
    include("implementation/operators/galerkin.jl")
    include("implementation/equations/projected_nse.jl")
end

@testset verbose=true "Flow cases                                                  " begin
    include("cases/forcings.jl")
    include("cases/lid_driven_cavity.jl")
    include("cases/rpcf.jl")
    include("cases/rayleigh_benard.jl")
    include("cases/channel.jl")
    include("cases/square_duct.jl")
end

@testset verbose=true "Documentation and examples                                  " begin
    include("documentation/case_docstrings.jl")
    include("integration/examples.jl")
end

include("integration/io.jl")

# Performance contracts run before loading the MPI extension so extension
# initialization cannot perturb allocation baselines.
include("performance/allocations.jl")

# MPI programs execute in isolated subprocesses and intentionally run last.
include("ext/MPIExt/runtests.jl")
