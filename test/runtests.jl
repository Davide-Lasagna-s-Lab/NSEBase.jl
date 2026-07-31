using Test

using FFTW
using LinearAlgebra
using NSEBase
using Random
using ReSolverRectangularGrids

include("support/rectangular_fixtures.jl")
include("support/field_fixtures.jl")
using .TestRectangularFixtures

@testset verbose=true "NSEBase                                                     " begin
    # These tests exercise NSEBase's public storage and container contracts.
    # Concrete-grid accuracy belongs to ReSolverRectangularGrids and is not
    # duplicated here.
    @testset verbose=true "Public interface contracts                                  " begin
        include("interface/notimplementederror.jl")
        include("interface/abstractgrid.jl")
        include("interface/wavenumbervector.jl")
        include("interface/axis_utils.jl")
        include("interface/broadcasting.jl")
        include("interface/field.jl")
        include("interface/ftfield.jl")
        include("interface/vectorfield.jl")
        include("interface/projectedfield.jl")
    end

    @testset verbose=true "Reusable numerical functionality                            " begin
        include("functionality/spectral_sanitation.jl")
        include("functionality/fft.jl")
        include("functionality/derivatives.jl")
        include("functionality/shifts.jl")
        include("functionality/norms.jl")
        include("functionality/weighting.jl")
        include("functionality/galerkin.jl")
        include("functionality/io.jl")
        include("functionality/equations.jl")
    end

    @testset verbose=true "Allocation and performance contracts                        " begin
        include("performance/allocations.jl")
    end

    @testset verbose=true "Package extensions                                          " begin
        # MPI tests run each program in a fresh subprocess and check its exit
        # status; CUDA tests are skipped explicitly when no device is present.
        include("ext/MPIExt/runtests.jl")
        include("ext/CUDAExt/runtests.jl")
    end
end
