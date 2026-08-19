using Test

import MPI
import FDGrids
import CUDA
import Adapt

using NSEBase

MPICUDAExt = Base.get_extension(NSEBase, :MPICUDAExt)

include("../../mock_channel_grid.jl")

# setup MPI environment
MPI.Initialized() || MPI.Init()
comm   = MPI.COMM_WORLD
nranks = MPI.Comm_size(comm)
rank   = MPI.Comm_rank(comm)

# Pick a wall-normal size that divides cleanly across any number of ranks
# used in CI (we test with 1 and 4 in `runtests.jl`).
const Ny = 16; const Nx = 7; const Nz = 9; const Nt = 5
const NHALO = 1

# construct grid
gp = MockChannelGrid(Ny, Nx, Nz, Nt)
g  = distributed(gp, comm; decomposed_physical_dims=(:y),
                           nprocesses              =(nranks,),
                           nhalo                   =(NHALO,))

Ny_local = Ny ÷ nranks
y_offset = rank * Ny_local

@testset "Decomposed CUDA grid                                                " begin
    gd = @test_nothrow CUDA.cu(g)
    @test gd isa MPICUDAExt.DecomposedGPUGrid{Float32, 4, (2, 1, 3, 4), (2, 3, 4), (1,), (1,), (16, 7, 9, 5), <:GPUGrid}

    gd_adapt = @test_nothrow Adapt.adapt(CUDA.KernelAdaptor(), gd)
    @test gd_adapt isa MPICUDAExt.DecomposedDeviceGrid{Float32, 4, (2, 1, 3, 4), (2, 3, 4), (1,), (1,), (16, 7, 9, 5), <:GPUGrid}
    @test gd_adapt.D₁ isa DiffMatrix{Float32, 3, true, <:CuArray}
    @test gd_adapt.D₂ isa DiffMatrix{Float32, 3, true, <:CuArray}
    @test isbits(gd_adapt)
end

@testset "Decomosed fields                                                    " begin
    @testset "FTField" begin
        # constructor
        # adaptation/CUDA.cu of normal decomposed fields
    end

    @testset "Field" begin
        # constructor
        # adaptation/CUDA.cu of normal decomposed fields
    end
end
