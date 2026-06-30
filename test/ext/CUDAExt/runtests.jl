using CUDA,
      FDGrids

import CUDA: i32

# utility function to check if CUDA.jl functions on this system
function cuda_available()
    try
        CUDA.functional()
    catch
        false
    end
end

const CUDAExt = Base.get_extension(NSEBase, :CUDAExt)

# construct mock grid used in tests
Nx = 15; Ny = 16; Nz = 15; Nt = 15;
g = MockChannelGrid(Ny, Nx, Nz, Nt)

if cuda_available()
    include("test_utils.jl")
    include("test_gpugrid.jl")
    include("test_gpufields.jl")
    include("test_fft.jl")
    include("test_derivatives.jl")
    include("test_galerkin.jl")
    include("test_dot.jl")
else
    @warn "Skipping GPU tests - CUDA not functional"
    @test_broken false
end
