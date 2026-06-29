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
