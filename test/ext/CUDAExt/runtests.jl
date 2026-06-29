using CUDA

# utility function to check if CUDA.jl functions on this system
function cuda_available()
    try
        CUDA.functional()
    catch
        false
    end
end

if !cuda_available()
    @warn "Skipping GPU tests - CUDA not functional"
    @test_broken false
else
    include("test_utils.jl")
    include("test_gpugrid.jl")
    include("test_gpufields.jl")
    include("test_fft.jl")
    include("test_derivatives.jl")
    include("test_galerkin.jl")
    include("test_dot.jl")
end
