using CUDA,
      FDGrids

import CUDA: i32

# Return whether CUDA.jl can execute kernels on the current system.
function cuda_available()
    try
        CUDA.functional()
    catch
        false
    end
end

const CUDAExt = Base.get_extension(NSEBase, :CUDAExt)

# Production channel fixture used throughout the CUDA tests. Its layout retains
# the original four-dimensional coverage and projected-field rank.
const CUDA_NX = 15
const CUDA_NY = 16
const CUDA_NZ = 15
const CUDA_NT = 15
const CUDA_CHANNEL_GRID = channel_grid(Nx=CUDA_NX, Ny=CUDA_NY, Nz=CUDA_NZ, Nt=CUDA_NT)

function cuda_test_projected_field(g; ncomponents::Int=1, nmodes::Int=3, seed::Int=1)
    rng = MersenneTwister(seed)
    modes = test_modes(g; ncomponents, nmodes, seed=seed + 1)
    homogeneous_size = map(dim -> transform_size(g)[dim], fft_storage_dims(g))
    data = randn(rng, Complex{eltype(g)}, nmodes, homogeneous_size...)
    return ProjectedField(g, data, modes)
end

if cuda_available()
    include("test_utils.jl")

    # Device fields require the concrete grid package to implement Adapt's
    # structural conversion contract. Keep that downstream limitation visible
    # without replacing the production grid with a test-only subtype.
    grid_adapts = try
        CUDA.cu(CUDA_CHANNEL_GRID)
        true
    catch error
        error isa NSEBase.NotImplementedError || rethrow()
        false
    end

    if grid_adapts
        include("test_gpugrid.jl")
        include("test_gpufields.jl")
        include("test_fft.jl")
        include("test_derivatives.jl")
        include("test_dot.jl")
        include("test_galerkin.jl")
    else
        @warn "Skipping grid-backed CUDA tests: ReSolverRectangularGrids does not yet implement device adaptation"
        @testset verbose=true "Production grid CUDA adaptation unavailable                 " begin
            @test_broken false
        end
    end
else
    @warn "Skipping GPU tests - CUDA not functional"
    @testset verbose=true "CUDA device unavailable                                     " begin
        @test_broken false
    end
end
