module MPICUDAExt

import CUDA,
       Adapt,
       MPI,
       FDGrids,
       HaloArrays,
       LinearAlgebra

# get extension namespaces
const MPIExt  = Base.get_extension(NSEBase, :MPIExt)
const CUDAExt = Base.get_extension(NSEBase, :CUDAExt)

# ! The core of this extension is the convenience constructor for a distributed grid stored
# ! on the device. This is achieved with methods such as
# !     - CUDA.cu(::DecomposedGrid)
# !     - NSEBase.distributed(::AbstractGrid, ::MPI.COMM, args..., device::Bool)
# ! 
# ! This results in a distributed grid stored on the device, which can then be used for
# ! specialised dispatch for constructors and fluid computations (derivatives, galerkin, etc.).
# !
# ! Required components for this to work are:
# !     - FDGridsCUDAExt: mul! method that uses rng and g_first to use kernels for only a
# !                       subset of the total (Adjoint)DiffMatrix.
# !     - HaloArrays: A HaloArrayCUDAExt extension that ensures correct construction, adaptation,
# !                   and any other methods that fall back to CUDA routines. Need to make sure
# !                   broadcasting and FFT plans work properly (and are benchmarked for speed).
# !     - NSEBase.project!: Define this method using CUDA kernels, or get a workaround
# !                         to avoid it if desired.
# !     - FTField, Field, FFTPlans: Constructors to get the correct backend and cache sizes, and
# !                                 data types.

const DecomposedGPUGrid{T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S} = DecomposedGrid{T, D, AXES, FFT_DIMS_ORDER, GP} where {GP<:GPUGrid}


# TODO: this
function Adapt.adapt_structure(to, g::DecomposedGPUGrid)
    throw(error("needs to be implemented"))
end

function CUDA.cu(g::MPIExt.DecomposedGrid{T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S, GP, W, P, COMM})
    g_d = CUDA.cu(parent(g))
    ws_d = CUDA.cu(NSEBase.weights(g))
    inh_points_d = CUDA.cu(g.inh_points)
    return MPIExt.DecomposedGrid{T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S}(g_d, g.comm, ws_d, inh_points_d)
end

NSEBase.distributed(g::NSEBase.AbstractGrid, comm::MPI.COMM, device::Bool; kwargs...) =
    device ? CUDA.cu(NSEBase.distributed(g, comm; kwargs...)) : NSEBase.distributed(g, comm; kwargs...)

# ! allows the user to call NSEBase.distributed(::GPUGrid, ...) directly
function MPIExt._local_inhomogeneous_points(g::CUDAExt.GPUGrid, cart)
    parent_points = NSEBase.points(g)
    return map(NSEBase.inhomogeneous_storage_dims(g)) do stor_dim
        global_vec = CuVector(vec(parent_points[stor_dim]))
        axis_inds  = _local_axis_indices(length(global_vec), stor_dim, cart)
        axis_inds isa Colon ? global_vec : global_vec[axis_inds]
    end
end



# TODO: requires CUDA extension on HaloArrays.jl
NSEBase.FTField(g::DecomposedGPUGrid{T}) where {T} =
    NSEBase.FTField(g, HaloArrays.HaloArray{Complex{T}}(comm(g), 
                                                        local_transform_size(g), 
                                                        nhalo(g); economic=true,
                                                                  device=true))

NSEBase.Field(g::DecomposedGPUGrid{T}) where {T} =
    NSEBase.Field(g, HaloArrays.HaloArray{T}(comm(g), 
                                             local_physical_size(g; dealias), 
                                             nhalo(g); economic=true,
                                                       device=true))

NSEBase.FFTPlans(g::DecomposedGPUGrid{T}; kwargs...) where {T} =
    _make_cufft_plans(NSEBase._fft_size(g), NSEBase.fft_storage_dims(g), T; kwargs...)



function NSEBase.project!(a::ProjectedField{G}, u::VectorField{N, <:FTField{G}}) where {G<:DecomposedGPUGrid} end

end
