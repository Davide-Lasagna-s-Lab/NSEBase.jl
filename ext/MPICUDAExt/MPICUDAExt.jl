module MPICUDAExt

import CUDA,
       Adapt,
       MPI,
       FDGrids,
       HaloArrays,
       LinearAlgebra,
       NSEBase

# get extension namespaces
const MPIExt  = Base.get_extension(NSEBase, :MPIExt)
const CUDAExt = Base.get_extension(NSEBase, :CUDAExt)

const DecomposedGPUGrid{T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S} = MPIExt.DecomposedGrid{T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S, GP} where {GP<:CUDAExt.GPUGrid}
NSEBase.FFTPlanStyle(::Type{<:DecomposedGPUGrid}) = CUDAExt.cuFFTStyle()

struct DecomposedDeviceGrid{T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S, GP, W, P} <: NSEBase.AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}
        parent::GP
       weights::W
    inh_points::P

    DecomposedDeviceGrid(gp::GP,
                         DDIMS, NHALO, S,
                         weights::W,
                      inh_points::P) where {
                    T, D, AXES, FFT_DIMS_ORDER,
                    GP<:CUDAExt.GPUGrid{T, D, AXES, FFT_DIMS_ORDER}, W, P} =
        new{T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S, GP, W, P}(gp, weights, inh_points)
end

function Adapt.adapt_structure(to, g::DecomposedGPUGrid{T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S}) where {T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S}
    g_d          = Adapt.adapt_structure(to, parent(g))
    ws_d         = Adapt.adapt_structure(to, NSEBase.weights(g))
    inh_points_d = Adapt.adapt_structure(to, g.inh_points)
    return DecomposedDeviceGrid(g_d, DDIMS, NHALO, S, ws_d, inh_points_d)
end

function CUDA.cu(g::MPIExt.DecomposedGrid{T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S, GP, W, P, COMM}) where {T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S, GP, W, P, COMM}
    g_d          = CUDA.cu(parent(g))
    ws_d         = CUDA.cu(NSEBase.weights(g))
    inh_points_d = CUDA.cu(g.inh_points)
    return MPIExt.DecomposedGrid(g_d, DDIMS, NHALO, S, ws_d, inh_points_d, g.comm)
end

NSEBase.FTField(g::DecomposedGPUGrid{T}) where {T} =
    NSEBase.FTField(g, CUDA.cu(HaloArrays.HaloArray{Complex{T}}(MPIExt.comm(g),
                                                                MPIExt.local_transform_size(g),
                                                                MPIExt.nhalo(g); economic=true)))

NSEBase.Field(g::DecomposedGPUGrid{T}; dealias::Bool=false) where {T} =
    NSEBase.Field(g, CUDA.cu(HaloArrays.HaloArray{T}(MPIExt.comm(g),
                                                     MPIExt.local_physical_size(g; dealias=dealias),
                                                     MPIExt.nhalo(g); economic=true)))


function NSEBase.project!(a::NSEBase.ProjectedField{G}, u::NSEBase.VectorField{N, <:NSEBase.FTField{G}}) where {N, G<:DecomposedGPUGrid}
    NSEBase.project!(a, u, CUDAExt.project_method(a, u))

    # Sum the per-rank partial projections into the global modal coefficients
    MPI.Allreduce!(parent(a), MPI.SUM, comm(NSEBase.grid(u)))
    return a
end

NSEBase.expand!(u::NSEBase.VectorField{N, <:NSEBase.FTField{G}}, a::NSEBase.ProjectedField{G}) where {N, G<:DecomposedGPUGrid} =
    NSEBase.expand!(u, a, CUDAExt.expand_method(u, a))

end

# TODO: test derivatives
# TODO: test galerkin methods
# TODO: test operators are consistent (produce the same result as the CPU decomposed operator)
