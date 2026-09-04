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

function NSEBase.ProjectedField(grid::DecomposedGPUGrid{T}, modes) where {T}
    Nm = size(modes[1], 1)
    return NSEBase.ProjectedField(grid,
                                  CUDA.zeros(Complex{T}, Nm,
                                        NSEBase.transform_size(grid)[collect(NSEBase.fft_storage_dims(grid))]...),
                                  modes)
end


NSEBase._spectral_dd!(out::F,
                        u::F,
                         ::Val{STORAGE_DIM},
                     mode::NSEBase.OperatorMode=NSEBase.Forward()) where {
                    STORAGE_DIM,
                    G<:DecomposedGPUGrid,
                    F<:Union{NSEBase.FTField{G}, NSEBase.ProjectedField{G}}} =
    CUDAExt._cuda_spectral_dd!(out, u, Val(STORAGE_DIM), Val(NSEBase.rfft_storage_dim(NSEBase.grid(u))), mode)

NSEBase._add_homogeneous_laplacian!(out::F, u::F) where {F<:NSEBase.FTField{<:DecomposedGPUGrid}} =
    CUDAExt._cuda_add_homogeneous_laplacian!(out, u)


function NSEBase.project!(a::NSEBase.ProjectedField{G},
                          u::NSEBase.VectorField{N, <:NSEBase.FTField{G}},
                     method::CUDAExt.ProjectMethod=CUDAExt.project_method(a, u)) where {N, G<:DecomposedGPUGrid}
    CUDAExt._project!(a, u, method)

    # Sum the per-rank partial projections into the global modal coefficients
    CUDA.synchronize() # make sure computation is finished on all processes
    MPI.Allreduce!(parent(a), MPI.SUM, MPIExt.comm(NSEBase.grid(u)))
    return a
end

NSEBase.expand!(u::NSEBase.VectorField{N, <:NSEBase.FTField{G}},
                a::NSEBase.ProjectedField{G},
           method::CUDAExt.ExpandMethod=CUDAExt.expand_method(u, a)) where {N, G<:DecomposedGPUGrid} =
    CUDAExt._expand!(u, a, method)

end
