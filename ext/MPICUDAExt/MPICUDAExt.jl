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

const DecomposedGPUGrid{T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S} = MPIExt.DecomposedGrid{T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S, GP} where {GP<:GPUGrid}
FFTPlanStyle(::Type{<:DecomposedGPUGrid}) = CUDAExt.cuFFTStyle()

struct DecomposedDeviceGrid{T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S, GP, W, P} <: NSEBase.AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}
        parent::GP
       weights::W
    inh_points::P

    DecomposedDeviceGrid{DDIMS, NHALO, S}(gp::GP,
                                     weights::W,
                                  inh_points::P) where {
                                T, D, AXES, FFT_DIMS_ORDER,
                                DDIMS, NHALO, S,
                                GP<:CUDAExt.GPUGrid{T, D, AXES, FFT_DIMS_ORDER}, W, P} =
        new{T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S, GP, W, P}(gp, weights, inh_points)
end

function Adapt.adapt_structure(to, g::DecomposedGPUGrid{T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S})
    g_d          = Adapt.adapt_structure(to, parent(g))
    ws_d         = Adapt.adapt_structure(to, NSEBase.weights(g))
    inh_points_d = Adapt.adapt_structure(to, g.inh_points)
    return DecomposedDeviceGrid{DDIMS, NHALO, S}(g_d, ws_d, inh_points_d)
end

function CUDA.cu(g::MPIExt.DecomposedGrid{T, D, AXES, FFT_DIMS_ORDER, DDIMS, NHALO, S, GP, W, P, COMM})
    g_d          = CUDA.cu(parent(g))
    ws_d         = CUDA.cu(NSEBase.weights(g))
    inh_points_d = CUDA.cu(g.inh_points)
    return MPIExt.DecomposedGrid{DDIMS, NHALO, S}(g_d, ws_d, inh_points_d, g.comm)
end

NSEBase.FTField(g::DecomposedGPUGrid{T}) where {T} =
    NSEBase.FTField(g, CUDA.cu(HaloArrays.HaloArray{Complex{T}}(comm(g),
                                                                local_transform_size(g),
                                                                nhalo(g); economic=true)))

NSEBase.Field(g::DecomposedGPUGrid{T}; dealias::Bool=false) where {T} =
    NSEBase.Field(g, CUDA.cu(HaloArrays.HaloArray{T}(comm(g),
                                                     local_physical_size(g; dealias),
                                                     nhalo(g); economic=true)))


function NSEBase.project!(a::NSEBase.ProjectedField{G}, u::NSEBase.VectorField{N, <:NSEBase.FTField{G}}) where {G<:DecomposedGPUGrid}
    NSEBase.project!(a, u, project_method(a, u))

    # Sum the per-rank partial projections into the global modal coefficients
    MPI.Allreduce!(parent(a), MPI.SUM, comm(NSEBase.grid(u)))
    return a
end

NSEBase.expand!(u::NSEBase.VectorField{N, <:NSEBase.FTField{G}}, a::NSEBase.ProjectedField{G}) where {G<:DecomposedGPUGrid} =
    NSEBase.expand!(u, a, expand_method(u, a))



# ! I need the projection to route through the _project_component! method for all cases.


# ! could also use trait dispatch in the project! and expand! dispatch paths to improve code reusability?
abstract type ProjectStyle end
struct DefaultProjectStyle <: ProjectStyle end
struct DecomposedProjectStyle <: ProjectStyle end
struct CUDAProjectStyle <: ProjectStyle end
ProjectStyle(::Type{<:AbstractGrid}) = DefaultProjectStyle()
ProjectStyle(::Type{<:DecomposedGrid}) = DecomposedProjectStyle()
ProjectStyle(::Type{<:GPUGrid}) = CUDAProjectStyle()

project!(a::ProjectedField{G}, u::VectorField{N, <:FTField{G}}) where {N, G} =
    project!(a, u, project_method(ProjectStyle(G), a, u))

function project!(a, u, method::ProjectMethod) end

function project_method(::DefaultProjectStyle, a, u) end
function project_method(::DecomposedProjectStyle, a, u) end
function project_method(::CUDAProjectStyle, a, u) end




end

# TODO: test derivatives
# TODO: test galerkin methods
# TODO: test FFT plans (construction and execution)
# TODO: test operators are consistent (produce the same result as the CPU decomposed operator)
