module MPICUDAExt

import CUDA,
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

function CUDA.cu(g::MPIExt.DecomposedGrid) end

function NSEBase.distributed(g::NSEBase.AbstractGrid, comm::MPI.COMM, device::Bool; kwargs...) end
function NSEBase.distributed(g::CUDAExt.GPUGrid, comm::MPI.COMM; kwargs...) end

function NSEBase.FTField(g::DecomposedGPUGrid) end
function NSEBase.Field(g::DecomposedGPUGrid) end
function NSEBase.FFTPlans(g::DecomposedGPUGrid; kwargs...) end

function NSEBase.project!(a::ProjectedField{G}, u::VectorField{N, <:FTField{G}}) where {G<:DecomposedGPUGrid} end

end
