module NSEBase

using LinearAlgebra, FFTW, JLD2
import FDGrids

export FFTW

export AbstractGrid
export RectangularGrid
export AbstractChannelGrid, AbstractChannel2D3CGrid, AbstractChannel3DGrid, AbstractSquareDuctGrid
export AbstractLidDrivenCavityGrid, AbstractLidDrivenCavity3DGrid
export ChannelGrid, SquareDuctGrid, LidDrivenCavityGrid
export CHANNEL_2D3C_AXES, CHANNEL_2D3C_FFT_ORDER, CHANNEL_2D3C_INHOMOGENEOUS_DIMS
export CHANNEL_3D_AXES, CHANNEL_3D_FFT_ORDER, CHANNEL_3D_INHOMOGENEOUS_DIMS
export SQUARE_DUCT_AXES, SQUARE_DUCT_FFT_ORDER, SQUARE_DUCT_INHOMOGENEOUS_DIMS
export LID_DRIVEN_CAVITY_2D_AXES, LID_DRIVEN_CAVITY_2D_FFT_ORDER, LID_DRIVEN_CAVITY_2D_INHOMOGENEOUS_DIMS
export LID_DRIVEN_CAVITY_3D_AXES, LID_DRIVEN_CAVITY_3D_PERIODIC_FFT_ORDER, LID_DRIVEN_CAVITY_3D_BOUNDED_FFT_ORDER
export storage_dim, physical_dim, physical_to_storage_dim, to_storage_order
export rfft_storage_dim, rfft_physical_dim
export points, growto, weights
export fft_storage_dims, fft_physical_dims
export spatial_fft_storage_dims, spatial_fft_physical_dims
export inhomogeneous_storage_dims, inhomogeneous_physical_dims
export spatial_inhomogeneous_storage_dims, spatial_inhomogeneous_physical_dims
export transform_size, fft_norm, wavenumber_scale
export WaveNumberVector, to_homogeneous_indices, to_wavenumber_vector
export FTField, Field, VectorField, grid
export add_base_flow!
export FFTPlans, FFT, IFFT
export ProjectedField, modes, project!, project, expand!, expand
export LoopGalerkin, GemmGalerkin
export dd!, ddx!, ddy!, ddz!, ddt!
export inhomogeneous_laplacian!, add_homogeneous_laplacian!, laplacian!
export shift!, shift, normdiff, minnormdiff
export save_grid, load_grid, save_field, load_field
export FarazmandWeight
export CartesianPrimitive3DNSE, CartesianPrimitive3DLNSE
export CartesianPrimitive2DNSE, CartesianPrimitive2DLNSE
export CartesianPrimitive2D3CNSE, CartesianPrimitive2D3CLNSE
export CartesianPrimitive3DBoussinesqNSE, CartesianPrimitive3DBoussinesqLNSE
export Forward, AdjointContinuous, AdjointDiscrete, NoForce, CompoundForcing, Mode
export construct_equations, CartesianPrimitive3D, CartesianPrimitive2D, CartesianPrimitive2D3C, PolarPrimitive
export CartesianPrimitive3DBoussinesq
export ncomp, cache_length, nonlinear_operator, linearised_operator
export ProjectedNSE
export PlaneCouetteFlow, PlanePoiseuilleFlow, SquareDuctFlow, LidDrivenCavityFlow
export RayleighBenardFlow
export plane_couette_base, plane_poiseuille_base, rpcf_base, rbc_base_temperature
export CoriolisForce, ConstantBodyForce

include("notimplementederror.jl")
include("grids/abstractgrid.jl")
include("wavenumbervector.jl")
include("ftfield.jl")
include("field.jl")
include("vectorfield.jl")
include("fft.jl")
include("projectedfield.jl")
include("galerkin.jl")
include("shifts.jl")
include("norms.jl")
include("weighting.jl")
include("broadcasting.jl")
include("derivatives.jl")

include("grids/rectangular.jl")
include("io.jl")
include("equations/types.jl")
include("cases/forcings.jl")
include("equations/cartesianprimitive_3d.jl")
include("equations/cartesianprimitive_2d.jl")
include("equations/cartesianprimitive_2d3c.jl")
include("equations/cartesianprimitive_3d_boussinesq.jl")
include("equations/projectednse.jl")
include("equations/shared.jl")
include("cases/channel.jl")
include("cases/square_duct.jl")
include("cases/lid_driven_cavity.jl")

# dummy function definition for MPI extension
export distributed

function derivative_matrix end

"""
    distributed(grid, comm; decomposed_physical_dims, nprocesses, nhalo)

Create an MPI-decomposed wrapper around `grid`. Loading MPI, FDGrids, and
HaloArrays activates the implementation. Only inhomogeneous spatial directions
can currently be decomposed; `nprocesses` and `nhalo` contain one entry per
selected physical direction.
"""
function distributed end

end
