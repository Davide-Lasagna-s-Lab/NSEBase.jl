module NSEBase

# TODO: add benchmark scripts, profile the cartesian primitive NSE and LNSE implementations

using LinearAlgebra, FFTW, JLD2

export FFTW

export AbstractGrid
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
export AdvectionForm, Advective, Divergence, Rotational
export AbstractNSE, CartesianPrimitiveNSE
export CartesianPrimitive2DNSE, CartesianPrimitive2D3CNSE, CartesianPrimitive3DNSE
export Mode, NonLinear, Forward, AdjointContinuous, AdjointDiscrete
export NoForce, CompoundForcing, BuoyancyForce
export construct_equations
export ProjectedNSE

include("notimplementederror.jl")
include("abstractgrid.jl")
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
include("io.jl")
include("equations/types.jl")
include("equations/forcings.jl")
include("equations/abstract_nse.jl")
include("equations/cartesianprimitive/operators.jl")
include("equations/cartesianprimitive/advection_2d.jl")
include("equations/cartesianprimitive/advection_2d3c.jl")
include("equations/cartesianprimitive/advection_3d.jl")
include("equations/projectednse.jl")

end
