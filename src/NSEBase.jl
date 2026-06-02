module NSEBase

# TODO: add benchmark scripts, profile the cartesian primitive NSE and LNSE implementations

using LinearAlgebra, FFTW, JLD2

export FFTW

export GridDecomposition, Undecomposed, Decomposed
export AbstractGrid, decomposition_dims, ndecomposed_dims
export points, growto, weights, fft_dims, inhomogeneous_dims, spatial_inhomogeneous_dims, to_storage_order
export transform_size, fft_norm, wavenumber_scale
export WaveNumberVector, to_homogeneous_indices, to_wavenumber_vector
export FTField, Field, VectorField, grid
export add_base_flow!
export FFTPlans, FFT, IFFT
export ProjectedField, modes, project!, project, expand!, expand
export LoopGalerkin, GemmGalerkin
export ddx!, ddx_1!, ddx_2!, ddx_3!, ddx_4!
export inhomogeneous_laplacian!, add_homogeneous_laplacian!, laplacian!
export shift!, shift, normdiff, minnormdiff
export save_grid, load_grid, save_field, load_field
export FarazmandWeight
export CartesianPrimitive3DNSE, CartesianPrimitive3DLNSE
export CartesianPrimitive2DNSE, CartesianPrimitive2DLNSE
export Forward, AdjointContinuous, AdjointDiscrete, NoForce, CompoundForcing, Mode
export construct_equations, CartesianPrimitive3D, CartesianPrimitive2D, PolarPrimitive
export ncomp, cache_length, nonlinear_operator, linearised_operator
export ProjectedNSE

include("notimplementederror.jl")
include("abstractgrid.jl")
include("wavenumbervector.jl")
include("spectralloops.jl")
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
include("equations/cartesianprimitive_3d.jl")
include("equations/cartesianprimitive_2d.jl")
include("equations/projectednse.jl")
include("equations/shared.jl")

end
