module NSEBase

using LinearAlgebra, FFTW

export FFTW

export AbstractGrid, points, growto
export x_dim, y_dim, z_dim, t_dim, spatial_hom_dims, fft_dims, physical_side, fft_norm, transform_size
export FTField, Field, VectorField, grid
export FFTPlans, FFT, IFFT, padded_size
export ProjectedField, modes, project!, project, expand!, expand
export ModeNumber, _modenumber_to_projected_indices
export ProjectedNSE
export for_each_mode
export _quadrature_weight
export wavenumber_scale, ddx!, ddx_x!, ddx_y!, ddx_z!, add_homogeneous_laplacian!, laplacian!
export Mode, Forward, AdjointDiscrete, AdjointContinuous
export NoForce
export CartesianPrimitiveNSE, CartesianPrimitiveLNSE

include("notimplementederror.jl")
include("abstractgrid.jl")
include("ftfield.jl")
include("spectralloops.jl")
include("field.jl")
include("vectorfield.jl")
include("fft.jl")
include("modenumber.jl")
include("projectedfield.jl")
include("broadcasting.jl")
include("projectednse.jl")
include("derivatives.jl")
include("cartesianprimitive.jl")

end
