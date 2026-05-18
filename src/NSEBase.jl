module NSEBase

using LinearAlgebra, FFTW, JLD2

export FFTW

export AbstractGrid, AbstractCartesianGrid, points, growto, weights, fft_dims, inhomogeneous_dims
export x_dim, y_dim, z_dim, t_dim
export ModeNumber
export FTField, Field, VectorField, grid
export add_base_flow!
export FFTPlans, FFT, IFFT
export ProjectedField, modes, project!, project, expand!, expand, get_mode_coefficient
export shift!, shift, normdiff, minnormdiff
export save_field, load_field
export CartesianPrimitiveNSE, CartesianPrimitiveLNSE, Forward, AdjointContinuous, AdjointDiscrete, NoForce
export construct_equations, CartesianPrimitive, PolarPrimitive
export ProjectedNSE

include("notimplementederror.jl")
include("abstractgrid.jl")
include("modenumber.jl")
include("spectralloops.jl")
include("ftfield.jl")
include("field.jl")
include("vectorfield.jl")
include("fft.jl")
include("projectedfield.jl")
include("shifts.jl")
include("broadcasting.jl")
include("derivatives.jl")
include("io.jl")
include("equations/cartesianprimitive.jl")
include("equations/projectednse.jl")
include("equations/shared.jl")

end
