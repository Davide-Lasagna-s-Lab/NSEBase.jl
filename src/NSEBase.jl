module NSEBase

using LinearAlgebra, FFTW

export FFTW

export AbstractGrid, points, growto, weights, fft_dims, inhomogeneous_dims
export ModeNumber
export FTField, Field, VectorField, grid
export FFTPlans, FFT, IFFT
export ProjectedField, modes, project!, project, expand!, expand
export shift!, shift, normdiff
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
include("equations/cartesianprimitive.jl")
include("equations/projectednse.jl")
include("equations/shared.jl")

end
