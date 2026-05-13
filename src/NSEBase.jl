module NSEBase

using LinearAlgebra, FFTW

export FFTW

export AbstractGrid, points
export FTField, Field, VectorField, grid
export FFTPlans, FFT, IFFT
export ProjectedField, modes, project!, project, expand!, expand
export CartesianPrimitiveNSE, CartesianPrimitiveLNSE, Forward, AdjointContinuous, AdjointDiscrete, NoForce
export ProjectedNSE

include("notimplementederror.jl")
include("abstractgrid.jl")
include("ftfield.jl")
include("field.jl")
include("vectorfield.jl")
include("fft.jl")
include("projectedfield.jl")
include("broadcasting.jl")
include("derivatives.jl")
include("equations/cartesianprimitive.jl")
include("projectednse.jl")

end
