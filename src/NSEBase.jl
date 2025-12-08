module NSEBase

using LinearAlgebra, FFTW

export AbstractScalarField
export VectorField
export FFTPlans
export ProjectedField, modes, expand!, project!, project
export NSE, LNSE
export ProjectedNSE

include("notimplementederror.jl")
include("abstractscalarfield.jl")
include("vectorfield.jl")
include("fft.jl")
include("projectedfield.jl")
include("broadcasting.jl")
include("nse.jl")
include("projectednse.jl")

# TODO: make a dependency for OpenChannelFlow.jl!!!

end
