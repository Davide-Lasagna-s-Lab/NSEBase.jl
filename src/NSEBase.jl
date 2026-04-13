module NSEBase

using LinearAlgebra, FFTW

export AbstractScalarField
export VectorField
export FFTPlans
export ProjectedField, modes, expand!, project!, project
export NSE, LNSE
export ProjectedNSE

include("notimplementederror.jl")
include("abstractgrid.jl")
include("ftfield.jl")
# include("field.jl")
# include("abstractscalarfield.jl")
# include("vectorfield.jl")
# include("fft.jl")
# include("projectedfield.jl")
# include("broadcasting.jl")
# include("nse.jl")
# include("projectednse.jl")

# TODO: somehow define a generic indexing interface for abstract scalar fields which will allow looping over modenumbers and thus more generic implementations here
# TODO: move vectorfield tests here

end
