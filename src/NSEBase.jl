module NSEBase

using LinearAlgebra, FFTW

export FFTW

export AbstractGrid, points
export FTField, Field, VectorField, grid
export FFTPlans, FFT, IFFT
export ProjectedField, modes, project!, project, expand!, expand
export ProjectedNSE
export @loop_nznt
export _loop_fft_dims, _loop_all_modes, _signed_wavenumber
export _quadrature_weight

include("notimplementederror.jl")
include("abstractgrid.jl")
include("modenumber.jl")
include("ftfield.jl")
include("field.jl")
include("vectorfield.jl")
include("fft.jl")
include("projectedfield.jl")
include("broadcasting.jl")
include("projectednse.jl")

end
