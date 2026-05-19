module NSEBase

using LinearAlgebra, FFTW, JLD2

export FFTW

export AbstractGrid
export points, growto, weights, fft_dims, inhomogeneous_dims, storage_order
export ddx_1!, ddx_2!, ddx_3!, ddx_4!
export ModeNumber
export FTField, Field, VectorField, grid
export add_base_flow!
export FFTPlans, FFT, IFFT
export ProjectedField, modes, project!, project, expand!, expand, get_mode_coefficient
export shift!, shift, normdiff, minnormdiff
export save_field, load_field
export CartesianPrimitive3DNSE, CartesianPrimitive3DLNSE
export CartesianPrimitive2DNSE, CartesianPrimitive2DLNSE
export Forward, AdjointContinuous, AdjointDiscrete, NoForce, Mode
export construct_equations, CartesianPrimitive, CartesianPrimitive2D, PolarPrimitive
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
include("equations/types.jl")
include("equations/cartesianprimitive_3d.jl")
include("equations/cartesianprimitive_2d.jl")
include("equations/projectednse.jl")
include("equations/shared.jl")

end
