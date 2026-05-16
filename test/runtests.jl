using Test

using LinearAlgebra
using NSEBase

include("fake.jl")

include("test_notimplementederror.jl")
include("test_modenumber.jl")
include("test_spectralloops.jl")
include("test_fftplans.jl")
include("test_fields.jl")
include("test_operators.jl")
