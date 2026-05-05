using Test

using NSEBase

# The purpose of these tests to assess that the if a subtype of AbstractScalarField behaves as expected.
# This can be assessed by testing:
#   - derived methods output the correct types
#   - derived types have well-defined behaviour (all methods output as expected)
#   - large collection/cache types and operators behave as expected
# 
# I also need to add some good comprehensive tests for the FFT methods, since these are constructed
# concretely and should work for N-dimensional arrays with up to 3D transformations.

include("test_notimplementederror.jl")
# include("test_abstractscalarfield.jl")
# include("test_vectorfield.jl")
include("test_fft.jl")
include("test_fftplans.jl")
# include("test_projectedfield.jl")
# include("test_projectednse.jl")
