module CUDAExt

using CUDA,
      Adapt,
      LinearAlgebra

import CUDA: i32
import Adapt: adapt_structure

import NSEBase,
       FDGrids

# make sure CUDA is functional
__init__() = @assert CUDA.functional(true)

"""
Toggle for extra info for tuning performed the first time
some methods are executed.
"""
const TUNING_INFO = Ref(false)

"""
Global state parameter storing optimal kernel threads for
given input types.
"""
const LAUNCH_PARAMS = Dict{Tuple{Type, NTuple}, Int32}()

include("utils.jl")
include("gpugrid.jl")
include("gpufields.jl")
include("fft.jl")
include("derivatives.jl")
include("dot.jl")
include("galerkin.jl")

end
