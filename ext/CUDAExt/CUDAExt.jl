module CUDAExt

using CUDA,
      Adapt,
      LinearAlgebra

import NSEBase,
       FDGrids

import CUDA: i32

# make sure CUDA is functional
__init__() = @assert CUDA.functional(true)

include("utils.jl")
include("gpugrid.jl")
include("gpufields.jl")
include("fft.jl")
include("derivatives.jl")
include("galerkin.jl")
include("dot.jl")

end
