module MPIExt

import FDGrids
import HaloArrays
import LinearAlgebra
import MPI
import NSEBase
using NSEBase: Forward, AdjointDiscrete, OperatorMode

include("decomposed.jl")
include("types.jl")
include("ftfield.jl")
include("field.jl")
include("fft.jl")
include("galerkin.jl")
include("norms.jl")
include("derivatives.jl")

end
