module MPIExt

import FDGrids
import HaloArrays
import LinearAlgebra
import MPI
import NSEBase

export comm, comm_size, comm_rank
export DecomposedGrid, distributed
export decomposition_storage_dims, decomposition_physical_dims, ndecomposed_dims
export global_size, global_first_index
export local_interior_range, local_boundary_ranges
export local_size, nhalo
export derivative_matrix
export interior_dd!, boundary_dd!
export interior_laplacian!, boundary_laplacian!

include("decomposed.jl")
include("types.jl")
include("ftfield.jl")
include("field.jl")
include("fft.jl")
include("galerkin.jl")
include("norms.jl")
include("derivatives.jl")

end
