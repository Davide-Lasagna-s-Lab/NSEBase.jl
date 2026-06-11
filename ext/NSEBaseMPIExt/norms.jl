# MPI-reduced inner product for decomposed grids.
#
# NSEBase's `LinearAlgebra.dot(u::FTField, v::FTField)` accumulates a
# weighted sum over every inhomogeneous index this rank owns. On a
# decomposed grid each rank owns only a slab, so the local accumulation
# is a partial sum; `MPI.Allreduce` combines partial sums into the
# global inner product on every rank. The `VectorField` overload in
# NSEBase reduces to a sum over per-component scalar `dot` calls; each
# call is already globally summed by the override below, so the vector
# version is correct without further changes.
#
# `norm(u) = sqrt(dot(u, u))` and any caller that derives an aggregate
# quantity from `dot` likewise inherits the global semantics — no
# separate `norm` override is needed.

"""
    LinearAlgebra.dot(u::FTField{<:DecomposedGrid}, v::FTField{<:DecomposedGrid}) -> Real

MPI-reduced weighted inner product of two distributed spectral fields.

Each rank computes the same partial sum that NSEBase computes for a
single-domain field, then `MPI.Allreduce` sums the partial values into
the global inner product. Every rank returns the same value.

`u` and `v` must share the same `DecomposedGrid` (so the partition,
weights, and Hermitian-symmetry multipliers agree across ranks).
"""
function LinearAlgebra.dot(u::NSEBase.FTField{G},
                           v::NSEBase.FTField{G}) where {FFT_DIMS_ORDER, G<:DecomposedGrid{<:Any,
                                                                                           <:Any,
                                                                                           <:Any,
                                                                                           FFT_DIMS_ORDER}}
    local_dot = NSEBase._dot(parent(u),
                             parent(v),
                             NSEBase.weights(NSEBase.grid(u)),
                             Val(FFT_DIMS_ORDER))
    return MPI.Allreduce(real(local_dot), MPI.SUM, comm(NSEBase.grid(u)))
end
