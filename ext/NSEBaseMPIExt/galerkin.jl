"""
    NSEBase.project!(a, u)

MPI-aware override of [`NSEBase.project!`](@ref) for decomposed grids.

The input argument `a` is zero-ed before computations begin.

Each rank holds only a local slab of the inhomogeneous dimension, so
`_project_component!` produces a partial inner product over that slab.
`MPI.Allreduce!` then sums the partial modal coefficients across all ranks,
recovering the correct global projection.
"""
function NSEBase.project!(a::NSEBase.ProjectedField{<:DecomposedGrid},
                          u::NSEBase.VectorField{N, <:NSEBase.FTField{<:DecomposedGrid}}) where {N}

    fill!(parent(a), zero(eltype(a)))
    for i in 1:N
        NSEBase._project_component!(parent(a),
                                    parent(u[i]),
                                    modes(a)[i],
                                    weights(grid(u)),
                                    Val(fft_dims(grid(u))))
    end

    # Sum the per-rank partial projections into the global modal coefficients.
    MPI.Allreduce!(parent(a), MPI.SUM, comm(NSEBase.grid(u)))
    return a
end

# `expand!` does not need an MPI override: it reads from `a` (which is already
# globally consistent after `project!`) and writes to `u` locally.