# ------------------------------------------------------------------ #
# HaloArray exchange for decomposed grids                            #
# ------------------------------------------------------------------ #

"""
    haloswap!(u::Field|FTField|VectorField, block=false)

Exchange halo cells for an NSEBase field on a [`DecomposedGrid`](@ref).

`block=false` (default) posts non-blocking sends/recvs and returns the
MPI request object(s); complete them with [`wait_halo_exchange!`](@ref).
`block=true` blocks until the exchange is done and returns `nothing`.
"""
haloswap!(u::NSEBase.Field{<:DecomposedGrid}, block::Bool = false) =
    HaloArrays.haloswap!(parent(u), block)

haloswap!(u::NSEBase.FTField{<:DecomposedGrid}, block::Bool = false) =
    HaloArrays.haloswap!(parent(u), block)

# VectorField: ntuple preserves the compile-time component count in the request structure.
haloswap!(u::NSEBase.VectorField{N, <:NSEBase.Field{<:DecomposedGrid}}, block::Bool = false) where {N} =
    ntuple(n -> haloswap!(u[n], block), Val(N))

haloswap!(u::NSEBase.VectorField{N, <:NSEBase.FTField{<:DecomposedGrid}}, block::Bool = false) where {N} =
    ntuple(n -> haloswap!(u[n], block), Val(N))

"""
    wait_halo_exchange!(requests)

Wait cooperatively until non-blocking halo exchange requests complete.

`MPI.AbstractMultiRequest` batches are polled with `yield()` between
polls so other ready tasks run while communication is in flight. Nested
tuples from all-dimension swaps and vector fields are handled recursively.
"""
function wait_halo_exchange!(requests::MPI.AbstractMultiRequest)
    while !MPI.Testall(requests)
        yield()
    end
    return nothing
end

# Recursively collapse nested tuples (vector fields, all-dimension swaps).
wait_halo_exchange!(tokens::Tuple) = (foreach(wait_halo_exchange!, tokens); nothing)