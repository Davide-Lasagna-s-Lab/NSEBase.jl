# Physical-space field constructors for decomposed grids.

"""
    NSEBase.Field(grid::DecomposedGrid; dealias=false) -> Field

Construct a zero-initialised physical-space field on a decomposed grid.

Storage is always `HaloArrays.HaloArray`-backed with face-only halo exchange
(`economic=true`). Decomposed dimensions must have positive halo widths;
non-decomposed and FFT-transformed dimensions keep zero halo width.

The interior size is `local_size(grid)` without dealiasing. With
`dealias=true`, the 3/2-rule padding is applied to the FFT-transformed
dimensions of that local size.
"""
function NSEBase.Field(g::DecomposedGrid{T}; dealias::Bool=false) where {T}
    # The local interior excludes ghost cells; HaloArray adds the halo layers
    # required by the decomposed finite-difference directions.
    localsize = dealias ?
        NSEBase.get_padded_size(local_size(g), NSEBase.fft_storage_dims(g)) :
        local_size(g)
    storage = HaloArrays.HaloArray{T}(comm(g), localsize, nhalo(g);
                                      economic=true)
    return NSEBase.Field(g, storage)
end

"""
    NSEBase.Field(grid::DecomposedGrid, func; dealias=false) -> Field

Construct a physical-space field by evaluating `func` at the grid points.

`func` is called as `func(coords...)` where `coords = NSEBase.points(grid;
dealias=dealias)`.  `points` must return per-rank local coordinate arrays
(as required by the decomposed-grid locality contract), so `func`
is evaluated only at the collocation points owned by this rank. The result is
written into the HaloArray interior, with halos left uninitialised until the
next exchange.
"""
function NSEBase.Field(g::DecomposedGrid{T},
                       func::Function;
                       dealias::Bool=false) where {T}
    # Allocate through the constructor above so analytic and zero-initialised
    # fields always follow the same halo-storage policy.
    u = NSEBase.Field(g; dealias=dealias)

    # `parent(u)` unwraps NSEBase's Field wrapper. When the storage is a
    # HaloArray, its broadcast axes cover only the owned local interior, so
    # ghost cells remain untouched until the first exchange.
    parent(u) .= T.(func.(NSEBase.points(g; dealias=dealias)...))
    return u
end
