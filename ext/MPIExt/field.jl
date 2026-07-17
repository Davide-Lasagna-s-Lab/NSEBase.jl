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
NSEBase.Field(g::DecomposedGrid{T}; dealias::Bool=false) where {T} = 
    NSEBase.Field(g, HaloArrays.HaloArray{T}(comm(g), 
                                             local_physical_size(g; dealias), 
                                             nhalo(g); economic=true))

"""
    NSEBase.Field(grid::DecomposedGrid, func; dealias=false) -> Field

Construct a physical-space field by evaluating `func` at the grid points.

`func` receives four arguments in physical `(x, y, z, t)` order, independently
of their storage order; absent coordinates are passed as `nothing`. `points`
must return per-rank local coordinate arrays (as required by the decomposed-grid
locality contract), so `func` is evaluated only at the collocation points owned
by this rank. The result is written into the HaloArray interior, with halos left
uninitialised until the next exchange.
"""
function NSEBase.Field(g::DecomposedGrid{T}, func::Function; dealias::Bool=false) where {T}
    u = NSEBase.Field(g; dealias=dealias)
    parent(u) .= func.(NSEBase._field_args(g; dealias)...)
    return u
end
