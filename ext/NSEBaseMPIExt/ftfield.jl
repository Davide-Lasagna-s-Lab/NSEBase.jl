# Spectral field constructors for decomposed grids.

"""
    NSEBase.FTField(grid::DecomposedGrid) -> FTField

Construct a zero-initialised spectral field on a decomposed grid.

Storage is always `HaloArrays.HaloArray`-backed with face-only halo exchange
(`economic=true`). Decomposed dimensions must have positive halo widths;
non-decomposed and FFT-transformed dimensions keep zero halo width. The
interior size is `NSEBase.transform_size(grid)`, derived from the local
`size(grid)` and therefore per-rank.

**Note**: halos are indexed as part of the Fourier coefficient array along
the decomposed (physical-space) dimension.  This is the intended layout for
finite-difference stencils applied in spectral space — each Fourier mode
carries a row of wall-normal coefficients, and the halo stores the ghost rows
needed by the FD stencil at process boundaries.
"""
function NSEBase.FTField(g::DecomposedGrid{T}) where {T}
    # transform_size is the local Fourier-space interior size. In particular,
    # the real-to-complex reduction is applied only along NSEBase's first FFT
    # dimension; decomposed finite-difference dimensions retain their row count.
    # HaloArray adds the halo layers required by the decomposed finite-difference
    # directions.
    localsize = NSEBase.transform_size(g)
    storage = HaloArrays.HaloArray{Complex{T}}(comm(g), localsize, nhalo(g);
                                               economic=true)
    return NSEBase.FTField(g, storage)
end
