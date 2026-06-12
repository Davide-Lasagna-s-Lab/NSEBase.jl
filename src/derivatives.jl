# Spectral differentiation operators for FTField, ProjectedField, and VectorField.
#
# Entry points are `ddx!`, `ddy!`, `ddz!`, `ddt!` — one per physical coordinate.
# Each resolves its direction to a compile-time `Val{STORAGE_DIM}` via
# `physical_to_storage_dim` and delegates to `dd!(out, u, ::Val{STORAGE_DIM})`.
#
# `dd!(out, u, ::Val{STORAGE_DIM})` is the core primitive, implemented with
# CartesianIndices loops.  For the rfft dimension all stored wavenumbers are
# non-negative so a single loop over the full array suffices.  For signed-FFT
# dimensions the index range is split into a positive block [1:(N÷2)+1] and a
# negative block [(N÷2)+2:N] so that the signed wavenumber can be computed
# branch-free from the loop index inside each block.
#
# When a physical direction is absent from the grid its storage dimension is
# `nothing` and the kernel is a no-op.
#
# Inhomogeneous (non-FFT) directions are NOT handled here — `dd!` throws
# `NotImplementedError` for those, and downstream packages must extend it
# with a grid-specific method (typically a matrix–vector multiply).
#
# The full Laplacian combines `inhomogeneous_laplacian!` (provided by
# downstream) with `add_homogeneous_laplacian!` (provided here), which
# subtracts the spatial ‖k‖² · u contribution from each spectral coefficient.

"""
    ddx!(out, u; adjoint=false) -> out
    ddy!(out, u; adjoint=false) -> out
    ddz!(out, u; adjoint=false) -> out
    ddt!(out, u; adjoint=false) -> out

Differentiate `u` along the named physical direction, storing the result in
`out`. Each wrapper resolves its direction to a `Val{STORAGE_DIM}` at the
call site and delegates to the low-level [`dd!`](@ref)`(out, u, ::Val)` primitive.

For an absent direction (e.g. `:z` on a 2D grid) the call is a compile-time no-op.
"""
ddx!(out, u; kwargs...) = dd!(out, u, physical_to_storage_dim(grid(u), Val(:x)); kwargs...)
ddy!(out, u; kwargs...) = dd!(out, u, physical_to_storage_dim(grid(u), Val(:y)); kwargs...)
ddz!(out, u; kwargs...) = dd!(out, u, physical_to_storage_dim(grid(u), Val(:z)); kwargs...)
ddt!(out, u; kwargs...) = dd!(out, u, physical_to_storage_dim(grid(u), Val(:t)); kwargs...)

dd!(out::VectorField{N}, u::VectorField{N}, sd::Val; kwargs...) where {N} =
    (for n in 1:N; dd!(out[n], u[n], sd; kwargs...); end; return out)

"""
    dd!(out, u, ::Val{STORAGE_DIM}; adjoint=false)

In-place derivative of `u` along the storage dimension encoded by
`Val(STORAGE_DIM)`.

For `STORAGE_DIM in FFT_DIMS_ORDER` the derivative is multiplication by
`±im * n * wavenumber_scale(grid, STORAGE_DIM)` where `n` is the signed
wavenumber. `adjoint=false` (default) gives `+im·n·scale·u`;
`adjoint=true` gives `-im·n·scale·u`.

For an inhomogeneous storage dimension the method throws
`NotImplementedError`. Downstream packages should extend this for each
non-homogeneous direction.

For `Val(nothing)` the function is a no-op.
"""
function dd!(out::F, u::F, ::Val{STORAGE_DIM};
             adjoint::Bool=false) where {
        STORAGE_DIM, T, D, AXES, FFT_DIMS_ORDER,
        G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER},
        F<:Union{FTField{G}, ProjectedField{G}}}

    isnothing(STORAGE_DIM) && return out
    STORAGE_DIM ∉ FFT_DIMS_ORDER &&
        throw(NotImplementedError(grid(u), Val(STORAGE_DIM)))

    scale = wavenumber_scale(grid(u), STORAGE_DIM)
    coeff = adjoint ? -im * T(scale) : im * T(scale)
    Nd    = size(u, STORAGE_DIM)
    pu    = parent(u)
    pout  = parent(out)

    if STORAGE_DIM == FFT_DIMS_ORDER[1]
        @inbounds for I in CartesianIndices(pu)
            pout[I] = (coeff * (I[STORAGE_DIM] - 1)) * pu[I]
        end
    else
        pos_ranges = ntuple(d -> d == STORAGE_DIM ? (1:(Nd >> 1) + 1)  : Base.OneTo(size(pu, d)), Val(D))
        neg_ranges = ntuple(d -> d == STORAGE_DIM ? ((Nd >> 1) + 2:Nd) : Base.OneTo(size(pu, d)), Val(D))
        @inbounds for I in CartesianIndices(pos_ranges)
            pout[I] = (coeff * (I[STORAGE_DIM] - 1)) * pu[I]
        end
        @inbounds for I in CartesianIndices(neg_ranges)
            pout[I] = (coeff * (I[STORAGE_DIM] - Nd - 1)) * pu[I]
        end
    end
    return out
end

"""
    add_homogeneous_laplacian!(out::FTField, u::FTField)
    add_homogeneous_laplacian!(out::VectorField, u::VectorField)

Add the homogeneous Laplacian contribution of `u` to `out`:

    out[mode] -= (∑_{d∈spatial_fft_storage_dims(g)} (wavenumber_scale(g, d) · n_d)²) · u[mode]

Call after computing the non-homogeneous (e.g. wall-normal) second derivative.
If the grid includes a transformed logical time coordinate, that direction is
not part of the spatial Laplacian.
"""
function add_homogeneous_laplacian!(out::FTField{G}, u::FTField{G}) where {T, D, AXES, FFT_DIMS_ORDER, G<:AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}}
    g = grid(u)

    H = spatial_fft_storage_dims(g)
    scales = map(dim -> dim in H ? T(wavenumber_scale(g, dim)) : zero(T), FFT_DIMS_ORDER)
    pu = parent(u)
    pout = parent(out)

    @inbounds for Ih in CartesianIndices(homogeneous_axes(u))
        k² = zero(T)
        k = to_wavenumber_vector(g, Ih)
        for j in eachindex(FFT_DIMS_ORDER)
            k² += (scales[j] * k[j])^2
        end

        for Iinh in CartesianIndices(inhomogeneous_axes(u))
            I = combine_indices(g, Iinh, Ih)
            pout[I...] -= k² * pu[I...]
        end
    end
    return out
end
add_homogeneous_laplacian!(out::VectorField{N}, u::VectorField{N}) where {N} =
    (for n in 1:N; add_homogeneous_laplacian!(out[n], u[n]); end; return out)

"""
    inhomogeneous_laplacian!(out::FTField, u::FTField; kwargs...) -> out

Apply the inhomogeneous (non-FFT) part of the Laplacian of `u` to `out`.

This contribution contains the second derivatives along directions that are
not in [`fft_storage_dims`](@ref), such as wall-normal collocation directions.  The
full spatial Laplacian is the sum of this contribution and the homogeneous
spectral contribution from [`add_homogeneous_laplacian!`](@ref).
"""
inhomogeneous_laplacian!(out::FTField, u::FTField) = throw(NotImplementedError(out, u))

"""
    laplacian!(out::FTField{G}, u::FTField{G}; kwargs...)
    laplacian!(out::VectorField{N}, u::VectorField{N}; kwargs...)

Compute the full Laplacian of `u` in-place, storing the result in `out`:

    out[mode] = (inhomogeneous second derivatives
                 - ∑_{d∈spatial_fft_storage_dims(g)} (wavenumber_scale(g, d) · n_d)²) · u[mode]

Only spatial transformed directions enter the homogeneous sum; a transformed
logical time coordinate is excluded through [`spatial_fft_storage_dims`](@ref).
"""
function laplacian!(out::FTField{G}, u::FTField{G}; kwargs...) where {G}
    inhomogeneous_laplacian!(out, u; kwargs...)
    add_homogeneous_laplacian!(out, u)
    return out
end
laplacian!(out::VectorField{N}, u::VectorField{N}; kwargs...) where {N} =
    (for n in 1:N; laplacian!(out[n], u[n]; kwargs...); end; return out)

"""
    init_requests!(u)

Return the halo-exchange request handle(s) for `u`, initiating a non-blocking
exchange if needed.

For serial (non-decomposed) grids this is a no-op that returns `nothing`.
MPIExt overrides this for `DecomposedGrid` fields to post non-blocking
MPI communication; the returned handle is passed to the three-argument forms of
`ddx!`, `ddy!`, `laplacian!`, etc. so that interior work can proceed while the
halo is in flight.
"""
# ------------------------------------------------------------------ #
# init_requests! / wait_requests! / init_* / complete_*              #
# ------------------------------------------------------------------ #
#
# Equations are written as:
#
#   requests = init_requests!(u)
#   init_laplacian!(out, u); init_ddx!(dudx, u); ...
#   <other work that doesn't need halo data — overlaps with MPI>
#   wait_requests!(requests)
#   complete_laplacian!(out, u); complete_ddx!(dudx, u); ...
#
# For serial grids: init_requests! returns nothing, wait_requests! is a
# no-op, init_* does the full computation, complete_* is a no-op.
# MPIExt overrides all four families for DecomposedGrid fields so
# that interior rows are computed while MPI halos are in flight.
#
# NOTE (overlap effectiveness): single-node benchmarking shows this split gives
# no speedup over a plain blocking halo swap (and an async progress thread is
# strictly worse) — there is little shared-memory comm latency to hide. The
# split is kept because it is no worse, and may still pay off on a multi-node
# cluster with real network latency, which has NOT yet been benchmarked. Do not
# add the complexity of progress polling / threads, or remove the split, without
# multi-node data. See https://github.com/Davide-Lasagna-s-Lab/NSEBase.jl/issues/21

init_requests!(u) = nothing
wait_requests!(::Nothing) = nothing

init_ddx!(out, u; kwargs...) = ddx!(out, u; kwargs...)
init_ddy!(out, u; kwargs...) = ddy!(out, u; kwargs...)
init_ddz!(out, u; kwargs...) = ddz!(out, u; kwargs...)
init_ddt!(out, u; kwargs...) = ddt!(out, u; kwargs...)

complete_ddx!(out, u; kwargs...) = out
complete_ddy!(out, u; kwargs...) = out
complete_ddz!(out, u; kwargs...) = out
complete_ddt!(out, u; kwargs...) = out

init_laplacian!(out, u; kwargs...) = laplacian!(out, u; kwargs...)
complete_laplacian!(out, u; kwargs...) = out
