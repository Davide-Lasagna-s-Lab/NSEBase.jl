# Halo exchange and finite-difference derivatives for decomposed grids.
#
# Concrete decomposed grids must implement
#   derivative_matrix(g, stor_dim::Int, ::Val{ORDER}, ::Val{ADJOINT})
# for every inhomogeneous spatial direction and every derivative order they support.


# ------------------------------------------------------------------ #
# init_requests! / wait_requests!                                    #
# ------------------------------------------------------------------ #
init_requests!(u::DecomposedScalarField) = HaloArrays.haloswap!(parent(u), false)
init_requests!(u::DecomposedVectorField{N}) where {N} =
    ntuple(n -> HaloArrays.haloswap!(parent(u[n]), false), Val(N))

wait_requests!(requests::MPI.AbstractMultiRequest) = (MPI.Waitall(requests); nothing)
wait_requests!(tokens::Tuple) = (foreach(wait_requests!, tokens); nothing)


# ------------------------------------------------------------------ #
# laplacian! — 2-arg blocking override for decomposed grids          #
# ------------------------------------------------------------------ #
"""
    NSEBase.laplacian!(out::DecomposedFTField,
                         u::DecomposedFTField;
                   adjoint::Bool=false) -> DecomposedFTField

Compute the full Laplacian of the distributed FTField `u` in-place,
storing the result in `out`.

The computation is split between the interior part of the decomposed
field which does not depend on the halo swap between processes, and
the boundary computations that can only be completed after the halo
swaps have completed.
"""
function NSEBase.laplacian!(out::DecomposedFTField,
                              u::DecomposedFTField;
                        adjoint::Bool=false)
    # start comm
    requests = init_requests!(u)

    # do interior work
    interior_laplacian!(out, u; adjoint=adjoint)

    # do boundary work once halo swap has concluded
    wait_requests!(requests)
    boundary_laplacian!(out, u; adjoint=adjoint)

    return out
end

"""
    interior_laplacian!(out::DecomposedFTField,
                          u::DecomposedFTField;
                    adjoint::Bool=false) -> DecomposedFTField

Compute the Laplacian operator on the parts of the distributed field
`u` that do not depend on data from the halo regions of the underlying
arrays.
"""
function interior_laplacian!(out::DecomposedFTField,
                               u::DecomposedFTField;
                         adjoint::Bool=false)
    g = NSEBase.grid(u)
    fd_stor_dims = NSEBase.spatial_inhomogeneous_storage_dims(g)

    first_sd = first(fd_stor_dims)
    _dd_over!(out, u, g, Val(first_sd), Val(2),
              (local_interior_range(g, first_sd),);
              adjoint=adjoint, accumulate=Val(false))

    # Zero boundary bands of the first FD direction before accumulating tail
    # dimensions, which span all first-dimension indices including halo bands.
    let a = parent(out)
        for rng in local_boundary_ranges(g, first_sd)
            isempty(rng) && continue
            fill!(selectdim(a, first_sd, rng), zero(eltype(a)))
        end
    end

    for sd in Base.tail(fd_stor_dims)
        _dd_over!(out, u, g, Val(sd), Val(2),
                  (local_interior_range(g, sd),);
                  adjoint=adjoint, accumulate=Val(true))
    end

    NSEBase.add_homogeneous_laplacian!(out, u)
    return out
end

"""
    boundary_laplacian!(out::DecomposedFTField,
                          u::DecomposedFTField;
                    adjoint::Bool=false) -> DecomposedFTField

Compute the remaining parts of the Laplacian of the distributed field
`u` that depend on the data in the halo regions of the underlying arrays.
This function should only be called after `wait_requests!(requests)` has
completed.
"""
function boundary_laplacian!(out::DecomposedFTField,
                               u::DecomposedFTField;
                         adjoint::Bool=false)
    g = NSEBase.grid(u)
    for sd in NSEBase.spatial_inhomogeneous_storage_dims(g)
        _dd_over!(out, u, g, Val(sd), Val(2),
                  local_boundary_ranges(g, sd);
                  adjoint=adjoint, accumulate=Val(true))
    end
    return out
end


# ------------------------------------------------------------------ #
# distributed derivatives                                            #
# ------------------------------------------------------------------ #
"""
    dd!(out::DecomposedFTField,
          u::DecomposedFTField,
           ::Val{STORAGE_DIM};
    adjoint::Bool=false) -> DecomposedFTField

In-place derivative of the distributed field `u` along the storage
dimension encoded by `Val(STORAGE_DIM)`.

For `STORAGE_DIM ∈ FFT_DIMS_ORDER` the code falls back to the standard
spectral derivative. For an inhomogeneous `STORAGE_DIM ∉ DDIMS` — a
finite-difference direction that is *not* decomposed, so every rank already
owns its full extent — it applies the (undecomposed) derivative matrix
directly over the whole local range, exactly as
[`interior_laplacian!`](@ref)/[`boundary_laplacian!`](@ref) already do for a
non-decomposed FD direction; no halo exchange is needed. A generic
[`NSEBase.inhomogeneous_dd!`](@ref) has no method for `DecomposedGrid`, so
that single-domain fallback cannot be used here.
"""
function NSEBase.dd!(out::F,
                       u::F,
                        ::Val{STORAGE_DIM};
                 adjoint::Bool=false) where {
    STORAGE_DIM, FFT_DIMS_ORDER, DDIMS, T, D, AXES,
    G<:DecomposedGrid{T, D, AXES, FFT_DIMS_ORDER, DDIMS},
    F<:Union{NSEBase.FTField{G}, NSEBase.ProjectedField{G}}} # ! can I get rid of the ProjectedField part?

    if STORAGE_DIM ∈ DDIMS
        _distributed_dd!(out, u, Val(STORAGE_DIM); adjoint=adjoint)
    elseif STORAGE_DIM ∈ FFT_DIMS_ORDER
        NSEBase._spectral_dd!(out, u, Val(STORAGE_DIM); adjoint=adjoint)
    else
        g = NSEBase.grid(u)
        _dd_over!(out, u, g, Val(STORAGE_DIM), Val(1),
                 (local_interior_range(g, STORAGE_DIM),); adjoint=adjoint, accumulate=Val(false))
    end

    return out
end

"""
    _distributed_dd!(out::DecomposedFTField,
                       u::DecomposedFTField,
                        ::Val{STORAGE_DIM};
                    adjoint=false) -> DecomposedFTField

Compute the derivative of a distributed field `u` along one of
decomposed storage dimensions, i.e. `STORAGE_DIM ∈ DDIMS`.
"""
function _distributed_dd!(out::Union{DecomposedFTField, DecomposedField},
                            u::Union{DecomposedFTField, DecomposedField},
                             ::Val{STORAGE_DIM};
                      adjoint::Bool=false) where {STORAGE_DIM}
    # start comm
    requests = init_requests!(u)

    # do interior work
    interior_dd!(out, u, Val(STORAGE_DIM); adjoint=adjoint)

    # do boundary work once halo swap has concluded
    wait_requests!(requests)
    boundary_dd!(out, u, Val(STORAGE_DIM); adjoint=adjoint)

    return out
end

"""
    interior_dd!(out::DecomposedFTField,
                   u::DecomposedFTField,
                    ::Val{STORAGE_DIM};
             adjoint::Bool=false) -> DecomposedFTField

Compute the derivative of the parts of the distributed field `u` along
the storage dimension `STORAGE_DIM` that do not depend on data from
the halo regions of the underlying arrays.
"""
interior_dd!(out::DecomposedFTField,
               u::DecomposedFTField,
                ::Val{STORAGE_DIM};
         adjoint::Bool=false) where {STORAGE_DIM} =
    _dd_over!(out, u, NSEBase.grid(u), Val(STORAGE_DIM), Val(1),
              (local_interior_range(NSEBase.grid(u), STORAGE_DIM),);
              adjoint=adjoint)

"""
    boundary_laplacian!(out::DecomposedFTField,
                          u::DecomposedFTField,
                           ::Val{STORAGE_DIM};
                    adjoint::Bool=false) -> DecomposedFTField

Compute the remaining parts of the derivative of the distributed field
`u` that depend on the data in the halo regions of the underlying arrays.
This function should only be called after `wait_requests!(requests)` has
completed.
"""
boundary_dd!(out::DecomposedFTField,
               u::DecomposedFTField,
                ::Val{STORAGE_DIM};
         adjoint::Bool=false) where {STORAGE_DIM} =
    _dd_over!(out, u, NSEBase.grid(u), Val(STORAGE_DIM), Val(1),
              local_boundary_ranges(NSEBase.grid(u), STORAGE_DIM);
              adjoint=adjoint)


# ------------------------------------------------------------------ #
# _dd_over! — FD kernel                                              #
# ------------------------------------------------------------------ #
"""
    _dd_over!(out, u, g::DecomposedGrid, ::Val{STORAGE_DIM}, ::Val{ORDER},
              ranges; adjoint::Bool=false, accumulate::Val=Val(false))

Compute the derivative of a distributed field `u` in-place along the
storage dimension `STORAGE_DIM ∈ DDIMS` and assign the
result to `out`.

Only a slice of the [`DiffMatrix`](@ref) is used for the computation,
corresponding to `range`, determined by both the particular part of the
overal decomposed field this is being computed on, as well as the particular
parts of the field locally that want to be computed.

Optionally, by setting `accumulate=Val(true)` the result can be accumulated
into `out`, instead of overwritting it with the result.

The Boolean adjoint keyword is resolved before entering the matrix kernel so
the selected forward or adjoint operator has one concrete type there.
"""
@inline function _dd_over!(out, u, g::DecomposedGrid, ::Val{STORAGE_DIM}, ::Val{ORDER},
                           ranges;
                           adjoint::Bool=false,
                           accumulate::Val{B}=Val(false)) where {STORAGE_DIM, ORDER, B}
    if adjoint
        A = NSEBase.derivative_matrix(g, STORAGE_DIM, Val(ORDER), Val(true))
        return _dd_over_matrix!(out, u, g, Val(STORAGE_DIM), ranges, A, accumulate)
    end
    A = NSEBase.derivative_matrix(g, STORAGE_DIM, Val(ORDER), Val(false))
    return _dd_over_matrix!(out, u, g, Val(STORAGE_DIM), ranges, A, accumulate)
end

@inline function _dd_over_matrix!(out, u, g::DecomposedGrid, dim::Val{STORAGE_DIM}, ranges,
                                  A, accumulate::Val) where {STORAGE_DIM}
    g_first = global_first_index(g, STORAGE_DIM)
    for rng in ranges
        isempty(rng) && continue
        LinearAlgebra.mul!(parent(out), A, parent(u),
                           dim, g_first, rng, accumulate)
    end
    return out
end
