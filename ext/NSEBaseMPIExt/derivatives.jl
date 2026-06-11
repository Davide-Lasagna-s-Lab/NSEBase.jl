# Finite-difference and spectral derivatives for decomposed grids.
#
# Functions are ordered from most user-facing (top) to most internal (bottom):
#
#   ddx! / ddy! / ddz! / ddt!          — named user entry points
#   dd!(out, u, ::Val{SD}, requests)    — storage-dim dispatch + halo sequencing
#   laplacian!(out, u, requests)        — full Laplacian with halo sequencing
#   interior_dd! / boundary_dd!         — split FD derivative (pre/post halo)
#   interior_laplacian! / boundary_laplacian! — split Laplacian
#   _dd_over!                           — FD kernel
#
# Physical direction names (:x, :y, :z, :t) appear only in ddx!/ddy!/ddz!/ddt!,
# which resolve them to Val{STORAGE_DIM} via physical_to_storage_dim.
# All other functions take Val{STORAGE_DIM} directly.
#
# Concrete decomposed grids must implement
#   derivative_matrix(g, stor_dim::Int, ::Val{ORDER}, ::Val{ADJOINT})
# for every inhomogeneous spatial direction and every derivative order they support.


# ------------------------------------------------------------------ #
# ddx! / ddy! / ddz! / ddt!                                          #
# ------------------------------------------------------------------ #

NSEBase.ddx!(out::NSEBase.FTField{G}, u::NSEBase.FTField{G}, requests; kwargs...) where {G<:DecomposedGrid} =
    NSEBase.dd!(out, u, NSEBase.physical_to_storage_dim(NSEBase.grid(u), Val(:x)), requests; kwargs...)
NSEBase.ddx!(out::NSEBase.VectorField{N, F}, u::NSEBase.VectorField{N, F}, requests::NTuple{N}; kwargs...) where {N, F<:NSEBase.FTField{<:DecomposedGrid}} =
    NSEBase.dd!(out, u, NSEBase.physical_to_storage_dim(NSEBase.grid(u[1]), Val(:x)), requests; kwargs...)

NSEBase.ddy!(out::NSEBase.FTField{G}, u::NSEBase.FTField{G}, requests; kwargs...) where {G<:DecomposedGrid} =
    NSEBase.dd!(out, u, NSEBase.physical_to_storage_dim(NSEBase.grid(u), Val(:y)), requests; kwargs...)
NSEBase.ddy!(out::NSEBase.VectorField{N, F}, u::NSEBase.VectorField{N, F}, requests::NTuple{N}; kwargs...) where {N, F<:NSEBase.FTField{<:DecomposedGrid}} =
    NSEBase.dd!(out, u, NSEBase.physical_to_storage_dim(NSEBase.grid(u[1]), Val(:y)), requests; kwargs...)

NSEBase.ddz!(out::NSEBase.FTField{G}, u::NSEBase.FTField{G}, requests; kwargs...) where {G<:DecomposedGrid} =
    NSEBase.dd!(out, u, NSEBase.physical_to_storage_dim(NSEBase.grid(u), Val(:z)), requests; kwargs...)
NSEBase.ddz!(out::NSEBase.VectorField{N, F}, u::NSEBase.VectorField{N, F}, requests::NTuple{N}; kwargs...) where {N, F<:NSEBase.FTField{<:DecomposedGrid}} =
    NSEBase.dd!(out, u, NSEBase.physical_to_storage_dim(NSEBase.grid(u[1]), Val(:z)), requests; kwargs...)

NSEBase.ddt!(out::NSEBase.FTField{G}, u::NSEBase.FTField{G}, requests; kwargs...) where {G<:DecomposedGrid} =
    NSEBase.dd!(out, u, NSEBase.physical_to_storage_dim(NSEBase.grid(u), Val(:t)), requests; kwargs...)
NSEBase.ddt!(out::NSEBase.VectorField{N, F}, u::NSEBase.VectorField{N, F}, requests::NTuple{N}; kwargs...) where {N, F<:NSEBase.FTField{<:DecomposedGrid}} =
    NSEBase.dd!(out, u, NSEBase.physical_to_storage_dim(NSEBase.grid(u[1]), Val(:t)), requests; kwargs...)


# ------------------------------------------------------------------ #
# dd! — halo sequencing                                              #
# ------------------------------------------------------------------ #

function NSEBase.dd!(out::F, u::F, ::Val{STORAGE_DIM}, requests;
                     kwargs...) where {F<:NSEBase.FTField{<:DecomposedGrid}, STORAGE_DIM}
    interior_dd!(out, u, Val(STORAGE_DIM); kwargs...)
    wait_halo_exchange!(requests)
    boundary_dd!(out, u, Val(STORAGE_DIM); kwargs...)
    return out
end

function NSEBase.dd!(out::NSEBase.VectorField{N, F}, u::NSEBase.VectorField{N, F},
                     ::Val{STORAGE_DIM}, requests::NTuple{N};
                     kwargs...) where {N, F<:NSEBase.FTField{<:DecomposedGrid}, STORAGE_DIM}
    for n in 1:N
        NSEBase.dd!(out[n], u[n], Val(STORAGE_DIM), requests[n]; kwargs...)
    end
    return out
end


# ------------------------------------------------------------------ #
# laplacian! — halo sequencing                                       #
# ------------------------------------------------------------------ #

function NSEBase.laplacian!(out::NSEBase.FTField{G}, u::NSEBase.FTField{G}, requests;
                            kwargs...) where {G<:DecomposedGrid}
    interior_laplacian!(out, u; kwargs...)
    wait_halo_exchange!(requests)
    boundary_laplacian!(out, u; kwargs...)
    return out
end

function NSEBase.laplacian!(out::NSEBase.FTField{G}, u::NSEBase.FTField{G};
                            kwargs...) where {G<:DecomposedGrid}
    requests = haloswap!(u, false)
    return NSEBase.laplacian!(out, u, requests; kwargs...)
end

function NSEBase.laplacian!(out::NSEBase.VectorField{N, F}, u::NSEBase.VectorField{N, F},
                            requests::NTuple{N}; kwargs...) where {N, F<:NSEBase.FTField{<:DecomposedGrid}}
    interior_laplacian!(out, u; kwargs...)
    wait_halo_exchange!(requests)
    boundary_laplacian!(out, u; kwargs...)
    return out
end

function NSEBase.laplacian!(out::NSEBase.VectorField{N, F}, u::NSEBase.VectorField{N, F};
                            kwargs...) where {N, F<:NSEBase.FTField{<:DecomposedGrid}}
    requests = haloswap!(u, false)
    return NSEBase.laplacian!(out, u, requests; kwargs...)
end


# ------------------------------------------------------------------ #
# interior_dd! / boundary_dd!                                        #
# ------------------------------------------------------------------ #

function interior_dd!(out::NSEBase.FTField{G}, u::NSEBase.FTField{G},
                      ::Val{STORAGE_DIM};
                      adjoint::Bool = false) where {G<:DecomposedGrid, STORAGE_DIM}
    isnothing(STORAGE_DIM) && return out
    g = NSEBase.grid(u)
    STORAGE_DIM in NSEBase.fft_storage_dims(g) &&
        return NSEBase.dd!(out, u, Val(STORAGE_DIM); adjoint = adjoint)
    return _dd_over!(out, u, g, Val(STORAGE_DIM), Val(1),
                     (local_interior_range(g, STORAGE_DIM),);
                     adjoint = adjoint)
end

function interior_dd!(out::NSEBase.VectorField{N, F}, u::NSEBase.VectorField{N, F},
                      ::Val{STORAGE_DIM}; kwargs...) where {N, F<:NSEBase.FTField{<:DecomposedGrid}, STORAGE_DIM}
    for n in 1:N
        interior_dd!(out[n], u[n], Val(STORAGE_DIM); kwargs...)
    end
    return out
end

function boundary_dd!(out::NSEBase.FTField{G}, u::NSEBase.FTField{G},
                      ::Val{STORAGE_DIM};
                      adjoint::Bool = false) where {G<:DecomposedGrid, STORAGE_DIM}
    isnothing(STORAGE_DIM) && return out
    g = NSEBase.grid(u)
    STORAGE_DIM in NSEBase.fft_storage_dims(g) && return out
    return _dd_over!(out, u, g, Val(STORAGE_DIM), Val(1),
                     local_boundary_ranges(g, STORAGE_DIM);
                     adjoint = adjoint)
end

function boundary_dd!(out::NSEBase.VectorField{N, F}, u::NSEBase.VectorField{N, F},
                      ::Val{STORAGE_DIM}; kwargs...) where {N, F<:NSEBase.FTField{<:DecomposedGrid}, STORAGE_DIM}
    for n in 1:N
        boundary_dd!(out[n], u[n], Val(STORAGE_DIM); kwargs...)
    end
    return out
end


# ------------------------------------------------------------------ #
# interior_laplacian! / boundary_laplacian!                          #
# ------------------------------------------------------------------ #

function interior_laplacian!(out::NSEBase.FTField{G}, u::NSEBase.FTField{G};
                             adjoint::Bool = false) where {G<:DecomposedGrid}
    g = NSEBase.grid(u)
    fd_stor_dims = NSEBase.spatial_inhomogeneous_storage_dims(g)

    first_sd = first(fd_stor_dims)
    _dd_over!(out, u, g, Val(first_sd), Val(2),
              (local_interior_range(g, first_sd),);
              adjoint = adjoint, accumulate = Val(false))

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
                  adjoint = adjoint, accumulate = Val(true))
    end

    NSEBase.add_homogeneous_laplacian!(out, u)
    return out
end

function interior_laplacian!(out::NSEBase.VectorField{N, F}, u::NSEBase.VectorField{N, F};
                              kwargs...) where {N, F<:NSEBase.FTField{<:DecomposedGrid}}
    for n in 1:N
        interior_laplacian!(out[n], u[n]; kwargs...)
    end
    return out
end

function boundary_laplacian!(out::NSEBase.FTField{G}, u::NSEBase.FTField{G};
                             adjoint::Bool = false) where {G<:DecomposedGrid}
    g = NSEBase.grid(u)
    for sd in NSEBase.spatial_inhomogeneous_storage_dims(g)
        _dd_over!(out, u, g, Val(sd), Val(2),
                  local_boundary_ranges(g, sd);
                  adjoint = adjoint, accumulate = Val(true))
    end
    return out
end

function boundary_laplacian!(out::NSEBase.VectorField{N, F}, u::NSEBase.VectorField{N, F};
                              kwargs...) where {N, F<:NSEBase.FTField{<:DecomposedGrid}}
    for n in 1:N
        boundary_laplacian!(out[n], u[n]; kwargs...)
    end
    return out
end


# ------------------------------------------------------------------ #
# _dd_over! — FD kernel                                              #
# ------------------------------------------------------------------ #

@inline function _dd_over!(out, u, g::DecomposedGrid, ::Val{STORAGE_DIM}, ::Val{ORDER},
                           ranges;
                           adjoint::Bool = false,
                           accumulate::Val{B} = Val(false)) where {STORAGE_DIM, ORDER, B}
    A = derivative_matrix(g, STORAGE_DIM, Val(ORDER), Val(adjoint))
    g_first = global_first_index(g, STORAGE_DIM)
    for rng in ranges
        isempty(rng) && continue
        LinearAlgebra.mul!(parent(out), A, parent(u),
                           Val(STORAGE_DIM), g_first, rng, accumulate)
    end
    return out
end
