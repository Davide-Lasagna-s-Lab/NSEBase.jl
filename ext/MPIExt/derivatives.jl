# Halo exchange and finite-difference derivatives for decomposed grids.
#
# Concrete decomposed grids must implement
#   derivative_matrix(g, stor_dim::Int, ::Val{ORDER}, ::Val{ADJOINT})
# for every inhomogeneous spatial direction and every derivative order they support.


# ------------------------------------------------------------------ #
# init_requests! / wait_requests!                                    #
# ------------------------------------------------------------------ #

NSEBase.init_requests!(u::DecomposedScalarField) = HaloArrays.haloswap!(parent(u), false)
NSEBase.init_requests!(u::DecomposedVectorField{N}) where {N} =
    ntuple(n -> HaloArrays.haloswap!(parent(u[n]), false), Val(N))

NSEBase.wait_requests!(requests::MPI.AbstractMultiRequest) = (MPI.Waitall(requests); nothing)
NSEBase.wait_requests!(tokens::Tuple) = (foreach(NSEBase.wait_requests!, tokens); nothing)


# ------------------------------------------------------------------ #
# laplacian! — 2-arg blocking override for decomposed grids          #
# ------------------------------------------------------------------ #

function NSEBase.laplacian!(out::DecomposedFTField, u::DecomposedFTField; kwargs...)
    requests = NSEBase.init_requests!(u)
    interior_laplacian!(out, u; kwargs...)
    NSEBase.wait_requests!(requests)
    boundary_laplacian!(out, u; kwargs...)
    return out
end

function NSEBase.laplacian!(out::DecomposedFTVectorField,
                            u::DecomposedFTVectorField; kwargs...)
    requests = NSEBase.init_requests!(u)
    interior_laplacian!(out, u; kwargs...)
    NSEBase.wait_requests!(requests)
    boundary_laplacian!(out, u; kwargs...)
    return out
end


# ------------------------------------------------------------------ #
# interior_dd! / boundary_dd!                                        #
# ------------------------------------------------------------------ #

function interior_dd!(out::DecomposedFTField, u::DecomposedFTField,
                      ::Val{STORAGE_DIM}; adjoint::Bool = false) where {STORAGE_DIM}
    isnothing(STORAGE_DIM) && return out
    g = NSEBase.grid(u)
    STORAGE_DIM in NSEBase.fft_storage_dims(g) &&
        return NSEBase.dd!(out, u, Val(STORAGE_DIM); adjoint = adjoint)
    return _dd_over!(out, u, g, Val(STORAGE_DIM), Val(1),
                     (local_interior_range(g, STORAGE_DIM),);
                     adjoint = adjoint)
end

function interior_dd!(out::DecomposedFTVectorField, u::DecomposedFTVectorField,
                      ::Val{STORAGE_DIM}; kwargs...) where {STORAGE_DIM}
    for n in eachindex(u)
        interior_dd!(out[n], u[n], Val(STORAGE_DIM); kwargs...)
    end
    return out
end

function boundary_dd!(out::DecomposedFTField, u::DecomposedFTField,
                      ::Val{STORAGE_DIM}; adjoint::Bool = false) where {STORAGE_DIM}
    isnothing(STORAGE_DIM) && return out
    g = NSEBase.grid(u)
    STORAGE_DIM in NSEBase.fft_storage_dims(g) && return out
    return _dd_over!(out, u, g, Val(STORAGE_DIM), Val(1),
                     local_boundary_ranges(g, STORAGE_DIM);
                     adjoint = adjoint)
end

function boundary_dd!(out::DecomposedFTVectorField, u::DecomposedFTVectorField,
                      ::Val{STORAGE_DIM}; kwargs...) where {STORAGE_DIM}
    for n in eachindex(u)
        boundary_dd!(out[n], u[n], Val(STORAGE_DIM); kwargs...)
    end
    return out
end


# ------------------------------------------------------------------ #
# interior_laplacian! / boundary_laplacian!                          #
# ------------------------------------------------------------------ #

function interior_laplacian!(out::DecomposedFTField, u::DecomposedFTField;
                             adjoint::Bool = false)
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

function interior_laplacian!(out::DecomposedFTVectorField,
                             u::DecomposedFTVectorField; kwargs...)
    for n in eachindex(u)
        interior_laplacian!(out[n], u[n]; kwargs...)
    end
    return out
end

function boundary_laplacian!(out::DecomposedFTField, u::DecomposedFTField;
                             adjoint::Bool = false)
    g = NSEBase.grid(u)
    for sd in NSEBase.spatial_inhomogeneous_storage_dims(g)
        _dd_over!(out, u, g, Val(sd), Val(2),
                  local_boundary_ranges(g, sd);
                  adjoint = adjoint, accumulate = Val(true))
    end
    return out
end

function boundary_laplacian!(out::DecomposedFTVectorField,
                             u::DecomposedFTVectorField; kwargs...)
    for n in eachindex(u)
        boundary_laplacian!(out[n], u[n]; kwargs...)
    end
    return out
end


# ------------------------------------------------------------------ #
# init_* / complete_* / init_laplacian! / complete_laplacian!        #
# ------------------------------------------------------------------ #

NSEBase.init_ddx!(out::DecomposedSpectralField, u::DecomposedSpectralField; kwargs...) = interior_dd!(out, u, NSEBase.physical_to_storage_dim(NSEBase.grid(u), Val(:x)); kwargs...)
NSEBase.init_ddy!(out::DecomposedSpectralField, u::DecomposedSpectralField; kwargs...) = interior_dd!(out, u, NSEBase.physical_to_storage_dim(NSEBase.grid(u), Val(:y)); kwargs...)
NSEBase.init_ddz!(out::DecomposedSpectralField, u::DecomposedSpectralField; kwargs...) = interior_dd!(out, u, NSEBase.physical_to_storage_dim(NSEBase.grid(u), Val(:z)); kwargs...)
NSEBase.init_ddt!(out::DecomposedSpectralField, u::DecomposedSpectralField; kwargs...) = interior_dd!(out, u, NSEBase.physical_to_storage_dim(NSEBase.grid(u), Val(:t)); kwargs...)

NSEBase.complete_ddx!(out::DecomposedSpectralField, u::DecomposedSpectralField; kwargs...) = boundary_dd!(out, u, NSEBase.physical_to_storage_dim(NSEBase.grid(u), Val(:x)); kwargs...)
NSEBase.complete_ddy!(out::DecomposedSpectralField, u::DecomposedSpectralField; kwargs...) = boundary_dd!(out, u, NSEBase.physical_to_storage_dim(NSEBase.grid(u), Val(:y)); kwargs...)
NSEBase.complete_ddz!(out::DecomposedSpectralField, u::DecomposedSpectralField; kwargs...) = boundary_dd!(out, u, NSEBase.physical_to_storage_dim(NSEBase.grid(u), Val(:z)); kwargs...)
NSEBase.complete_ddt!(out::DecomposedSpectralField, u::DecomposedSpectralField; kwargs...) = boundary_dd!(out, u, NSEBase.physical_to_storage_dim(NSEBase.grid(u), Val(:t)); kwargs...)

NSEBase.init_laplacian!(out::DecomposedSpectralField, u::DecomposedSpectralField; kwargs...) = interior_laplacian!(out, u; kwargs...)
NSEBase.complete_laplacian!(out::DecomposedSpectralField, u::DecomposedSpectralField; kwargs...) = boundary_laplacian!(out, u; kwargs...)


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
