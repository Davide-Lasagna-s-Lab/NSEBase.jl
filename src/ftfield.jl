# Fourier transformed scalar field.

# TODO: implement some sort of norm interface?

"""
    FTField{G} where {G<:AbstractGrid}

Fourier transformed representation of a scalar field on
a concrete sub-type of AbstractGrid.

# Fields
- `grid`: concrete instance of `AbstractGrid`
- `data`: fourier coefficient values of scalar field
"""
struct FTField{G<:AbstractGrid, A<:AbstractArray, T, D} <: AbstractArray{Complex{T}, D}
    grid::G
    data::A

    # generic constructor that bypasses sanitation
    FTField(grid::G, data::A) where {T, D, G<:AbstractGrid{T, D}, A<:AbstractArray{<:Any, D}} =
        new{G, A, T, D}(grid, Complex{T}.(data))

    # main constructor which sanitises data
    FTField(grid::G, data::A) where {T, D, H, G<:AbstractGrid{T, D, H}, A<:Array{<:Any, D}} =
        new{G, A, T, D}(grid, Complex{T}.(normalise_mean!(apply_symmetry!(data, H), H)))
end

# construct from standard array and sanitise input
FTField(grid::AbstractGrid{T}) where {T} = FTField(grid, zeros(Complex{T}, transform_size(grid)))

Base.IndexStyle(::Type{<:FTField})                                    = Base.IndexLinear()
Base.parent(u::FTField)                                               = u.data
Base.eltype(::FTField{<:AbstractGrid{T}}) where {T}                   = Complex{T}
Base.similar(u::FTField{<:AbstractGrid{T}}, ::Type{S}=T) where {T, S} = FTField(similar(grid(u), S), zero(parent(u)))
Base.size(u::FTField)                                                 = Base.size(parent(u))
Base.copy(u::FTField)                                                 = (v = Base.similar(u); parent(v) .= parent(u); return v)
Base.zero(u::FTField{<:AbstractGrid{T}}) where {T}                    = (v = Base.similar(u); parent(v) .= zero(T)  ; return v)

# linear indexing
Base.@propagate_inbounds function Base.getindex(u::FTField, i)
    @boundscheck checkbounds(parent(u), i)
    @inbounds v = parent(u)[i]
    return v
end

Base.@propagate_inbounds function Base.setindex!(u::FTField, v, i)
    @boundscheck checkbounds(parent(u), i)
    @inbounds parent(u)[i] = v
    return v
end

grid(u::FTField) = u.grid


# --------------------- #
# inner-product methods #
# --------------------- #
LinearAlgebra.dot(u::FTField, v::FTField) = throw(NotImplementedError(u, v))
LinearAlgebra.norm(u::FTField) = sqrt(dot(u, u))


# --------------- #
# utility methods #
# --------------- #
@inline function _average_complex(z1::T, z2::T) where {T}
    _re = 0.5 * (real(z1) + real(z2))
    _im = 0.5 * (imag(z1) - imag(z2))
    return _re + im * _im
end

"""
    apply_symmetry!(data::AbstractArray{T, D}, ::Val{H})

Enforce Hermitian symmetry on `data` after a multi-dimensional RFFT along
the dimensions listed in `H` (in transform order).

For a real-valued physical field, Fourier coefficients must satisfy
û(-k) = conj(û(k)). After the RFFT along `H[1]`, only non-negative
frequencies along that axis are stored. Along the remaining axes `H[2:end]`
both signs are present, but at the zero-frequency plane of `H[1]` (index 1)
the coefficients must still satisfy the conjugate-symmetry constraint — this
function enforces it by averaging each conjugate pair in-place.

Dimensions not in `H` are untransformed and are iterated over in full.
No-op when only one dimension is transformed (`length(H) == 1`).

# Note on cache efficiency
The DC plane of `H[1]` (the rfft dim) has memory stride `size(data, H[1])`
rather than 1, because the rfft dim is stored as the leading (stride-1)
dimension. The loop iterates the remaining dims in ascending stride order,
which is optimal for the current layout. If this function is a bottleneck,
storing the rfft dim last would give stride-1 access to the DC plane.
"""
@generated function apply_symmetry!(data::AbstractArray{T, D}, ::Val{H}) where {T, D, H}
    length(H) <= 1 && return :(return data)

    # All subset enumeration happens at code-generation time so the compiled
    # hot loop contains only literal index expressions — no closures, no dynamic
    # dispatch, no allocations.
    secondary = H[2:end]
    blocks    = Expr[]

    for mask_int in 1:(1 << length(secondary)) - 1

        # active subset of secondary dims for this pass
        active_dims = Tuple(secondary[k] for k in eachindex(secondary) if Bool((mask_int >> (k-1)) & 1))
        half_dim    = active_dims[end]

        # per-dim range literals (e.g. 1:1, 2:Base.size(data,3)>>1+1, ...)
        range_exprs = ntuple(D) do d
            if d == H[1]
                :(1:1)
            elseif d == half_dim
                :(2:(Base.size(data, $d) >> 1) + 1)
            elseif d in active_dims
                :(2:Base.size(data, $d))
            elseif d in secondary
                :(1:1)
            else
                :(1:Base.size(data, $d))
            end
        end

        # per-dim neg-index literals (e.g. I[1], Base.size(data,2)-I[2]+2, ...)
        neg_exprs = ntuple(d -> d in active_dims ? :(Base.size(data, $d) - I[$d] + 2) : :(I[$d]), D)

        push!(blocks, quote
            for I in CartesianIndices($(Expr(:tuple, range_exprs...)))
                neg       = $(Expr(:call, :CartesianIndex, neg_exprs...))
                _av       = _average_complex(data[I], data[neg])
                data[I]   = _av
                data[neg] = conj(_av)
            end
        end)
    end

    return Base.remove_linenums!(quote
        $(blocks...)
        return data
    end)
end
apply_symmetry!(data, H::Dims) = apply_symmetry!(data, Val(H))

"""
    normalise_mean!(data::Array{<:Any, D}, ::Val{H})

Enforce purely real values at mean component of Fourier transformed
data.
"""
@generated function normalise_mean!(data::Array{<:Any, D}, ::Val{H}) where {D, H}
    indxs = [d ∈ H ? 1 : :(:) for d in 1:D]
    slice = Expr(:ref, :data, indxs...)
    return :(@views $slice .= real.($slice); return data)
end
normalise_mean!(data, H::Dims) = normalise_mean!(data, Val(H))
