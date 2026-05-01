# Fourier transformed scalar field.

# TODO: implement some sort of norm interface?

"""
    FTField{G} where {G<:AbstractGrid}

Fourier transformed representation of a scalar field on
a concrete sub-type of AbstractGrid.

# Fields
- `grid`: concrete instance of `AbstractGrid`
- `data`: Fourier coefficient values of scalar field
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
 
Apply hermitian symmetry to a D-dimensional array `data` whose axes listed
in the tuple `H` have been RFFT-transformed, in the order they were applied.
"""
@generated function apply_symmetry!(data::AbstractArray{T, D}, ::Val{H}) where {T, D, H}
    # check if symmetry needs special treatment
    length(H) == 1 && return :(return data)

    # generate loops
    blocks = Expr[]
    for mask_int in 0:(1 << length(H[2:end])) - 1
        active_mask = ntuple(k -> Bool((mask_int >> (k-1)) & 1), length(H[2:end]))
        all(!, active_mask) && length(H[2:end]) > 0 && continue
        push!(blocks, make_case(active_mask, D, H))
    end

    # assemble and return
    return quote
        $(blocks...)
        return data
    end
end
apply_symmetry!(data, H::Dims) = apply_symmetry!(data, Val(H))

function make_case(active_mask, D, H)
    # determine which directions are being looped over
    active_syms = [H[2:end][k] for k in 1:length(H[2:end]) if active_mask[k]]
    half_range_dim = isempty(active_syms) ? nothing : active_syms[end]

    # forward and negative frequency indices
    idx_pos = Vector{Any}(undef, D)
    idx_neg = Vector{Any}(undef, D)
    loop_vars = Dict{Int, Symbol}()
    for d in 1:D
        if d == H[1]
            idx_pos[d] = 1
            idx_neg[d] = 1
        elseif d in active_syms
            var = Symbol("n_", d)
            loop_vars[d] = var
            idx_pos[d] = var
            idx_neg[d] = :($(Symbol("end")) - $var + 2)
        elseif d in H[2:end]
            idx_pos[d] = 1
            idx_neg[d] = 1
        else
            var = Symbol("n_", d)
            loop_vars[d] = var
            idx_pos[d] = var
            idx_neg[d] = var
        end
    end

    # averaging kernel
    kernel = quote
        _av  = _average_complex(data[$(idx_pos...)], data[$(idx_neg...)])
        data[$(idx_pos...)] =      _av
        data[$(idx_neg...)] = conj(_av)
    end

    # generate loop block
    looped_dims = sort(collect(keys(loop_vars)))
    body = kernel
    for d in looped_dims
        var = loop_vars[d]
        if d == half_range_dim
            range = :(2:(Base.size(data, $d) >> 1) + 1)
        elseif d in active_syms
            range = :(2:Base.size(data, $d))
        else
            range = :(1:Base.size(data, $d))
        end
        body = :(for $var in $range; $body; end)
    end

    return body
end

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
