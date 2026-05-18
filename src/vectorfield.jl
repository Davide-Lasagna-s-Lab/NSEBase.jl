# This file contains the concrete implementation of the vector fields based
# based on the abstract scalar field defined elsewhere.

"""
    VectorField{N, S} where {S<:Union{FTField, Field}}

A vectorfield made up of an ordered list of scalar fields either
represented using `FTField` or `Field`.

# Fields
- `elements`: scalar field components of the vectorfield
"""
struct VectorField{N, S} <: AbstractVector{S}
    elements::NTuple{N, S}

    function VectorField(elements::Vararg{S, N}) where {S<:Union{FTField, Field}, N}
        new{N, S}(elements)
    end
end

"""
    add_base_flow!(u::VectorField, base::NTuple{N}) -> u

Add the laminar base flow `base` to the zero-wavenumber slice of the spectral
vector field `u` in-place, recovering the full (base + perturbation) velocity.

`base` has one entry per velocity component.  Each entry is either a vector of
length `Ny` (values at the inhomogeneous grid points) or `nothing` (component
is skipped, i.e. no base flow in that direction).

# Index construction

For a field stored as a `D`-dimensional spectral array, the zero-wavenumber
slice is selected by the index tuple `idx` built from the grid's `fft_dims`:

- FFT dimensions (homogeneous, transformed): index `1`, the DC bin.
  FFTW always places the zero-frequency coefficient first, for both the real
  FFT (rfft) and the full complex FFT.
- Inhomogeneous dimensions: `Colon()`, selecting all grid points.

For a 4D channel grid with `CHANNEL_FFT_ORDER = (2, 3, 4)` this gives
`idx = (Colon(), 1, 1, 1)`, i.e. `parent(u[n])[:, 1, 1, 1]` — the full
wall-normal profile at zero streamwise, spanwise, and temporal wavenumber.

# Example

```julia
# Channel flow: Couette base U(y)=y, no base in v or w.
add_base_flow!(u, (U, nothing, nothing))

# Equivalent to the original hand-written channel specialisation:
#   u[1][:, 1, 1, 1] .+= U
```
"""
function add_base_flow!(u::VectorField{N, <:FTField{<:AbstractGrid}}, base::NTuple{N}) where {N}
    g   = grid(u)
    # For each array dimension d: FFT dims use index 1 (DC bin), inhomogeneous
    # dims use Colon() to span all grid points.
    idx = ntuple(d -> d ∈ fft_dims(g) ? 1 : Colon(), ndims(parent(u[1])))
    for n in 1:N
        base[n] !== nothing && (parent(u[n])[idx...] .+= base[n])
    end
    return u
end

grid(u::VectorField) = grid(u[1])

"""
    growto(u::VectorField, target_size)

Return an equivalent vector field with a new homogeneous resolution.

This is optional.  Packages should only implement it if they support
resolution-changing transforms such as `FFT(u, target_size)` and
`IFFT(û, target_size)`.
"""
growto(u::VectorField{N, <:FTField}, target_size) where {N} = VectorField([growto(u[n], target_size) for n in 1:N]...)


# ------------------- #
# constructor methods #
# ------------------- #
VectorField(g::AbstractGrid, ::Type{S}=FTField; N::Int=3, kwargs...) where {S} = VectorField([S(g; kwargs...) for _ in 1:N]...)
VectorField(g::AbstractGrid, funcs...         ; dealias::Bool=false)           = VectorField([Field(g, f; dealias=dealias) for f in funcs]...)


# ------------- #
# array methods #
# ------------- #
Base.IndexStyle(::Type{<:VectorField})                               = Base.IndexLinear()
Base.parent(q::VectorField)                                          = q.elements
Base.getindex(q::VectorField, i::Int)                                = parent(q)[i]
Base.setindex!(q::VectorField{N, F}, v::F, i::Int) where {N, F}      = (parent(q)[i] .= v; return v)
Base.size(::VectorField{N}) where {N}                                = (N,)
Base.eltype(::VectorField{N, F}) where {N, F}                        = F
Base.similar(q::VectorField{N}, ::Type{T}=eltype(q[1])) where {N, T} = VectorField([Base.similar(q.elements[n], T) for n in 1:N]...)
Base.copy(q::VectorField{N}) where {N}                               = VectorField([copy(q.elements[n]) for n in 1:N]...)
Base.zero(q::VectorField{N}) where {N}                               = VectorField([zero(q.elements[n]) for n in 1:N]...)


# ------------ #
# norm methods #
# ------------ #
LinearAlgebra.dot(q::VectorField{N}, p::VectorField{N}) where {N} = sum(dot(q[n], p[n]) for n in 1:N)
LinearAlgebra.norm(q::VectorField)                                = sqrt(dot(q, q))

function normdiff(u::VectorField{N, <:FTField{G}}, v::VectorField{N, <:FTField{G}},
                  shifts=ntuple(Returns(0), length(ORDER)),
                  tmp=nothing) where {N, G<:AbstractGrid{T, D, AXES, ORDER}} where {T, D, AXES, ORDER}
    s = zero(real(eltype(u[1])))
    for n in 1:N
        s += normdiff(u[n], v[n], shifts, tmp)^2
    end
    return sqrt(s)
end

"""
    minnormdiff(u, v, N=(32,32,32), tmp1=zero(v), tmp2=zero(v)) -> (Real, NTuple)

Return `(min_diff, (s1, s2, s3))`: the minimum of `‖u − shift(v, shifts)‖` over
a `N[1]×N[2]×N[3]` grid of shifts covering one full period in each of the three
homogeneous directions (in `fft_dims` order), and the shift at which the minimum
is attained.

`tmp1` and `tmp2` are optional pre-allocated workspaces of the same type as `v`.
"""
function minnormdiff(u::Union{FTField{G}, VectorField{<:Any, <:FTField{G}}},
                     v::Union{FTField{G}, VectorField{<:Any, <:FTField{G}}},
                     N::NTuple{3, Int}=(32, 32, 32),
                     tmp1=zero(v),
                     tmp2=zero(v)) where {G<:AbstractGrid{T, D, AXES, ORDER}} where {T, D, AXES, ORDER}
    g        = grid(u)
    min_diff = typemax(real(eltype(u isa VectorField ? u[1] : u)))
    s_min    = ntuple(k -> zero(T), 3)

    steps = ntuple(k -> 2π / (wavenumber_scale(g, ORDER[k]) * N[k]), 3)
    zero_step = ntuple(k -> zero(steps[1]), 3)

    tmp1 .= v
    for i3 in 0:N[3]-1
        for i2 in 0:N[2]-1
            for i1 in 0:N[1]-1
                diff = normdiff(u, tmp1, ntuple(Returns(zero(T)), 3), tmp2)
                if diff < min_diff
                    min_diff = diff
                    s_min = (steps[1]*i1, steps[2]*i2, steps[3]*i3)
                end
                shift!(tmp1, Base.setindex(zero_step, steps[1], 1))
            end
            shift!(tmp1, Base.setindex(zero_step, steps[2], 2))
        end
        shift!(tmp1, Base.setindex(zero_step, steps[3], 3))
    end

    return min_diff, s_min
end
