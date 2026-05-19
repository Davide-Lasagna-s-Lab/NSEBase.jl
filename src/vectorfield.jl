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

