# Spectral norm weights for weighted inner products on projected fields.

"""
    FarazmandWeight{N, T}

A spectral norm weight for the Farazmand et al. weighted inner product on
projected fields.  The weight at signed wavenumber `k::WaveNumberVector{N}` is

```math
w(k) = \\frac{1}{1 + \\sum_{j=1}^{N} (\\sigma_j \\, k_j)^2}.
```

# Scale ordering — important

Both `scales` and `k` are indexed in **`fft_dims(g) = FFT_DIMS_ORDER` order**, not in
physical-coordinate order.  For a grid whose `FFT_DIMS_ORDER = (2, 3, 4)` corresponds
to coordinates `(x, z, t)`, `scales[1]` is the streamwise scale, `scales[2]`
the spanwise scale, and `scales[3]` the temporal scale.

When `FFT_DIMS_ORDER` does not start at array dimension 1 (e.g. for a channel grid
where the wall-normal direction lives at array dimension 1) the inhomogeneous
direction is *absent* from `scales` and `k` entirely: those tuples enumerate
only the homogeneous (FFT) dimensions in `FFT_DIMS_ORDER` order.

# Constructors

    FarazmandWeight(g::AbstractGrid)
    FarazmandWeight(σ::Real, σs::Real...)

The grid form builds the weight from `wavenumber_scale(g, FFT_DIMS_ORDER[k])` for each
homogeneous dimension `k`.  The varargs form takes the scales explicitly,
promoting them to a common `Real` type — useful when the desired scales
differ from those returned by the grid (e.g. when working on a non-physical
re-scaling of the grid).  Order of the positional arguments must match
`fft_dims(g)`.
"""
struct FarazmandWeight{N, T<:Real}
    scales::NTuple{N, T}
end

function FarazmandWeight(g::AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}) where {T, D, AXES, FFT_DIMS_ORDER}
    FarazmandWeight(ntuple(k -> T(wavenumber_scale(g, FFT_DIMS_ORDER[k])), length(FFT_DIMS_ORDER)))
end

FarazmandWeight(σ::Real, σs::Real...) = FarazmandWeight(promote(σ, σs...))

"""
    A[k::WaveNumberVector] -> Real

Evaluate the Farazmand weight at signed wavenumber `k`:

```math
w(k) = \\frac{1}{1 + \\sum_{j=1}^{N} (\\sigma_j \\, k_j)^2}
```

Both `A.scales` and `k` are interpreted in the grid's `fft_dims = FFT_DIMS_ORDER` order
(see the [`FarazmandWeight`](@ref) docstring), so `k[j]` is the signed integer
wavenumber along the `j`-th homogeneous dimension.
"""
Base.getindex(A::FarazmandWeight{N}, k::WaveNumberVector{N}) where {N} =
    1 / (1 + sum(j -> (A.scales[j] * k[j])^2, 1:N))


"""
    lmul!(A::FarazmandWeight, a::ProjectedField) -> a

Apply the spectral weight `A` in-place to every coefficient of `a`, multiplying
each `a[m, k]` by `A[k]`, and return `a`.
"""
function LinearAlgebra.lmul!(A::FarazmandWeight{N},
                             a::ProjectedField{G}) where {N, G<:AbstractGrid}
    g = grid(a)
    # Scale every Galerkin coefficient at this wavenumber by the Farazmand weight.
    for_each_homogeneous_index(g) do _, homogeneous_indices...
        k = to_wavenumber_vector(g, homogeneous_indices)
        w = A[k]
        for m in axes(a, 1)
            @inbounds a[m, homogeneous_indices...] *= w
        end
    end
    return a
end

"""
    dot(a::ProjectedField, A::FarazmandWeight, b::ProjectedField) -> Real

Compute the `A`-weighted inner product

```math
\\langle a, b \\rangle_A = \\frac{1}{2} \\sum_{\\mathbf{k}} c_{k_1}\\,
    w(\\mathbf{k})\\, \\sum_m \\operatorname{Re}\\!\\bigl(
    \\bar{a}_{m,\\mathbf{k}}\\, b_{m,\\mathbf{k}}\\bigr)
```

where `c_{k_1}` is `1` for the zero rfft wavenumber and `2` otherwise.
"""
function LinearAlgebra.dot(a::ProjectedField{G},
                           A::FarazmandWeight{N},
                           b::ProjectedField{G}) where {N, G<:AbstractGrid}
    T = real(eltype(a))
    s = zero(T)
    g = grid(a)
    # Accumulate the weighted inner product over all wavenumbers and Galerkin modes.
    # one_or_two accounts for rfft Hermitian symmetry (+k stored, −k implicit).
    for_each_homogeneous_index(g) do one_or_two, homogeneous_indices...
        k = to_wavenumber_vector(g, homogeneous_indices)
        w = A[k]
        for m in axes(a, 1)
            @inbounds s += one_or_two * w * real(LinearAlgebra.dot(a[m, homogeneous_indices...], b[m, homogeneous_indices...]))
        end
    end
    return s / 2
end
