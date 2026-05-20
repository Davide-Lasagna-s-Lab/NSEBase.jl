# Spectral norm weights for weighted inner products on projected fields.

"""
    FarazmandWeight{N, T}

A spectral norm weight for the Farazmand et al. weighted inner product on
projected fields. The weight at wavenumber `k::WaveNumberVector{N}` is

```math
w(k) = \\frac{1}{1 + \\sum_{j=1}^{N} (\\sigma_j \\, k_j)^2}
```

where `σ_j = wavenumber_scale(g, ORDER[j])` are the physical wavenumber scales
of the grid in `fft_dims` order, and `k_j = k[j]` are the signed integer
wavenumbers stored in `k`.

# Constructor

    FarazmandWeight(g::AbstractGrid)

Build the weight from the grid's wavenumber scales.
"""
struct FarazmandWeight{N, T<:Real}
    scales::NTuple{N, T}
end

function FarazmandWeight(g::AbstractGrid{T, D, AXES, ORDER}) where {T, D, AXES, ORDER}
    FarazmandWeight(ntuple(k -> T(wavenumber_scale(g, ORDER[k])), length(ORDER)))
end

Base.getindex(A::FarazmandWeight{N}, k::WaveNumberVector{N}) where {N} =
    1 / (1 + sum(j -> (A.scales[j] * k[j])^2, 1:N))


"""
    lmul!(A::FarazmandWeight, a::ProjectedField) -> a

Apply the spectral weight `A` in-place to every coefficient of `a`.
"""
function LinearAlgebra.lmul!(A::FarazmandWeight{N},
                             a::ProjectedField{G}) where {N, G<:AbstractGrid}
    g  = grid(a)
    pa = parent(a)
    for_each_wavenumber(g) do _, homogeneous_indices...
        # Look up the weight for this wavenumber vector.
        k = to_wavenumber_vector(g, homogeneous_indices)
        w = A[k]
        # the index over modes is the first in the array a so this loop ordering is efficient
        for m in axes(a, 1)
            @inbounds pa[m, homogeneous_indices...] *= w
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
    for_each_wavenumber(g) do one_or_two, homogeneous_indices...
        # Weight for this wavenumber; one_or_two accounts for Hermitian conjugate pairs.
        k = to_wavenumber_vector(g, homogeneous_indices)
        w = A[k]
        for m in axes(a, 1)
            @inbounds s += one_or_two * w * real(LinearAlgebra.dot(parent(a)[m, homogeneous_indices...], parent(b)[m, homogeneous_indices...]))
        end
    end
    return s / 2
end
