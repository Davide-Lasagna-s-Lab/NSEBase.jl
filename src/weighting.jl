# Spectral norm weights for weighted inner products on projected fields.

"""
    FarazmandWeight{N, T}

A spectral norm weight for the Farazmand et al. weighted inner product on
projected fields. The weight at mode `n::ModeNumber{N}` is

```math
w(n) = \\frac{1}{1 + \\sum_{k=1}^{N} (\\sigma_k \\, n_k)^2}
```

where `σ_k = wavenumber_scale(g, ORDER[k])` are the physical wavenumber scales
of the grid in `fft_dims` order, and `n_k = n.ns[k]` are the signed integer
wavenumbers stored in `n`.

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

Base.getindex(A::FarazmandWeight{N}, n::ModeNumber{N}) where {N} =
    1 / (1 + sum(k -> (A.scales[k] * n.ns[k])^2, 1:N))


"""
    mul!(a::ProjectedField, A::FarazmandWeight) -> a

Apply the spectral weight `A` in-place to every coefficient of `a`.
"""
function LinearAlgebra.mul!(a::ProjectedField{G},
                             A::FarazmandWeight{N}) where {N, G<:AbstractGrid{T, D, AXES, ORDER}} where {T, D, AXES, ORDER}
    g  = grid(a)
    pa = parent(a)
    for_each_mode(g) do _, spectral...
        # Look up the weight for this mode's wavenumber vector.
        n = ModeNumber(indices_to_wavenumbers(g, spectral))
        w = A[n]
        for m in axes(a, 1)
            @inbounds pa[m, spectral...] *= w
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
                            b::ProjectedField{G}) where {N, G<:AbstractGrid{T, D, AXES, ORDER}} where {T, D, AXES, ORDER}
    s = zero(T)
    g = grid(a)
    for_each_mode(g) do one_or_two, spectral...
        # Weight for this wavenumber; one_or_two accounts for Hermitian conjugate pairs.
        n = ModeNumber(indices_to_wavenumbers(g, spectral))
        w = A[n]
        for m in axes(a, 1)
            @inbounds s += one_or_two * w * real(LinearAlgebra.dot(parent(a)[m, spectral...], parent(b)[m, spectral...]))
        end
    end
    return s / 2
end
