# Spectral norm weight for Farazmand-style weighted inner products on ProjectedField.
#
# When searching for exact coherent structures in wall-bounded flows, it is
# useful to weight the optimisation inner product so that small-scale (high
# wavenumber) structures are penalised relative to large-scale ones.  This
# avoids iterates converging to spurious high-frequency noise.
#
# `FarazmandWeight` implements the weight function
#
#   w(k) = 1 / (1 + Σ_j (σ_j · k_j)²)
#
# where σ_j is a user-supplied length scale for each homogeneous direction and
# k_j is the signed wavenumber.  The weight is 1 at k=0 and decays to zero at
# high wavenumbers.  The scales σ can be read from the grid (default) or
# supplied explicitly for non-physical re-scalings.
#
# Two operations are provided:
#   lmul!(A, a)       — multiply every coefficient of a by w(k), in-place.
#   dot(a, A, b)      — compute ⟨a, b⟩_A = Σ_k w(k) Re(ā · b) / 2.

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

    # Scale every Galerkin coefficient by the Farazmand weight.
    # Outer loop: wavenumber k — one weight value per kH combination.
    # Inner loop: mode index m — same weight applies to all modes at k.
    for Ih in CartesianIndices(homogeneous_axes(a))
        k = to_wavenumber_vector(g, Tuple(Ih))
        for m in axes(parent(a), 1)
            @inbounds parent(a)[m, Ih] *= A[k]
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
    s = zero(real(eltype(a)))
    g = grid(a)

    # Outer loop: wavenumber k — one weight and one rfft multiplicity per kH.
    # Inner loop: mode m — sum over all modes at each k.
    for Ih in CartesianIndices(homogeneous_axes(a))
        k          = to_wavenumber_vector(g, Tuple(Ih))
        # rfft zero wavenumber is self-conjugate (counted once); all others
        # have an implicit negative-kx partner (counted twice).
        one_or_two = Ih[1] == 1 ? 1 : 2
        w = one_or_two * A[k]
        for m in axes(parent(a), 1)
            I = CartesianIndex(m, Ih.I...)
            @inbounds s += w * real(LinearAlgebra.dot(a[I], b[I]))
        end
    end

    return s / 2
end
