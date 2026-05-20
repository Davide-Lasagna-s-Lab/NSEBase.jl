# Galerkin projection and reconstruction operators for ProjectedField.

"""
    get_mode_coefficient(modes, grid::AbstractGrid, n::Int, m::Int,
                         inhomogeneous_indices::NTuple, homogeneous_indices...) -> Number

Return the complex coefficient of basis mode `m` for vector field component `n`
at inhomogeneous storage indices `inhomogeneous_indices` (one per inhomogeneous
dimension in ascending array-dimension order) and homogeneous storage indices
`homogeneous_indices...` (in `fft_dims` order).

Downstream packages must extend this for their specific `modes` representation.
"""
get_mode_coefficient(modes, grid::AbstractGrid, n::Int, m::Int, inhomogeneous_indices::NTuple, homogeneous_indices...) =
    throw(NotImplementedError(modes, grid, n, m, inhomogeneous_indices))

raw"""
    project!(a::ProjectedField, u::VectorField) -> a

Project the spectral vector field `u` onto the basis stored by `a`:

```math
a_{m,\mathbf{k}} = \sum_n \sum_\mathbf{j} w_\mathbf{j}
    \overline{\phi_{n,m,\mathbf{j},\mathbf{k}}} u_{n,\mathbf{j},\mathbf{k}}
```

where `φ` is returned by [`get_mode_coefficient`](@ref), `w` comes from
[`weights`](@ref), `j` indexes inhomogeneous dimensions, and `k` indexes
spectral dimensions in `fft_dims(grid)`.
"""
function project!(a::ProjectedField{G},
                  u::VectorField{N, <:FTField{G}}) where {N, G<:AbstractGrid{T}} where {T}
    a .= zero(Complex{T})
    g        = grid(u)
    w        = weights(g)
    inh_size = map(d -> size(g, d), inhomogeneous_dims(g))
    # For each stored wavenumber, accumulate the weighted inner product of u
    # against each basis mode φ over all inhomogeneous (wall-normal) locations.
    for_each_homogeneous_index(g) do _, homogeneous_indices...
        for m in axes(a, 1)
            acc = zero(Complex{T})
            for I in CartesianIndices(inh_size)
                inhomogeneous_indices = Tuple(I)
                # Full D-dimensional index into the FTField parent array.
                indices = _combine_indices(g, inhomogeneous_indices, homogeneous_indices)
                for n in 1:N
                    @inbounds acc += w[I] * conj(get_mode_coefficient(modes(a), g, n, m, inhomogeneous_indices, homogeneous_indices...)) * parent(u[n])[indices...]
                end
            end
            @inbounds a[m, homogeneous_indices...] = acc
        end
    end
    return a
end

"""
    project(u::VectorField, modes) -> ProjectedField

Allocate a `ProjectedField` and project `u` onto `modes`.  See [`project!`](@ref).
"""
project(u::VectorField, modes) = project!(ProjectedField(grid(u), modes), u)

"""
    expand!(u::VectorField, a::ProjectedField) -> u

Reconstruct the velocity field `u` from the projected coefficients `a`:

```math
u_n[\\mathbf{j}, \\mathbf{k}] = \\sum_m a_{m,\\mathbf{k}} \\, \\phi_{n,m,\\mathbf{j},\\mathbf{k}}
```

where `φ_{n,m,j,k}` is returned by [`get_mode_coefficient`](@ref) and
`j`, `k` index inhomogeneous and spectral dimensions respectively.
"""
function expand!(u::VectorField{N, <:FTField{G}},
                 a::ProjectedField{G}) where {N, G<:AbstractGrid{T, D, AXES, ORDER}} where {T, D, AXES, ORDER}
    u .= zero(Complex{T})
    g        = grid(u)
    inh_size = map(d -> size(g, d), inhomogeneous_dims(g))
    # For each stored wavenumber, scatter the projected coefficient back into
    # the FTField by adding the contribution of each basis mode at every
    # inhomogeneous (wall-normal) location.
    for_each_homogeneous_index(g) do _, homogeneous_indices...
        for m in axes(a, 1)
            coeff = @inbounds a[m, homogeneous_indices...]
            for I in CartesianIndices(inh_size)
                inhomogeneous_indices = Tuple(I)
                # Full D-dimensional index into the FTField parent array.
                indices = _combine_indices(g, inhomogeneous_indices, homogeneous_indices)
                for n in 1:N
                    @inbounds parent(u[n])[indices...] += coeff * get_mode_coefficient(modes(a), g, n, m, inhomogeneous_indices, homogeneous_indices...)
                end
            end
        end
    end
    return u
end

"""
    expand(a::ProjectedField) -> VectorField

Allocate a `VectorField` and reconstruct it from the projected coefficients `a`.
See [`expand!`](@ref).
"""
expand(a::ProjectedField) = expand!(VectorField(grid(a), FTField), a)
