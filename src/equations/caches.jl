# Shared helpers for the Cartesian NSE/LNSE operators.
#
# Two things are factored out here so the per-family operator files stay free
# of boilerplate:
#   - `alloc_caches`  : allocate the spectral and physical scratch-cache pools.
#   - `vorticity!` / `neg_cross!` : the physical/spectral kernels the rotational
#     form of the three-component 3D operators reuses.

"""
    alloc_caches(g, ncomp, nspectral, nphysical; dealias=true) -> (scache, pcache)

Allocate the scratch-cache pools shared by an NSE/LNSE operator.

`scache` is a length-`nspectral` vector of spectral [`VectorField`](@ref)s (each
of `ncomp` [`FTField`](@ref) components on the undealiased grid); `pcache` is a
length-`nphysical` vector of physical `VectorField`s (each of `ncomp`
[`Field`](@ref) components, on the dealiased grid when `dealias=true`).

Centralising this replaces the repeated comprehensions in every operator
constructor. Cache pools are sized for the advective form, which is the largest;
the leaner divergence/rotational forms use a subset of the same pool.
"""
function alloc_caches(g::AbstractGrid, ncomp::Integer,
                      nspectral::Integer, nphysical::Integer; dealias::Bool=true)
    scache = [VectorField(ntuple(_ -> FTField(g),                 ncomp)...) for _ in 1:nspectral]
    pcache = [VectorField(ntuple(_ -> Field(g; dealias=dealias),  ncomp)...) for _ in 1:nphysical]
    return scache, pcache
end

"""
    vorticity!(W, u, tmp) -> W

Compute the spectral vorticity `W = ∇×u` of a 3-component spectral velocity
field, using `tmp` (a spectral [`FTField`](@ref)) as scratch:

    ω_x = ∂y u_z − ∂z u_y
    ω_y = ∂z u_x − ∂x u_z
    ω_z = ∂x u_y − ∂y u_x

Used by the rotational advection form of the 3-component families.
"""
function vorticity!(W::VectorField{3}, u::VectorField{3}, tmp::FTField)
    ddy!(W[1], u[3]); ddz!(tmp, u[2]); W[1] .-= tmp
    ddz!(W[2], u[1]); ddx!(tmp, u[3]); W[2] .-= tmp
    ddx!(W[3], u[2]); ddy!(tmp, u[1]); W[3] .-= tmp
    return W
end

"""
    neg_cross!(C, A, B) -> C

Write the negated cross product `C = −(A × B) = B × A` of two 3-component
physical fields into `C`:

    C_1 = A_3 B_2 − A_2 B_3
    C_2 = A_1 B_3 − A_3 B_1
    C_3 = A_2 B_1 − A_1 B_2

The rotational form forms `C = −(ω × u)` so that adding its forward transform to
`out` contributes the `−(ω × u)` advection term directly.
"""
function neg_cross!(C::VectorField{3}, A::VectorField{3}, B::VectorField{3})
    @. C[1] = A[3]*B[2] - A[2]*B[3]
    @. C[2] = A[1]*B[3] - A[3]*B[1]
    @. C[3] = A[2]*B[1] - A[1]*B[2]
    return C
end
