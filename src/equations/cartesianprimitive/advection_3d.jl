# Advection helpers for 3D Cartesian primitive velocity.
#
# Organised by role: nonlinear, then linearised (forward), then continuous
# adjoint, then discrete adjoint; within each role, by advection form.
#
# The advective form is generic over NCOMP (NCOMP=3 is plain velocity; NCOMP=4
# also advects a scalar — the Boussinesq temperature — since (u·∇)θ has the same
# structure as (u·∇)uᵢ). The divergence/rotational forms are velocity-only
# (NCOMP=3): rotational is `ω×u`, which has no scalar analogue.


# ------------------------------------------------------------------ #
# rotational-form kernels                                            #
# ------------------------------------------------------------------ #
"""
    vorticity!(W, u, tmp) -> W

Spectral vorticity `W = ∇×u` of a 3-component field (`tmp` is spectral scratch):
`ω_x = ∂y u_z − ∂z u_y`, `ω_y = ∂z u_x − ∂x u_z`, `ω_z = ∂x u_y − ∂y u_x`.
"""
function vorticity!(W::VectorField{3}, u::VectorField{3}, tmp::FTField)
    ddy!(W[1], u[3]); ddz!(tmp, u[2]); W[1] .-= tmp
    ddz!(W[2], u[1]); ddx!(tmp, u[3]); W[2] .-= tmp
    ddx!(W[3], u[2]); ddy!(tmp, u[1]); W[3] .-= tmp
    return W
end

"""
    neg_cross!(C, A, B) -> C

Negated cross product `C = −(A × B) = B × A` of two 3-component physical fields.
The rotational form builds `C = −(ω × u)` so adding its transform to `out` gives
the advection term directly.
"""
function neg_cross!(C::VectorField{3}, A::VectorField{3}, B::VectorField{3})
    @. C[1] = A[3]*B[2] - A[2]*B[3]
    @. C[2] = A[1]*B[3] - A[3]*B[1]
    @. C[3] = A[2]*B[1] - A[1]*B[2]
    return C
end


# ================================================================== #
# NONLINEAR                                                          #
# ================================================================== #
# advective: N_i = u_j ∂_j u_i, generic over NCOMP.
function advection!(out, u, eq::CartesianPrimitiveNSE{3, NCOMP}, ::Advective) where {NCOMP}
    dudx = eq.scache[1]; dudy = eq.scache[2]; dudz = eq.scache[3]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]

    ddx!(dudx, u); ddy!(dudy, u); ddz!(dudz, u)
    eq.plans(U, u); eq.plans(dUdx, dudx); eq.plans(dUdy, dudy); eq.plans(dUdz, dudz)
    for n in 1:NCOMP
        @. dUdx[n] = -U[1]*dUdx[n] - U[2]*dUdy[n] - U[3]*dUdz[n]
    end
    eq.plans(out, dUdx, add=true)
    return out
end

# divergence: N_i = ∂_j(u_i u_j)
function advection!(out, u, eq::CartesianPrimitiveNSE{3, 3}, ::Divergence)
    sA = eq.scache[1]; sB = eq.scache[2]; t = eq.scache[3][1]
    U  = eq.pcache[1]; P = eq.pcache[2][1]

    eq.plans(U, u)
    @. P = U[1]*U[1]; eq.plans(sA[1], P)
    @. P = U[2]*U[2]; eq.plans(sA[2], P)
    @. P = U[3]*U[3]; eq.plans(sA[3], P)
    @. P = U[1]*U[2]; eq.plans(sB[1], P)
    @. P = U[1]*U[3]; eq.plans(sB[2], P)
    @. P = U[2]*U[3]; eq.plans(sB[3], P)

    ddx!(t, sA[1]); out[1] .-= t
    ddy!(t, sB[1]); out[1] .-= t
    ddz!(t, sB[2]); out[1] .-= t
    ddx!(t, sB[1]); out[2] .-= t
    ddy!(t, sA[2]); out[2] .-= t
    ddz!(t, sB[3]); out[2] .-= t
    ddx!(t, sB[2]); out[3] .-= t
    ddy!(t, sB[3]); out[3] .-= t
    ddz!(t, sA[3]); out[3] .-= t
    return out
end

# rotational: N = ω×u  (the ∇(½|u|²) gradient is dropped — removed by projection)
function advection!(out, u, eq::CartesianPrimitiveNSE{3, 3}, ::Rotational)
    W  = eq.scache[1]; t = eq.scache[2][1]
    U  = eq.pcache[1]; Om = eq.pcache[2]; C = eq.pcache[3]

    vorticity!(W, u, t)
    eq.plans(U, u); eq.plans(Om, W)
    neg_cross!(C, Om, U)
    eq.plans(out, C, add=true)
    return out
end


# ================================================================== #
# LINEARISED (forward)                                              #
# ================================================================== #
# --- base-flow setup (shared by every linearised mode) ---
function lnse_setup!(u, eq::CartesianPrimitiveNSE{3, NCOMP}, ::Advective) where {NCOMP}
    dudx = eq.scache[1]; dudy = eq.scache[2]; dudz = eq.scache[3]
    U    = eq.pcache[1]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]
    ddx!(dudx, u); ddy!(dudy, u); ddz!(dudz, u)
    eq.plans(U, u); eq.plans(dUdy, dudy); eq.plans(dUdz, dudz)
    return nothing
end

# divergence reuses the advective base-flow caching.
lnse_setup!(u, eq::CartesianPrimitiveNSE{3, 3}, ::Divergence) = lnse_setup!(u, eq, Advective())

# rotational additionally caches the base vorticity Ω (in pcache[5]).
function lnse_setup!(u, eq::CartesianPrimitiveNSE{3, 3}, ::Rotational)
    lnse_setup!(u, eq, Advective())
    W  = eq.scache[4]; t = eq.scache[2][1]
    Om = eq.pcache[5]
    vorticity!(W, u, t)
    eq.plans(Om, W)
    return nothing
end

# --- forward advection ---
# advective: L(v)_i = (U·∇)v_i + (v·∇)U_i, generic over NCOMP.
function linearised_advection!(out, v, eq::CartesianPrimitiveNSE{3, NCOMP}, ::Advective) where {NCOMP}
    dudx = eq.scache[1]; dvdx = eq.scache[2]; dvdy = eq.scache[3]; dvdz = eq.scache[4]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]
    V    = eq.pcache[5]; dVdx = eq.pcache[6]; dVdy = eq.pcache[7]; dVdz = eq.pcache[8]

    ddx!(dvdx, v); ddy!(dvdy, v); ddz!(dvdz, v)
    eq.plans(V, v); eq.plans(dUdx, dudx)
    eq.plans(dVdx, dvdx); eq.plans(dVdy, dvdy); eq.plans(dVdz, dvdz)
    for n in 1:NCOMP
        @. dVdx[n]  = -U[1]*dVdx[n] - U[2]*dVdy[n] - U[3]*dVdz[n]
        @. dVdx[n] -=  V[1]*dUdx[n] + V[2]*dUdy[n] + V[3]*dUdz[n]
    end
    eq.plans(out, dVdx, add=true)
    return out
end

# divergence: L(v)_i = ∂_j(U_i v_j + U_j v_i)
function linearised_advection!(out, v, eq::CartesianPrimitiveNSE{3, 3}, ::Divergence)
    sA = eq.scache[1]; sB = eq.scache[2]; t = eq.scache[3][1]
    U  = eq.pcache[1]; V = eq.pcache[5]; P = eq.pcache[6][1]

    eq.plans(V, v)
    @. P = 2*U[1]*V[1];          eq.plans(sA[1], P)
    @. P = 2*U[2]*V[2];          eq.plans(sA[2], P)
    @. P = 2*U[3]*V[3];          eq.plans(sA[3], P)
    @. P = U[1]*V[2] + U[2]*V[1]; eq.plans(sB[1], P)
    @. P = U[1]*V[3] + U[3]*V[1]; eq.plans(sB[2], P)
    @. P = U[2]*V[3] + U[3]*V[2]; eq.plans(sB[3], P)

    ddx!(t, sA[1]); out[1] .-= t
    ddy!(t, sB[1]); out[1] .-= t
    ddz!(t, sB[2]); out[1] .-= t
    ddx!(t, sB[1]); out[2] .-= t
    ddy!(t, sA[2]); out[2] .-= t
    ddz!(t, sB[3]); out[2] .-= t
    ddx!(t, sB[2]); out[3] .-= t
    ddy!(t, sB[3]); out[3] .-= t
    ddz!(t, sA[3]); out[3] .-= t
    return out
end

# rotational: L(v) = (∇×v)×U + Ω×v
function linearised_advection!(out, v, eq::CartesianPrimitiveNSE{3, 3}, ::Rotational)
    Wv = eq.scache[1]; t = eq.scache[2][1]
    U  = eq.pcache[1]; Om  = eq.pcache[5]
    V  = eq.pcache[6]; OmV = eq.pcache[7]; C = eq.pcache[8]

    vorticity!(Wv, v, t)
    eq.plans(V, v); eq.plans(OmV, Wv)
    neg_cross!(C, OmV, U)
    @. C[1] += Om[3]*V[2] - Om[2]*V[3]
    @. C[2] += Om[1]*V[3] - Om[3]*V[1]
    @. C[3] += Om[2]*V[1] - Om[1]*V[2]
    eq.plans(out, C, add=true)
    return out
end


# ================================================================== #
# CONTINUOUS ADJOINT                                                 #
# ================================================================== #
# advective: +(U·∇)w − Σ_i w_i ∇U_i. The cross-gradient sum runs over all NCOMP
# components but only contributes to the NDIM velocity adjoint components.
function adjoint_continuous_advection!(out, v, eq::CartesianPrimitiveNSE{3, NCOMP}, ::Advective) where {NCOMP}
    dudx = eq.scache[1]; dvdx = eq.scache[2]; dvdy = eq.scache[3]; dvdz = eq.scache[4]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]
    V    = eq.pcache[5]; dVdx = eq.pcache[6]; dVdy = eq.pcache[7]; dVdz = eq.pcache[8]

    ddx!(dvdx, v); ddy!(dvdy, v); ddz!(dvdz, v)
    eq.plans(V, v); eq.plans(dUdx, dudx)
    eq.plans(dVdx, dvdx); eq.plans(dVdy, dvdy); eq.plans(dVdz, dvdz)
    for n in 1:NCOMP
        @. dVdx[n] = U[1]*dVdx[n] + U[2]*dVdy[n] + U[3]*dVdz[n]
    end
    dVdz .= 0
    for i in 1:NCOMP
        @. dVdz[1] -= V[i]*dUdx[i]
        @. dVdz[2] -= V[i]*dUdy[i]
        @. dVdz[3] -= V[i]*dUdz[i]
    end
    eq.plans(out, dVdx, add=true); eq.plans(out, dVdz, add=true)
    return out
end

# divergence: +(U·∇)w in conservative form ∂_j(U_j w_i), plus the cross-gradient.
function adjoint_continuous_advection!(out, v, eq::CartesianPrimitiveNSE{3, 3}, ::Divergence)
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]
    V    = eq.pcache[5]; P    = eq.pcache[6][1]; G = eq.pcache[7]

    eq.plans(dUdx, eq.scache[1])
    eq.plans(V, v)

    t  = eq.scache[1][1]
    sP = eq.scache[2]
    for i in 1:3
        @. P = U[1]*V[i]; eq.plans(sP[1], P); ddx!(t, sP[1]); out[i] .+= t
        @. P = U[2]*V[i]; eq.plans(sP[1], P); ddy!(t, sP[1]); out[i] .+= t
        @. P = U[3]*V[i]; eq.plans(sP[1], P); ddz!(t, sP[1]); out[i] .+= t
    end

    G .= 0
    for i in 1:3
        @. G[1] -= V[i]*dUdx[i]
        @. G[2] -= V[i]*dUdy[i]
        @. G[3] -= V[i]*dUdz[i]
    end
    eq.plans(out, G, add=true)
    return out
end

# rotational has no distinct continuous-adjoint form; reuse the divergence one.
adjoint_continuous_advection!(out, v, eq::CartesianPrimitiveNSE{3, 3}, ::Rotational) =
    adjoint_continuous_advection!(out, v, eq, Divergence())


# ================================================================== #
# DISCRETE ADJOINT (advective only; div/rot are a future addition)  #
# ================================================================== #
function adjoint_discrete_advection!(out, v, eq::CartesianPrimitiveNSE{3, NCOMP}, ::Advective) where {NCOMP}
    dudx = eq.scache[1]; u1v = eq.scache[2]; u2v = eq.scache[3]; u3v = eq.scache[4]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]
    V    = eq.pcache[5]; U1V = eq.pcache[6]; U2V = eq.pcache[7]; U3V = eq.pcache[8]

    eq.plans(V, v)
    for n in 1:NCOMP
        @. U1V[n] = U[1]*V[n]
        @. U2V[n] = U[2]*V[n]
        @. U3V[n] = U[3]*V[n]
    end
    eq.plans(u1v, U1V); eq.plans(u2v, U2V); eq.plans(u3v, U3V)

    eq.plans(dUdx, dudx)
    for n in 1:NCOMP
        out[n] .-= ddx!(dudx[1], u1v[n], adjoint=true) .+
                   ddy!(dudx[2], u2v[n], adjoint=true) .+
                   ddz!(dudx[3], u3v[n], adjoint=true)
    end
    U1V .= 0
    for n in 1:NCOMP
        @. U1V[1] -= V[n]*dUdx[n]
        @. U1V[2] -= V[n]*dUdy[n]
        @. U1V[3] -= V[n]*dUdz[n]
    end
    eq.plans(out, U1V, add=true)
    return out
end
