# Advection helpers for the three-component Cartesian operators.
#
# Each helper writes the advection contribution into `out` (which already holds
# the viscous term). They are dispatched on the `AdvectionForm` tag; the operator
# call skeletons in operators.jl select the method via the operator's FORM type
# parameter. `_lnse_setup!` caches the base-flow quantities the chosen form needs.
#
# Cache-slot conventions follow the advective layout (scache 1:4, pcache 1:8);
# the leaner forms use a documented subset.


# ================================================================== #
# Advective form:  N_i = u_j ∂_j u_i                                  #
# ================================================================== #

function _nse_advection!(out, u, eq::CartesianPrimitive3DNSE, ::Advective)
    dudx = eq.scache[1]; dudy = eq.scache[2]; dudz = eq.scache[3]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]

    ddx!(dudx, u); ddy!(dudy, u); ddz!(dudz, u)
    eq.plans(U, u); eq.plans(dUdx, dudx); eq.plans(dUdy, dudy); eq.plans(dUdz, dudz)
    for n in 1:3
        @. dUdx[n] = -U[1]*dUdx[n] - U[2]*dUdy[n] - U[3]*dUdz[n]
    end
    eq.plans(out, dUdx, add=true)
    return out
end

# base-flow caching for the advective LNSE: physical base velocity U and the
# physical base-flow gradients ∂U/∂y, ∂U/∂z; the spectral ∂u/∂x is left in
# scache[1] for the 2-arg call to transform.
function _lnse_setup!(u, eq, ::Advective)
    dudx = eq.scache[1]; dudy = eq.scache[2]; dudz = eq.scache[3]
    U    = eq.pcache[1]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]
    ddx!(dudx, u); ddy!(dudy, u); ddz!(dudz, u)
    eq.plans(U, u); eq.plans(dUdy, dudy); eq.plans(dUdz, dudz)
    return nothing
end

function _fwd_advection!(out, v, eq, ::Advective)
    dudx = eq.scache[1]; dvdx = eq.scache[2]; dvdy = eq.scache[3]; dvdz = eq.scache[4]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]
    V    = eq.pcache[5]; dVdx = eq.pcache[6]; dVdy = eq.pcache[7]; dVdz = eq.pcache[8]

    ddx!(dvdx, v); ddy!(dvdy, v); ddz!(dvdz, v)
    eq.plans(V, v); eq.plans(dUdx, dudx)
    eq.plans(dVdx, dvdx); eq.plans(dVdy, dvdy); eq.plans(dVdz, dvdz)
    for n in 1:3
        @. dVdx[n]  = -U[1]*dVdx[n] - U[2]*dVdy[n] - U[3]*dVdz[n]
        @. dVdx[n] -=  V[1]*dUdx[n] + V[2]*dUdy[n] + V[3]*dUdz[n]
    end
    eq.plans(out, dVdx, add=true)
    return out
end

function _adjcont_advection!(out, v, eq, ::Advective)
    dudx = eq.scache[1]; dvdx = eq.scache[2]; dvdy = eq.scache[3]; dvdz = eq.scache[4]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]
    V    = eq.pcache[5]; dVdx = eq.pcache[6]; dVdy = eq.pcache[7]; dVdz = eq.pcache[8]

    ddx!(dvdx, v); ddy!(dvdy, v); ddz!(dvdz, v)
    eq.plans(V, v); eq.plans(dUdx, dudx)
    eq.plans(dVdx, dvdx); eq.plans(dVdy, dvdy); eq.plans(dVdz, dvdz)
    for n in 1:3
        @. dVdx[n] = U[1]*dVdx[n] + U[2]*dVdy[n] + U[3]*dVdz[n]
    end
    dVdz .= 0
    for i in 1:3
        @. dVdz[1] -= V[i]*dUdx[i]
        @. dVdz[2] -= V[i]*dUdy[i]
        @. dVdz[3] -= V[i]*dUdz[i]
    end
    eq.plans(out, dVdx, add=true); eq.plans(out, dVdz, add=true)
    return out
end

function _adjdisc_advection!(out, v, eq, ::Advective)
    dudx = eq.scache[1]; u1v = eq.scache[2]; u2v = eq.scache[3]; u3v = eq.scache[4]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]
    V    = eq.pcache[5]; U1V = eq.pcache[6]; U2V = eq.pcache[7]; U3V = eq.pcache[8]

    eq.plans(V, v)
    for n in 1:3
        @. U1V[n] = U[1]*V[n]
        @. U2V[n] = U[2]*V[n]
        @. U3V[n] = U[3]*V[n]
    end
    eq.plans(u1v, U1V); eq.plans(u2v, U2V); eq.plans(u3v, U3V)

    eq.plans(dUdx, dudx)
    for n in 1:3
        out[n] .-= ddx!(dudx[1], u1v[n], adjoint=true) .+
                   ddy!(dudx[2], u2v[n], adjoint=true) .+
                   ddz!(dudx[3], u3v[n], adjoint=true)
    end
    U1V .= 0
    for n in 1:3
        @. U1V[1] -= V[n]*dUdx[n]
        @. U1V[2] -= V[n]*dUdy[n]
        @. U1V[3] -= V[n]*dUdz[n]
    end
    eq.plans(out, U1V, add=true)
    return out
end

# The discrete adjoint of the divergence / rotational forms is Phase B; until
# then only the advective discrete adjoint exists.
_adjdisc_advection!(out, v, eq, form::AdvectionForm) =
    throw(ArgumentError("discrete adjoint is implemented for the advective form only; " *
                        "got $(typeof(form)). Use AdjointContinuous for divergence/rotational."))


# ================================================================== #
# Divergence form:  N_i = ∂_j(u_i u_j)                                #
# ================================================================== #
# Transform u to physical once (3 inverse), form the 6 unique products, transform
# them back (6 forward), then take the divergence with the spectral / FD
# derivatives: 9 transforms vs the advective 15.

function _nse_advection!(out, u, eq::CartesianPrimitive3DNSE, ::Divergence)
    sA = eq.scache[1]; sB = eq.scache[2]; t = eq.scache[3][1]
    U  = eq.pcache[1]; P = eq.pcache[2][1]

    eq.plans(U, u)
    @. P = U[1]*U[1]; eq.plans(sA[1], P)   # s11
    @. P = U[2]*U[2]; eq.plans(sA[2], P)   # s22
    @. P = U[3]*U[3]; eq.plans(sA[3], P)   # s33
    @. P = U[1]*U[2]; eq.plans(sB[1], P)   # s12
    @. P = U[1]*U[3]; eq.plans(sB[2], P)   # s13
    @. P = U[2]*U[3]; eq.plans(sB[3], P)   # s23

    # out -= ∂_j(u_i u_j)
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


# ================================================================== #
# Rotational form:  N = ω × u  (dropping ∇(½|u|²))                    #
# ================================================================== #
# Vorticity via the cheap spectral / FD derivatives, then velocity and vorticity
# to physical (6 inverse), the cross product, and back (3 forward): 9 transforms.

function _nse_advection!(out, u, eq::CartesianPrimitive3DNSE, ::Rotational)
    W  = eq.scache[1]; t = eq.scache[2][1]
    U  = eq.pcache[1]; Om = eq.pcache[2]; C = eq.pcache[3]

    vorticity!(W, u, t)
    eq.plans(U, u); eq.plans(Om, W)
    neg_cross!(C, Om, U)                    # C = −(ω × u)
    eq.plans(out, C, add=true)
    return out
end


# ------------------------------------------------------------------ #
# Divergence form — linearised operators                             #
# ------------------------------------------------------------------ #
# The divergence setup needs the same base-flow quantities as the advective one
# (base velocity U, physical gradients ∂U/∂y, ∂U/∂z, and the spectral ∂u/∂x left
# in scache[1]), so it reuses it directly.
_lnse_setup!(u, eq, ::Divergence) = _lnse_setup!(u, eq, Advective())

# forward:  L(v)_i = ∂_j(U_i v_j + U_j v_i)
function _fwd_advection!(out, v, eq, ::Divergence)
    sA = eq.scache[1]; sB = eq.scache[2]; t = eq.scache[3][1]
    U  = eq.pcache[1]; V = eq.pcache[5]; P = eq.pcache[6][1]

    eq.plans(V, v)
    @. P = 2*U[1]*V[1];          eq.plans(sA[1], P)   # S11
    @. P = 2*U[2]*V[2];          eq.plans(sA[2], P)   # S22
    @. P = 2*U[3]*V[3];          eq.plans(sA[3], P)   # S33
    @. P = U[1]*V[2] + U[2]*V[1]; eq.plans(sB[1], P)  # S12
    @. P = U[1]*V[3] + U[3]*V[1]; eq.plans(sB[2], P)  # S13
    @. P = U[2]*V[3] + U[3]*V[2]; eq.plans(sB[3], P)  # S23

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

# continuous adjoint:  +(U·∇)w  (divergence form ∂_j(U_j w_i))  − Σ_i w_i ∇U_i
function _adjcont_advection!(out, v, eq, ::Divergence)
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]
    V    = eq.pcache[5]; P    = eq.pcache[6][1]; G = eq.pcache[7]

    eq.plans(dUdx, eq.scache[1])      # base ∂U/∂x to physical (setup left it spectral)
    eq.plans(V, v)

    t  = eq.scache[1][1]              # scache[1] free after dUdx transform → spectral scratch
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


# ------------------------------------------------------------------ #
# Rotational form — linearised operators                             #
# ------------------------------------------------------------------ #
# The rotational forward needs the base velocity U and base vorticity Ω; the
# continuous adjoint reuses the divergence computation (the rotational identity
# applies only to self-advection), so it needs the base gradients too. The setup
# therefore caches the advective base-flow quantities plus Ω (in pcache[5]).
function _lnse_setup!(u, eq, ::Rotational)
    _lnse_setup!(u, eq, Advective())           # U, ∂U/∂y, ∂U/∂z; base ∂u/∂x in scache[1]
    W  = eq.scache[4]; t = eq.scache[2][1]
    Om = eq.pcache[5]
    vorticity!(W, u, t)
    eq.plans(Om, W)
    return nothing
end

# forward:  L(v) = (∇×v)×U + Ω×v
function _fwd_advection!(out, v, eq, ::Rotational)
    Wv = eq.scache[1]; t = eq.scache[2][1]
    U  = eq.pcache[1]; Om  = eq.pcache[5]
    V  = eq.pcache[6]; OmV = eq.pcache[7]; C = eq.pcache[8]

    vorticity!(Wv, v, t)               # ω_v = ∇×v
    eq.plans(V, v); eq.plans(OmV, Wv)
    neg_cross!(C, OmV, U)              # C = −(ω_v × U)
    @. C[1] += Om[3]*V[2] - Om[2]*V[3] # − (Ω × v)
    @. C[2] += Om[1]*V[3] - Om[3]*V[1]
    @. C[3] += Om[2]*V[1] - Om[1]*V[2]
    eq.plans(out, C, add=true)
    return out
end

# continuous adjoint: no distinct rotational form (rotational applies to
# self-advection only); use the conservative divergence computation. The
# rotational setup caches the base gradients it needs.
_adjcont_advection!(out, v, eq, ::Rotational) = _adjcont_advection!(out, v, eq, Divergence())
