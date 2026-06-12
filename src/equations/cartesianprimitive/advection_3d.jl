# Advection helpers for 3D, three-component Cartesian primitive velocity.

function _nse_advection!(out, u, eq::CartesianPrimitiveNSE{3, 3}, ::Advective)
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

function _lnse_setup!(u, eq::CartesianPrimitiveLNSE{3, 3}, ::Advective)
    dudx = eq.scache[1]; dudy = eq.scache[2]; dudz = eq.scache[3]
    U    = eq.pcache[1]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]
    ddx!(dudx, u); ddy!(dudy, u); ddz!(dudz, u)
    eq.plans(U, u); eq.plans(dUdy, dudy); eq.plans(dUdz, dudz)
    return nothing
end

function _linearised_advection!(out, v, eq::CartesianPrimitiveLNSE{3, 3}, ::Advective)
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

function _adjcont_advection!(out, v, eq::CartesianPrimitiveLNSE{3, 3}, ::Advective)
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

function adjoint_discrete_advection!(out, v, eq::CartesianPrimitiveLNSE{3, 3}, ::Advective)
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

function _nse_advection!(out, u, eq::CartesianPrimitiveNSE{3, 3}, ::Divergence)
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

function _nse_advection!(out, u, eq::CartesianPrimitiveNSE{3, 3}, ::Rotational)
    W  = eq.scache[1]; t = eq.scache[2][1]
    U  = eq.pcache[1]; Om = eq.pcache[2]; C = eq.pcache[3]

    vorticity!(W, u, t)
    eq.plans(U, u); eq.plans(Om, W)
    neg_cross!(C, Om, U)
    eq.plans(out, C, add=true)
    return out
end

function _linearised_advection!(out, v, eq::CartesianPrimitiveLNSE{3, 3}, ::Divergence)
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

function _adjcont_advection!(out, v, eq::CartesianPrimitiveLNSE{3, 3}, ::Divergence)
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

function _lnse_setup!(u, eq::CartesianPrimitiveLNSE{3, 3}, ::Rotational)
    _lnse_setup!(u, eq, Advective())
    W  = eq.scache[4]; t = eq.scache[2][1]
    Om = eq.pcache[5]
    vorticity!(W, u, t)
    eq.plans(Om, W)
    return nothing
end

function _linearised_advection!(out, v, eq::CartesianPrimitiveLNSE{3, 3}, ::Rotational)
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

_adjcont_advection!(out, v, eq::CartesianPrimitiveLNSE{3, 3}, ::Rotational) =
    _adjcont_advection!(out, v, eq, Divergence())
