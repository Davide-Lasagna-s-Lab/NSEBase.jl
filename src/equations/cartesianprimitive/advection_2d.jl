# Advection helpers for 2D, two-component Cartesian primitive velocity.
#
# Organised by role: nonlinear, then linearised (forward), then continuous
# adjoint, then discrete adjoint; within each role, by advection form. The 2D
# vorticity is a scalar (ω_z = ∂x v − ∂y u), so the rotational kernels are inline.


# ================================================================== #
# NONLINEAR                                                          #
# ================================================================== #
# advective: N_i = u_j ∂_j u_i
function advection!(out, u::VectorField{2, F}, eq::CartesianPrimitiveNSE{2, 2}, ::Advective) where {F<:FTField}
    dudx = eq.scache[1]; dudy = eq.scache[2]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]

    ddx!(dudx, u); ddy!(dudy, u)
    eq.plans(U, u); eq.plans(dUdx, dudx); eq.plans(dUdy, dudy)
    for n in 1:2
        @. dUdx[n] = -U[1]*dUdx[n] - U[2]*dUdy[n]
    end
    eq.plans(out, dUdx, add=true)
    return out
end

# divergence: N_i = ∂_j(u_i u_j)
function advection!(out, u::VectorField{2, F}, eq::CartesianPrimitiveNSE{2, 2}, ::Divergence) where {F<:FTField}
    sA = eq.scache[1]; s12 = eq.scache[2][1]; t = eq.scache[2][2]
    U  = eq.pcache[1]; P = eq.pcache[2][1]

    eq.plans(U, u)
    @. P = U[1]*U[1]; eq.plans(sA[1], P)
    @. P = U[2]*U[2]; eq.plans(sA[2], P)
    @. P = U[1]*U[2]; eq.plans(s12, P)

    ddx!(t, sA[1]); out[1] .-= t
    ddy!(t, s12);   out[1] .-= t
    ddx!(t, s12);   out[2] .-= t
    ddy!(t, sA[2]); out[2] .-= t
    return out
end

# rotational: N = ω×u (the ∇(½|u|²) gradient is dropped — removed by projection)
function advection!(out, u::VectorField{2, F}, eq::CartesianPrimitiveNSE{2, 2}, ::Rotational) where {F<:FTField}
    W = eq.scache[1][1]; t = eq.scache[1][2]
    U = eq.pcache[1]; Om = eq.pcache[2][1]; C = eq.pcache[3]

    ddx!(W, u[2]); ddy!(t, u[1]); W .-= t
    eq.plans(U, u); eq.plans(Om, W)
    @. C[1] =  Om*U[2]
    @. C[2] = -Om*U[1]
    eq.plans(out, C, add=true)
    return out
end


# ================================================================== #
# LINEARISED (forward)                                              #
# ================================================================== #
# --- base-flow setup (shared by every linearised mode) ---
function lnse_setup!(u::VectorField{2, F}, eq::CartesianPrimitiveNSE{2, 2}, ::Advective) where {F<:FTField}
    dudx = eq.scache[1]; dudy = eq.scache[2]
    U    = eq.pcache[1]; dUdy = eq.pcache[3]
    ddx!(dudx, u); ddy!(dudy, u)
    eq.plans(U, u); eq.plans(dUdy, dudy)
    return nothing
end

lnse_setup!(u, eq::CartesianPrimitiveNSE{2, 2}, ::Divergence) = lnse_setup!(u, eq, Advective())

function lnse_setup!(u::VectorField{2, F}, eq::CartesianPrimitiveNSE{2, 2}, ::Rotational) where {F<:FTField}
    lnse_setup!(u, eq, Advective())
    W = eq.scache[3][1]
    ddx!(W, u[2]); W .-= eq.scache[2][1]
    eq.plans(eq.pcache[2][1], W)
    return nothing
end

# --- forward advection ---
# advective: L(v)_i = (U·∇)v_i + (v·∇)U_i
function linearised_advection!(out, v::VectorField{2, F}, eq::CartesianPrimitiveNSE{2, 2}, ::Advective) where {F<:FTField}
    dudx = eq.scache[1]; dvdx = eq.scache[2]; dvdy = eq.scache[3]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]
    V    = eq.pcache[4]; dVdx = eq.pcache[5]; dVdy = eq.pcache[6]

    ddx!(dvdx, v); ddy!(dvdy, v)
    eq.plans(V, v); eq.plans(dUdx, dudx)
    eq.plans(dVdx, dvdx); eq.plans(dVdy, dvdy)
    for n in 1:2
        @. dVdx[n]  = -U[1]*dVdx[n] - U[2]*dVdy[n]
        @. dVdx[n] -=  V[1]*dUdx[n] + V[2]*dUdy[n]
    end
    eq.plans(out, dVdx, add=true)
    return out
end

# divergence: L(v)_i = ∂_j(U_i v_j + U_j v_i)
function linearised_advection!(out, v::VectorField{2, F}, eq::CartesianPrimitiveNSE{2, 2}, ::Divergence) where {F<:FTField}
    sA = eq.scache[1]; s12 = eq.scache[2][1]; t = eq.scache[3][1]
    U  = eq.pcache[1]; V = eq.pcache[4]; P = eq.pcache[5][1]

    eq.plans(V, v)
    @. P = 2*U[1]*V[1];           eq.plans(sA[1], P)
    @. P = 2*U[2]*V[2];           eq.plans(sA[2], P)
    @. P = U[1]*V[2] + U[2]*V[1]; eq.plans(s12, P)

    ddx!(t, sA[1]); out[1] .-= t
    ddy!(t, s12);   out[1] .-= t
    ddx!(t, s12);   out[2] .-= t
    ddy!(t, sA[2]); out[2] .-= t
    return out
end

# rotational: L(v) = (∇×v)×U + Ω×v
function linearised_advection!(out, v::VectorField{2, F}, eq::CartesianPrimitiveNSE{2, 2}, ::Rotational) where {F<:FTField}
    Wv = eq.scache[1][1]; t = eq.scache[2][1]
    U  = eq.pcache[1]; Om = eq.pcache[2][1]
    V  = eq.pcache[4]; Omv = eq.pcache[5][1]; C = eq.pcache[6]

    ddx!(Wv, v[2]); ddy!(t, v[1]); Wv .-= t
    eq.plans(V, v); eq.plans(Omv, Wv)
    @. C[1] =  Omv*U[2] + Om*V[2]
    @. C[2] = -Omv*U[1] - Om*V[1]
    eq.plans(out, C, add=true)
    return out
end


# ================================================================== #
# CONTINUOUS ADJOINT                                                 #
# ================================================================== #
# advective: +(U·∇)w − Σ_i w_i ∇U_i
function adjoint_continuous_advection!(out, v::VectorField{2, F}, eq::CartesianPrimitiveNSE{2, 2}, ::Advective) where {F<:FTField}
    dudx = eq.scache[1]; dvdx = eq.scache[2]; dvdy = eq.scache[3]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]
    V    = eq.pcache[4]; dVdx = eq.pcache[5]; dVdy = eq.pcache[6]

    ddx!(dvdx, v); ddy!(dvdy, v)
    eq.plans(V, v); eq.plans(dUdx, dudx)
    eq.plans(dVdx, dvdx); eq.plans(dVdy, dvdy)
    for n in 1:2
        @. dVdx[n] = U[1]*dVdx[n] + U[2]*dVdy[n]
    end
    dVdy .= 0
    for i in 1:2
        @. dVdy[1] -= V[i]*dUdx[i]
        @. dVdy[2] -= V[i]*dUdy[i]
    end
    eq.plans(out, dVdx, add=true); eq.plans(out, dVdy, add=true)
    return out
end

# divergence: +(U·∇)w in conservative form, plus the cross-gradient.
function adjoint_continuous_advection!(out, v::VectorField{2, F}, eq::CartesianPrimitiveNSE{2, 2}, ::Divergence) where {F<:FTField}
    U = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]
    V = eq.pcache[4]; P = eq.pcache[5][1]; G = eq.pcache[6]

    eq.plans(dUdx, eq.scache[1])
    eq.plans(V, v)

    t  = eq.scache[1][1]
    sP = eq.scache[2][1]
    for i in 1:2
        @. P = U[1]*V[i]; eq.plans(sP, P); ddx!(t, sP); out[i] .+= t
        @. P = U[2]*V[i]; eq.plans(sP, P); ddy!(t, sP); out[i] .+= t
    end

    G .= 0
    for i in 1:2
        @. G[1] -= V[i]*dUdx[i]
        @. G[2] -= V[i]*dUdy[i]
    end
    eq.plans(out, G, add=true)
    return out
end

# rotational has no distinct continuous-adjoint form; reuse the divergence one.
adjoint_continuous_advection!(out, v::VectorField{2, F}, eq::CartesianPrimitiveNSE{2, 2}, ::Rotational) where {F<:FTField} =
    adjoint_continuous_advection!(out, v, eq, Divergence())


# ================================================================== #
# DISCRETE ADJOINT (advective only)                                 #
# ================================================================== #
function adjoint_discrete_advection!(out, v::VectorField{2, F}, eq::CartesianPrimitiveNSE{2, 2}, ::Advective) where {F<:FTField}
    dudx = eq.scache[1]; u1v = eq.scache[2]; u2v = eq.scache[3]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]
    V    = eq.pcache[4]; U1V = eq.pcache[5]; U2V = eq.pcache[6]

    eq.plans(V, v)
    for n in 1:2
        @. U1V[n] = U[1]*V[n]
        @. U2V[n] = U[2]*V[n]
    end
    eq.plans(u1v, U1V); eq.plans(u2v, U2V)

    eq.plans(dUdx, dudx)
    for n in 1:2
        out[n] .-= ddx!(dudx[1], u1v[n], adjoint=true) .+
                   ddy!(dudx[2], u2v[n], adjoint=true)
    end
    U1V .= 0
    for n in 1:2
        @. U1V[1] -= V[n]*dUdx[n]
        @. U1V[2] -= V[n]*dUdy[n]
    end
    eq.plans(out, U1V, add=true)
    return out
end
