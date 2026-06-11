# Concrete Navier-Stokes operators for 2D-3C (u, v, w) Cartesian flows.
#
# A 2D-3C flow lives on a two-dimensional spatial grid (x, y) but carries
# three velocity components.  The in-plane pair (u, v) satisfies the standard
# incompressible 2D NSE; the out-of-plane component w is governed by:
#
#   ∂w/∂t = Δw/Re − (u·∇_{xy})w
#
# There is no pressure gradient in z, and w does not contribute to advection
# (the 2D incompressibility constraint involves only ∂u/∂x + ∂v/∂y = 0).
# The Laplacian is therefore two-dimensional for all three components.
#
# Variants provided:
#   CartesianPrimitive2D3CNSE  — nonlinear NSE: out = Δu/Re − (u_in·∇)u + force
#   CartesianPrimitive2D3CLNSE{Forward}          — forward linearised operator
#   CartesianPrimitive2D3CLNSE{AdjointContinuous}— continuous adjoint
#   CartesianPrimitive2D3CLNSE{AdjointDiscrete}  — discrete adjoint
#
# All cache layouts and operator conventions follow cartesianprimitive_2d.jl
# with the sole structural changes being VectorField{3} (not {2}) and loops
# running to 3.  In the linearised operators, only V[1] and V[2] contribute
# to the base-flow advection term (V[3] carries no advection in a 2D grid).


# ------------------------- #
# concrete 2D3C NSE struct  #
# ------------------------- #
"""
    CartesianPrimitive2D3CNSE{T, FFT, S, P, BF}

Nonlinear Navier-Stokes operator for 2D-3C Cartesian flows.

Evaluates `out = Δu/Re − (u_in·∇)u + force(out, u, Forward())` in spectral
space, where the advecting velocity `u_in = (u, v)` contains only the
in-plane components and the Laplacian is two-dimensional for all three
velocity components.
"""
mutable struct CartesianPrimitive2D3CNSE{T, FFT, S, P, BF}
              Re::T
     const plans::FFT
    const scache::Vector{VectorField{3, S}}
    const pcache::Vector{VectorField{3, P}}
    const  force::BF
end

function CartesianPrimitive2D3CNSE(g::G, Re;
                               force::BF=NoForce(),
                               flags    =FFTW.EXHAUSTIVE) where {T, G<:AbstractGrid{T}, BF}
    plans  = FFTPlans(g, flags=flags)
    scache = [VectorField([FTField(g)               for _ in 1:3]...) for _ in 1:2]
    pcache = [VectorField([  Field(g, dealias=true) for _ in 1:3]...) for _ in 1:3]
    return CartesianPrimitive2D3CNSE(T(Re), plans, scache, pcache, force)
end

# ------------------------- #
# concrete 2D3C LNSE struct #
# ------------------------- #
"""
    CartesianPrimitive2D3CLNSE{MODE, T, FFT, S, P, BF}

Linearised Navier-Stokes operator for 2D-3C Cartesian flows, parameterised
on `MODE <: Mode`.

Identical structure to [`CartesianPrimitive2DLNSE`](@ref) with three velocity
components instead of two.  In all linearised variants, only the in-plane
perturbation components V[1] and V[2] contribute to the base-flow advection
term; V[3] (the out-of-plane perturbation) is advected but does not advect.
"""
mutable struct CartesianPrimitive2D3CLNSE{MODE, T, FFT, S, P, BF}
              Re::T
     const plans::FFT
    const scache::Vector{VectorField{3, S}}
    const pcache::Vector{VectorField{3, P}}
    const  force::BF

    CartesianPrimitive2D3CLNSE{MODE}(Re::T,
                                  plans::FFT,
                                 scache::Vector{VectorField{3, S}},
                                 pcache::Vector{VectorField{3, P}},
                                  force::BF) where {MODE, T, FFT, S, P, BF} =
        new{MODE, T, FFT, S, P, BF}(Re, plans, scache, pcache, force)
end

function CartesianPrimitive2D3CLNSE(g::G, Re;
                                 mode::Mode=AdjointDiscrete(),
                                force::BF  =NoForce(),
                                flags      =FFTW.EXHAUSTIVE) where {T, G<:AbstractGrid{T}, BF}
    plans  = FFTPlans(g, flags=flags)
    scache = [VectorField([FTField(g)               for _ in 1:3]...) for _ in 1:3]
    pcache = [VectorField([  Field(g, dealias=true) for _ in 1:3]...) for _ in 1:6]
    return CartesianPrimitive2D3CLNSE{typeof(mode)}(T(Re), plans, scache, pcache, force)
end


# ------------- #
# nonlinear NSE #
# ------------- #
function (eq::CartesianPrimitive2D3CNSE)(::Real,
                                        u::VectorField{3, F},
                                      out::VectorField{3, F}) where {F<:FTField}
    dudx = eq.scache[1]; dudy = eq.scache[2]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]

    laplacian!(out, u)
    out .*= 1/eq.Re

    ddx!(dudx, u); ddy!(dudy, u)

    eq.plans(U, u); eq.plans(dUdx, dudx); eq.plans(dUdy, dudy)
    for n in 1:3
        @. dUdx[n] = -U[1]*dUdx[n] - U[2]*dUdy[n]
    end
    eq.plans(out, dUdx, add=true)

    eq.force(out, u, Forward())
    return out
end


# -------------- #
# linearised NSE #
# -------------- #
# 3-arg: set up base-flow cache then delegate to 2-arg
function (eq::CartesianPrimitive2D3CLNSE)(::Real,
                                         u::VectorField{3, F},
                                         v::VectorField{3, F},
                                       out::VectorField{3, F}) where {F<:FTField}
    dudx = eq.scache[1]; dudy = eq.scache[2]
    U    = eq.pcache[1]; dUdy = eq.pcache[3]

    ddx!(dudx, u); ddy!(dudy, u)
    eq.plans(U, u); eq.plans(dUdy, dudy)

    eq(0, v, out)
    return out
end

# forward LNSE
function (eq::CartesianPrimitive2D3CLNSE{Forward})(::Real,
                                                   v::VectorField{3, F},
                                                 out::VectorField{3, F}) where {F<:FTField}
    dudx = eq.scache[1]; dvdx = eq.scache[2]; dvdy = eq.scache[3]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]
    V    = eq.pcache[4]; dVdx = eq.pcache[5]; dVdy = eq.pcache[6]

    laplacian!(out, v)
    out .*= 1/eq.Re

    ddx!(dvdx, v); ddy!(dvdy, v)

    eq.plans(V, v); eq.plans(dUdx, dudx)
    eq.plans(dVdx, dvdx); eq.plans(dVdy, dvdy)
    for n in 1:3
        @. dVdx[n]  = -U[1]*dVdx[n] - U[2]*dVdy[n]
        @. dVdx[n] -=  V[1]*dUdx[n] + V[2]*dUdy[n]
    end
    eq.plans(out, dVdx, add=true)

    eq.force(out, v, Forward())
    return out
end

# continuous adjoint LNSE
function (eq::CartesianPrimitive2D3CLNSE{AdjointContinuous})(::Real,
                                                             v::VectorField{3, F},
                                                           out::VectorField{3, F}) where {F<:FTField}
    dudx = eq.scache[1]; dvdx = eq.scache[2]; dvdy = eq.scache[3]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]
    V    = eq.pcache[4]; dVdx = eq.pcache[5]; dVdy = eq.pcache[6]

    laplacian!(out, v)
    out .*= 1/eq.Re

    ddx!(dvdx, v); ddy!(dvdy, v)

    eq.plans(V, v); eq.plans(dUdx, dudx)
    eq.plans(dVdx, dvdx); eq.plans(dVdy, dvdy)
    for n in 1:3
        @. dVdx[n] = U[1]*dVdx[n] + U[2]*dVdy[n]
    end
    dVdy .= 0
    for i in 1:3
        @. dVdy[1] -= V[i]*dUdx[i]
        @. dVdy[2] -= V[i]*dUdy[i]
    end
    eq.plans(out, dVdx, add=true); eq.plans(out, dVdy, add=true)

    eq.force(out, v, AdjointContinuous())
    return out
end

# discrete adjoint LNSE
function (eq::CartesianPrimitive2D3CLNSE{AdjointDiscrete})(::Real,
                                                           v::VectorField{3, F},
                                                         out::VectorField{3, F}) where {F<:FTField}
    dudx = eq.scache[1]; u1v  = eq.scache[2]; u2v  = eq.scache[3]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]
    V    = eq.pcache[4]; U1V  = eq.pcache[5]; U2V  = eq.pcache[6]

    laplacian!(out, v, adjoint=true)
    out .*= 1/eq.Re

    eq.plans(V, v)
    for n in 1:3
        @. U1V[n] = U[1]*V[n]
        @. U2V[n] = U[2]*V[n]
    end
    eq.plans(u1v, U1V); eq.plans(u2v, U2V)

    eq.plans(dUdx, dudx)
    for n in 1:3
        out[n] .-= ddx!(dudx[1], u1v[n], adjoint=true) .+
                   ddy!(dudx[2], u2v[n], adjoint=true)
    end
    U1V .= 0
    for n in 1:3
        @. U1V[1] -= V[n]*dUdx[n]
        @. U1V[2] -= V[n]*dUdy[n]
    end
    eq.plans(out, U1V, add=true)

    eq.force(out, v, AdjointDiscrete())
    return out
end
