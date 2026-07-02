# Concrete Navier-Stokes operators for two-component (u, v) planar Cartesian flows.
#
# Mirror of cartesianprimitive_3d.jl for 2D flows: velocity has no spanwise w
# component and no z-derivative terms appear.  All cache naming and operator
# conventions follow the 3D file; refer to that file for the full design notes,
# including the halo-exchange pattern used by every call method below.
#
# Variants provided:
#   CartesianPrimitive2DNSE  — nonlinear NSE: out = Δu/Re − (u·∇)u + force
#   CartesianPrimitive2DLNSE{Forward}          — forward linearised operator
#   CartesianPrimitive2DLNSE{AdjointContinuous}— continuous adjoint
#   CartesianPrimitive2DLNSE{AdjointDiscrete}  — discrete adjoint


# ----------------------- #
# concrete 2D NSE struct  #
# ----------------------- #
"""
    CartesianPrimitive2DNSE{T, FFT, S, P, BF}

Nonlinear Navier-Stokes operator for two-component planar Cartesian flows.

Evaluates `out = Δu/Re − (u·∇)u + force(out, u, Forward())` in spectral space.
Identical structure to [`CartesianPrimitive3DNSE`](@ref) with the spanwise
velocity component and all z-derivatives removed.
"""
mutable struct CartesianPrimitive2DNSE{T, FFT, S, P, BF}
              Re::T
     const plans::FFT
    const scache::Vector{VectorField{2, S}}
    const pcache::Vector{VectorField{2, P}}
    const  force::BF
end

function CartesianPrimitive2DNSE(g::G, Re;
                             force::BF=NoForce(),
                             flags    =FFTW.EXHAUSTIVE) where {T, G<:AbstractGrid{T}, BF}
    plans  = FFTPlans(g, flags=flags)
    scache = [VectorField([FTField(g)               for _ in 1:2]...) for _ in 1:2]
    pcache = [VectorField([  Field(g, dealias=true) for _ in 1:2]...) for _ in 1:3]
    return CartesianPrimitive2DNSE(T(Re), plans, scache, pcache, force)
end

# ----------------------- #
# concrete 2D LNSE struct #
# ----------------------- #
"""
    CartesianPrimitive2DLNSE{MODE, T, FFT, S, P, BF}

Linearised Navier-Stokes operator for two-component planar Cartesian flows,
parameterised on `MODE <: Mode`.

Identical structure to [`CartesianPrimitive3DLNSE`](@ref) with z-derivative
terms absent.  The three-argument `(t, u, v, out)` form caches base-flow
gradients from `u` then delegates to the two-argument `(t, v, out)` form.
"""
mutable struct CartesianPrimitive2DLNSE{MODE, T, FFT, S, P, BF}
              Re::T
     const plans::FFT
    const scache::Vector{VectorField{2, S}}
    const pcache::Vector{VectorField{2, P}}
    const  force::BF

    CartesianPrimitive2DLNSE{MODE}(Re::T,
                                plans::FFT,
                               scache::Vector{VectorField{2, S}},
                               pcache::Vector{VectorField{2, P}},
                                force::BF) where {MODE, T, FFT, S, P, BF} =
        new{MODE, T, FFT, S, P, BF}(Re, plans, scache, pcache, force)
end

function CartesianPrimitive2DLNSE(g::G, Re;
                               mode::Mode=AdjointDiscrete(),
                              force::BF  =NoForce(),
                              flags      =FFTW.EXHAUSTIVE) where {T, G<:AbstractGrid{T}, BF}
    plans  = FFTPlans(g, flags=flags)
    scache = [VectorField([FTField(g)               for _ in 1:2]...) for _ in 1:3]
    pcache = [VectorField([  Field(g, dealias=true) for _ in 1:2]...) for _ in 1:6]
    return CartesianPrimitive2DLNSE{typeof(mode)}(T(Re), plans, scache, pcache, force)
end


# ------------- #
# nonlinear NSE #
# ------------- #
function (eq::CartesianPrimitive2DNSE)(::Real,
                                      u::VectorField{2, F},
                                    out::VectorField{2, F}) where {F<:FTField}
    dudx = eq.scache[1]; dudy = eq.scache[2]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]

    laplacian!(out, u)
    out .*= 1/eq.Re

    ddx!(dudx, u)
    ddy!(dudy, u)

    eq.plans(U, u)
    eq.plans(dUdx, dudx); eq.plans(dUdy, dudy)
    for n in 1:2
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
function (eq::CartesianPrimitive2DLNSE)(::Real,
                                       u::VectorField{2, F},
                                       v::VectorField{2, F},
                                     out::VectorField{2, F}) where {F<:FTField}
    dudx = eq.scache[1]; dudy = eq.scache[2]
    U    = eq.pcache[1]; dUdy = eq.pcache[3]

    ddx!(dudx, u)
    ddy!(dudy, u)

    eq.plans(U, u)
    eq.plans(dUdy, dudy)

    eq(0, v, out)
    return out
end

# forward LNSE
function (eq::CartesianPrimitive2DLNSE{Forward})(::Real,
                                                 v::VectorField{2, F},
                                               out::VectorField{2, F}) where {F<:FTField}
    dudx = eq.scache[1]; dvdx = eq.scache[2]; dvdy = eq.scache[3]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]
    V    = eq.pcache[4]; dVdx = eq.pcache[5]; dVdy = eq.pcache[6]

    laplacian!(out, v)
    out .*= 1/eq.Re

    ddx!(dvdx, v)
    ddy!(dvdy, v)

    eq.plans(V, v)
    eq.plans(dUdx, dudx)
    eq.plans(dVdx, dvdx); eq.plans(dVdy, dvdy)
    for n in 1:2
        @. dVdx[n]  = -U[1]*dVdx[n] - U[2]*dVdy[n]
        @. dVdx[n] -=  V[1]*dUdx[n] + V[2]*dUdy[n]
    end
    eq.plans(out, dVdx, add=true)

    eq.force(out, v, Forward())
    return out
end

# continuous adjoint LNSE
function (eq::CartesianPrimitive2DLNSE{AdjointContinuous})(::Real,
                                                           v::VectorField{2, F},
                                                         out::VectorField{2, F}) where {F<:FTField}
    dudx = eq.scache[1]; dvdx = eq.scache[2]; dvdy = eq.scache[3]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]
    V    = eq.pcache[4]; dVdx = eq.pcache[5]; dVdy = eq.pcache[6]

    laplacian!(out, v)
    out .*= 1/eq.Re

    ddx!(dvdx, v)
    ddy!(dvdy, v)

    eq.plans(V, v)
    eq.plans(dUdx, dudx)
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

    eq.force(out, v, AdjointContinuous())
    return out
end

# discrete adjoint LNSE
function (eq::CartesianPrimitive2DLNSE{AdjointDiscrete})(::Real,
                                                         v::VectorField{2, F},
                                                       out::VectorField{2, F}) where {F<:FTField}
    dudx = eq.scache[1]; u1v  = eq.scache[2]; u2v  = eq.scache[3]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]
    V    = eq.pcache[4]; U1V  = eq.pcache[5]; U2V  = eq.pcache[6]

    laplacian!(out, v; adjoint=true)
    out .*= 1/eq.Re

    eq.plans(V, v)
    for n in 1:2
        @. U1V[n] = U[1]*V[n]
        @. U2V[n] = U[2]*V[n]
    end
    eq.plans(u1v, U1V); eq.plans(u2v, U2V)

    eq.plans(dUdx, dudx)
    for n in 1:2
        ddx!(dudx[1], u1v[n]; adjoint=true)
        ddy!(dudx[2], u2v[n]; adjoint=true)
        out[n] .-= dudx[1] .+ dudx[2]
    end
    U1V .= 0
    for n in 1:2
        @. U1V[1] -= V[n]*dUdx[n]
        @. U1V[2] -= V[n]*dUdy[n]
    end
    eq.plans(out, U1V, add=true)

    eq.force(out, v, AdjointDiscrete())
    return out
end
