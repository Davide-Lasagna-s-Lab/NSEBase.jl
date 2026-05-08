# Concrete Cartesian primitive NSE/LNSE operators.
#
# Body forces (Coriolis, buoyancy, …) are passed as a callable at construction
# time.  The callable signature is: body_force(out, u, mode::Mode)
# where `mode` identifies the equation type (Forward, AdjointDiscrete, …).
# Use NoForce() for flows with no body force.
#
# The structs carry no grid type parameter.  Physical-direction derivatives
# (ddx_x!, ddx_y!, ddx_z!) look up the correct array dimension from the
# grid type already embedded in each FTField argument at call time.

# ----------- #
# mode types  #
# ----------- #
abstract type               Mode end
struct Forward           <: Mode end
struct AdjointDiscrete   <: Mode end
struct AdjointContinuous <: Mode end

# ---------------------------------- #
# default body force (no-op)         #
# ---------------------------------- #
struct NoForce end
(::NoForce)(_, _, _) = nothing

# ------------------------------------------------ #
# concrete NSE struct                              #
# ------------------------------------------------ #
mutable struct CartesianPrimitiveNSE{T, FFT, S, P, BF}
              Re::T
     const plans::FFT
    const scache::Vector{VectorField{3, S}}
    const pcache::Vector{VectorField{3, P}}
    const  force::BF
end

function CartesianPrimitiveNSE(g::G, Re;
                                force::BF=NoForce(),
                                flags=FFTW.EXHAUSTIVE) where {T, G<:AbstractGrid{T}, BF}
    plans  = FFTPlans(g, flags=flags)
    scache = [VectorField([FTField(g)               for _ in 1:3]...) for _ in 1:3]
    pcache = [VectorField([  Field(g, dealias=true) for _ in 1:3]...) for _ in 1:4]
    return CartesianPrimitiveNSE(T(Re), plans, scache, pcache, force)
end

# ------------------------------------------------ #
# concrete LNSE struct                             #
# ------------------------------------------------ #
mutable struct CartesianPrimitiveLNSE{MODE<:Mode, T, FFT, S, P, BF}
              Re::T
     const plans::FFT
    const scache::Vector{VectorField{3, S}}
    const pcache::Vector{VectorField{3, P}}
    const  force::BF
end

function CartesianPrimitiveLNSE(g::G, Re;
                                 mode::Mode=AdjointDiscrete(),
                                 force::BF=NoForce(),
                                 flags=FFTW.EXHAUSTIVE) where {T, G<:AbstractGrid{T}, BF}
    plans  = FFTPlans(g, flags=flags)
    scache = [VectorField([FTField(g)               for _ in 1:3]...) for _ in 1:4]
    pcache = [VectorField([  Field(g, dealias=true) for _ in 1:3]...) for _ in 1:8]
    return CartesianPrimitiveLNSE{typeof(mode)}(T(Re), plans, scache, pcache, force)
end


# -------------------- #
# nonlinear NSE        #
# -------------------- #
function (eq::CartesianPrimitiveNSE)(::Real,
                                      u::VectorField{3, F},
                                    out::VectorField{3, F}) where {F<:FTField}
    dudx = eq.scache[1]; dudy = eq.scache[2]; dudz = eq.scache[3]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]

    laplacian!(out, u)
    out .*= 1/eq.Re

    ddx_x!(dudx, u); ddx_y!(dudy, u); ddx_z!(dudz, u)

    eq.plans(U, u); eq.plans(dUdx, dudx); eq.plans(dUdy, dudy); eq.plans(dUdz, dudz)
    for n in 1:3
        @. dUdx[n] = -U[1]*dUdx[n] - U[2]*dUdy[n] - U[3]*dUdz[n]
    end
    eq.plans(out, dUdx, true)

    eq.force(out, u, Forward())
    return out
end


# ------------------- #
# linearised NSE      #
# ------------------- #

# 3-arg: set up base-flow cache then delegate to 2-arg
function (eq::CartesianPrimitiveLNSE)(::Real,
                                       u::VectorField{3, F},
                                       v::VectorField{3, F},
                                     out::VectorField{3, F}) where {F<:FTField}
    dudx = eq.scache[1]; dudy = eq.scache[2]; dudz = eq.scache[3]
    U    = eq.pcache[1]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]

    ddx_x!(dudx, u); ddx_y!(dudy, u); ddx_z!(dudz, u)
    eq.plans(U, u); eq.plans(dUdy, dudy); eq.plans(dUdz, dudz)

    eq(0, v, out)
    return out
end

# forward LNSE
function (eq::CartesianPrimitiveLNSE{Forward})(::Real,
                                                v::VectorField{3, F},
                                              out::VectorField{3, F}) where {F<:FTField}
    dudx = eq.scache[1]; dvdx = eq.scache[2]; dvdy = eq.scache[3]; dvdz = eq.scache[4]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]
    V    = eq.pcache[5]; dVdx = eq.pcache[6]; dVdy = eq.pcache[7]; dVdz = eq.pcache[8]

    laplacian!(out, v)
    out .*= 1/eq.Re

    ddx_x!(dvdx, v); ddx_y!(dvdy, v); ddx_z!(dvdz, v)

    eq.plans(V, v); eq.plans(dUdx, dudx)
    eq.plans(dVdx, dvdx); eq.plans(dVdy, dvdy); eq.plans(dVdz, dvdz)
    for n in 1:3
        @. dVdx[n]  = -U[1]*dVdx[n] - U[2]*dVdy[n] - U[3]*dVdz[n]
        @. dVdx[n] -=  V[1]*dUdx[n] + V[2]*dUdy[n] + V[3]*dUdz[n]
    end
    eq.plans(out, dVdx, true)

    eq.force(out, v, Forward())
    return out
end

# continuous adjoint LNSE
function (eq::CartesianPrimitiveLNSE{AdjointContinuous})(::Real,
                                                          v::VectorField{3, F},
                                                        out::VectorField{3, F}) where {F<:FTField}
    dudx = eq.scache[1]; dvdx = eq.scache[2]; dvdy = eq.scache[3]; dvdz = eq.scache[4]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]
    V    = eq.pcache[5]; dVdx = eq.pcache[6]; dVdy = eq.pcache[7]; dVdz = eq.pcache[8]

    laplacian!(out, v)
    out .*= 1/eq.Re

    ddx_x!(dvdx, v); ddx_y!(dvdy, v); ddx_z!(dvdz, v)

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
    eq.plans(out, dVdx, true); eq.plans(out, dVdz, true)

    eq.force(out, v, AdjointContinuous())
    return out
end

# discrete adjoint LNSE
function (eq::CartesianPrimitiveLNSE{AdjointDiscrete})(::Real,
                                                        v::VectorField{3, F},
                                                      out::VectorField{3, F}) where {F<:FTField}
    dudx = eq.scache[1]; u1v  = eq.scache[2]; u2v  = eq.scache[3]; u3v  = eq.scache[4]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]
    V    = eq.pcache[5]; U1V  = eq.pcache[6]; U2V  = eq.pcache[7]; U3V  = eq.pcache[8]

    laplacian!(out, v, adjoint=true)
    out .*= 1/eq.Re

    eq.plans(V, v)
    for n in 1:3
        @. U1V[n] = U[1]*V[n]
        @. U2V[n] = U[2]*V[n]
        @. U3V[n] = U[3]*V[n]
    end
    eq.plans(u1v, U1V); eq.plans(u2v, U2V); eq.plans(u3v, U3V)

    eq.plans(dUdx, dudx)
    for n in 1:3
        out[n] .-= ddx_x!(dudx[1], u1v[n], adjoint=true) .+
                   ddx_y!(dudx[2], u2v[n], adjoint=true) .+
                   ddx_z!(dudx[3], u3v[n], adjoint=true)
    end
    U1V .= 0
    for n in 1:3
        @. U1V[1] -= V[n]*dUdx[n]
        @. U1V[2] -= V[n]*dUdy[n]
        @. U1V[3] -= V[n]*dUdz[n]
    end
    eq.plans(out, U1V, true)

    eq.force(out, v, AdjointDiscrete())
    return out
end
