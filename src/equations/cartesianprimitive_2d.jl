# Concrete Cartesian primitive NSE/LNSE operators for 2-component (2D) flows.
#
# These operators govern flows with velocity (u_x, u_y) in the x-y plane
# with no spanwise z-direction.  All cache naming follows the 3D convention
# (cartesianprimitive_3d.jl); z-derivative terms are simply absent.


# -------------------------------- #
# Abstract 2D Cartesian grid type  #
# -------------------------------- #
"""
    AbstractCartesianGrid2D{T, AXES, ORDER} <: AbstractCartesianGrid{T, 3, AXES, ORDER}

Abstract supertype for 2-component Cartesian grids stored in 3D arrays
`(x, y, t)` (no spanwise z direction).

`AXES = (x_dim, y_dim, nothing, t_dim)` gives the array dimension for each
coordinate; `AXES[3] = nothing` marks the absent z direction.  Subtypes
inherit the coordinate-indexed derivative wrappers `ddx_1!` (x), `ddx_2!` (y),
`ddx_4!` (t).

# Example

A 2D channel grid stored as `(y, x, t)` (wall-normal first):

```julia
struct Channel2DGrid{T} <: AbstractCartesianGrid2D{T, (2, 1, nothing, 3), (2, 3)} end
```
"""
abstract type AbstractCartesianGrid2D{T, AXES, ORDER} <: AbstractCartesianGrid{T, 3, AXES, ORDER} end

"""
    x_dim(grid::AbstractCartesianGrid2D) -> Int

Return the array dimension corresponding to the streamwise coordinate `x`.
"""
x_dim(::AbstractCartesianGrid2D{<:Any, AXES}) where {AXES} = AXES[1]

"""
    y_dim(grid::AbstractCartesianGrid2D) -> Int

Return the array dimension corresponding to the wall-normal coordinate `y`.
"""
y_dim(::AbstractCartesianGrid2D{<:Any, AXES}) where {AXES} = AXES[2]

"""
    t_dim(grid::AbstractCartesianGrid2D) -> Int

Return the array dimension corresponding to time `t`.
"""
t_dim(::AbstractCartesianGrid2D{<:Any, AXES}) where {AXES} = AXES[4]


# ----------------------- #
# concrete 2D NSE struct  #
# ----------------------- #
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

    ddx_1!(dudx, u)
    ddx_2!(dudy, u)

    eq.plans(U, u); eq.plans(dUdx, dudx); eq.plans(dUdy, dudy)
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

    ddx_1!(dudx, u); ddx_2!(dudy, u)
    eq.plans(U, u); eq.plans(dUdy, dudy)

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

    ddx_1!(dvdx, v); ddx_2!(dvdy, v)

    eq.plans(V, v); eq.plans(dUdx, dudx)
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

    ddx_1!(dvdx, v); ddx_2!(dvdy, v)

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

    laplacian!(out, v, adjoint=true)
    out .*= 1/eq.Re

    eq.plans(V, v)
    for n in 1:2
        @. U1V[n] = U[1]*V[n]
        @. U2V[n] = U[2]*V[n]
    end
    eq.plans(u1v, U1V); eq.plans(u2v, U2V)

    eq.plans(dUdx, dudx)
    for n in 1:2
        out[n] .-= ddx_1!(dudx[1], u1v[n], adjoint=true) .+
                   ddx_2!(dudx[2], u2v[n], adjoint=true)
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
