# Concrete Cartesian primitive NSE/LNSE operators for 3-component (3D) flows.
#
# Body forces (Coriolis, buoyancy, …) are passed as a callable at construction
# time.  The callable signature is: body_force(out, u, mode::Mode)
# where `mode` identifies the equation type (Forward, AdjointDiscrete, …).
# Use NoForce() for flows with no body force.


# -------------------------------- #
# Abstract 3D Cartesian grid type  #
# -------------------------------- #
"""
    AbstractCartesianGrid3D{T, AXES, ORDER} <: AbstractCartesianGrid{T, 4, AXES, ORDER}

Abstract supertype for 3-component Cartesian grids stored in 4D arrays
`(x, y, z, t)`.

`AXES = (x_dim, y_dim, z_dim, t_dim)` gives the array dimension for each
coordinate.  Subtypes inherit the accessors `x_dim`, `y_dim`, `z_dim`, `t_dim`
and the named derivative wrappers `ddx_x!`, `ddx_y!`, `ddx_z!`, `dds!`.

# Example

A channel-flow grid stored as `(y, x, z, t)` (wall-normal first):

```julia
struct ChannelGrid{T} <: AbstractCartesianGrid3D{T, (2, 1, 3, 4), (2, 3, 4)} end
```
"""
abstract type AbstractCartesianGrid3D{T, AXES, ORDER} <: AbstractCartesianGrid{T, 4, AXES, ORDER} end

"""
    x_dim(grid::AbstractCartesianGrid3D) -> Int

Return the array dimension corresponding to the streamwise coordinate `x`.
"""
x_dim(::AbstractCartesianGrid3D{<:Any, AXES}) where {AXES} = AXES[1]

"""
    y_dim(grid::AbstractCartesianGrid3D) -> Int

Return the array dimension corresponding to the wall-normal coordinate `y`.
"""
y_dim(::AbstractCartesianGrid3D{<:Any, AXES}) where {AXES} = AXES[2]

"""
    z_dim(grid::AbstractCartesianGrid3D) -> Int

Return the array dimension corresponding to the spanwise coordinate `z`.
"""
z_dim(::AbstractCartesianGrid3D{<:Any, AXES}) where {AXES} = AXES[3]

"""
    t_dim(grid::AbstractCartesianGrid3D) -> Int

Return the array dimension corresponding to time `t`.
"""
t_dim(::AbstractCartesianGrid3D{<:Any, AXES}) where {AXES} = AXES[4]


# ----------------------- #
# concrete 3D NSE struct  #
# ----------------------- #
mutable struct CartesianPrimitive3DNSE{T, FFT, S, P, BF}
              Re::T
     const plans::FFT
    const scache::Vector{VectorField{3, S}}
    const pcache::Vector{VectorField{3, P}}
    const  force::BF
end

function CartesianPrimitive3DNSE(g::G, Re;
                             force::BF=NoForce(),
                             flags    =FFTW.EXHAUSTIVE) where {T, G<:AbstractGrid{T}, BF}
    plans  = FFTPlans(g, flags=flags)
    scache = [VectorField([FTField(g)               for _ in 1:3]...) for _ in 1:3]
    pcache = [VectorField([  Field(g, dealias=true) for _ in 1:3]...) for _ in 1:4]
    return CartesianPrimitive3DNSE(T(Re), plans, scache, pcache, force)
end

# ----------------------- #
# concrete 3D LNSE struct #
# ----------------------- #
mutable struct CartesianPrimitive3DLNSE{MODE, T, FFT, S, P, BF}
              Re::T
     const plans::FFT
    const scache::Vector{VectorField{3, S}}
    const pcache::Vector{VectorField{3, P}}
    const  force::BF

    CartesianPrimitive3DLNSE{MODE}(Re::T,
                                plans::FFT,
                               scache::Vector{VectorField{3, S}},
                               pcache::Vector{VectorField{3, P}},
                                force::BF) where {MODE, T, FFT, S, P, BF} =
        new{MODE, T, FFT, S, P, BF}(Re, plans, scache, pcache, force)
end

function CartesianPrimitive3DLNSE(g::G, Re;
                               mode::Mode=AdjointDiscrete(),
                              force::BF  =NoForce(),
                              flags      =FFTW.EXHAUSTIVE) where {T, G<:AbstractGrid{T}, BF}
    plans  = FFTPlans(g, flags=flags)
    scache = [VectorField([FTField(g)               for _ in 1:3]...) for _ in 1:4]
    pcache = [VectorField([  Field(g, dealias=true) for _ in 1:3]...) for _ in 1:8]
    return CartesianPrimitive3DLNSE{typeof(mode)}(T(Re), plans, scache, pcache, force)
end


# ------------- #
# nonlinear NSE #
# ------------- #
function (eq::CartesianPrimitive3DNSE)(::Real,
                                      u::VectorField{3, F},
                                    out::VectorField{3, F}) where {F<:FTField}
    dudx = eq.scache[1]; dudy = eq.scache[2]; dudz = eq.scache[3]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]

    laplacian!(out, u)
    out .*= 1/eq.Re

    ddx_x!(dudx, u)
    ddx_y!(dudy, u)
    ddx_z!(dudz, u)

    eq.plans(U, u); eq.plans(dUdx, dudx); eq.plans(dUdy, dudy); eq.plans(dUdz, dudz)
    for n in 1:3
        @. dUdx[n] = -U[1]*dUdx[n] - U[2]*dUdy[n] - U[3]*dUdz[n]
    end
    eq.plans(out, dUdx, add=true)

    eq.force(out, u, Forward())
    return out
end


# -------------- #
# linearised NSE #
# -------------- #
# 3-arg: set up base-flow cache then delegate to 2-arg
function (eq::CartesianPrimitive3DLNSE)(::Real,
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
function (eq::CartesianPrimitive3DLNSE{Forward})(::Real,
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
    eq.plans(out, dVdx, add=true)

    eq.force(out, v, Forward())
    return out
end

# continuous adjoint LNSE
function (eq::CartesianPrimitive3DLNSE{AdjointContinuous})(::Real,
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
    eq.plans(out, dVdx, add=true); eq.plans(out, dVdz, add=true)

    eq.force(out, v, AdjointContinuous())
    return out
end

# discrete adjoint LNSE
function (eq::CartesianPrimitive3DLNSE{AdjointDiscrete})(::Real,
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
    eq.plans(out, U1V, add=true)

    eq.force(out, v, AdjointDiscrete())
    return out
end
