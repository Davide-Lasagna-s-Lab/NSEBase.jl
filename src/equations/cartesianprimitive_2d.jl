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
@doc raw"""
    CartesianPrimitive2DNSE{T, FFT, S, P, BF}
    CartesianPrimitive2DNSE(grid, Re; force=NoForce(), flags=FFTW.EXHAUSTIVE)

Nonlinear Navier-Stokes operator for two-component planar Cartesian flows.

For state ``Q=(U,V)``, ``\nabla=(\partial_x,\partial_y)``, and
``\nu=Re^{-1}``, `eq(t,Q,out)` evaluates the pressure-free spatial residual

```math
\begin{aligned}
\mathcal N_1(Q) &= \nu\nabla_h^2U-U\partial_xU-V\partial_yU
                  +\mathcal F_{N,1}(Q),\\
\mathcal N_2(Q) &= \nu\nabla_h^2V-U\partial_xV-V\partial_yV
                  +\mathcal F_{N,2}(Q).
\end{aligned}
```

Derivatives are evaluated spectrally or by the matrices supplied by the grid;
products are formed in physical space, with transform padding when dealiasing
is enabled. The force term denotes the actual call
`force(out,Q,Forward())` and may act on either component.

The grid must map [`ddx!`](@ref) and [`ddy!`](@ref) to physical ``x`` and
``y``, and physical ``z`` must be absent. Otherwise [`laplacian!`](@ref) would
include a spatial direction that is missing from advection. The low-level
constructor trusts that grid contract; the bundled two-dimensional channel
and cavity grids provide it.

The argument `t` is accepted for time-stepper compatibility and is not used.
A transformed logical `t` direction participates in field storage and inner
products but not in this spatial residual or its Laplacian. Pressure is not
computed here; a divergence-free [`ProjectedNSE`](@ref) basis removes its
gradient after projection.
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
@doc raw"""
    CartesianPrimitive2DLNSE{MODE, T, FFT, S, P, BF}
    CartesianPrimitive2DLNSE(grid, Re; mode=AdjointDiscrete(),
                             force=NoForce(), flags=FFTW.EXHAUSTIVE)

Forward or adjoint linearisation of [`CartesianPrimitive2DNSE`](@ref) about a
cached state ``Q=(U,V)``. Let ``q=(u,v)`` be a forward perturbation,
``p=(a,b)`` an adjoint field, and ``\nu=Re^{-1}``.

For `MODE == Forward`, the two equations are

```math
\begin{aligned}
(L_Qq)_1 &= \nu\nabla_h^2u-U\partial_xu-V\partial_yu
            -u\partial_xU-v\partial_yU+\mathcal F_{F,1}(q),\\
(L_Qq)_2 &= \nu\nabla_h^2v-U\partial_xv-V\partial_yv
            -u\partial_xV-v\partial_yV+\mathcal F_{F,2}(q).
\end{aligned}
```

For `MODE == AdjointContinuous`, integration by parts before discretisation
gives

```math
\begin{aligned}
(L_Q^{\dagger,c}p)_1 &= \nu\nabla^2a+U\partial_xa+V\partial_ya
                        -a\partial_xU-b\partial_xV+\mathcal F_{C,1}(p),\\
(L_Q^{\dagger,c}p)_2 &= \nu\nabla^2b+U\partial_xb+V\partial_yb
                        -a\partial_yU-b\partial_yV+\mathcal F_{C,2}(p).
\end{aligned}
```

This continuous formula assumes ``\partial_xU+\partial_yV=0`` and boundary
conditions that eliminate the integration-by-parts terms. For
`MODE == AdjointDiscrete`, let ``D_x,D_y`` be the derivative matrices actually
used by the forward code, let ``M_f`` denote the configured numerical
multiplication by a real field ``f`` (including padding and truncation when
dealiasing is enabled), and let ``\dagger`` denote the weighted
adjoint under [`dot`](@ref). The implementation evaluates

```math
\begin{aligned}
(L_{h,Q}^{\dagger}p)_1
 &= \nu(\nabla_h^2)^\dagger a-D_x^\dagger M_Ua-D_y^\dagger M_Va
    -M_{D_xU}a-M_{D_xV}b+\mathcal F_{D,1}(p),\\
(L_{h,Q}^{\dagger}p)_2
 &= \nu(\nabla_h^2)^\dagger b-D_x^\dagger M_Ub-D_y^\dagger M_Vb
    -M_{D_yU}a-M_{D_yV}b+\mathcal F_{D,2}(p).
\end{aligned}
```

In particular, the transpose of ``-M_UD_x`` is ``-D_x^\dagger M_U``: the
physical multiplication is performed before the adjoint derivative. This is
the exact transpose of the discretised hydrodynamic operator, including the
FFT padding/truncation pipeline. See [`AdjointDiscrete`](@ref) for the full
inner-product derivation. A force is included in that identity only when its
`AdjointDiscrete()` action is the weighted transpose of its `Forward()` action.

The four-argument call `eq(t,Q,q,out)` prepares the base-state transforms and
gradients, then applies the selected operator. Subsequent three-argument calls
`eq(t,q,out)` reuse that base cache. The object owns mutable workspaces and is
not reentrant or safe for concurrent calls.
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
# Four arguments: prepare the base-flow cache, then delegate to the cached three-argument action.
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
