# Concrete Navier-Stokes operators for three-component (u, v, w) Cartesian flows.
#
# Each operator is a callable struct whose `(::Real, u, out)` method evaluates
# one of the following PDEs in spectral space, writing the result into `out`:
#
#   CartesianPrimitive3DNSE  — nonlinear NSE: out = Δu/Re − (u·∇)u + force
#   CartesianPrimitive3DLNSE{Forward}          — forward linearised operator
#   CartesianPrimitive3DLNSE{AdjointContinuous}— continuous adjoint
#   CartesianPrimitive3DLNSE{AdjointDiscrete}  — discrete adjoint
#
# All variants share the same cache layout: `scache` holds spectral-space
# scratch VectorFields (for derivative intermediates) and `pcache` holds
# physical-space scratch VectorFields (for de-aliased nonlinear products).
#
# Body forces are injected via a callable `force(out, u, mode::Mode)`.
# Pass `NoForce()` when no body force is needed.
#
# Construct via [`construct_equations`](@ref) rather than directly to ensure
# cache and plan sizes are consistent.
#
# NOTE: on a single node an overlap measures no faster than a blocking swap
# (benchmarks/mpi_overlap.jl); it is not included in this implementation. See
# https://github.com/Davide-Lasagna-s-Lab/NSEBase.jl/issues/21 for a discussion
# on its efficacy.


# ----------------------- #
# concrete 3D NSE struct  #
# ----------------------- #
@doc raw"""
    CartesianPrimitive3DNSE{T, FFT, S, P, BF}
    CartesianPrimitive3DNSE(grid, Re; force=NoForce(), flags=FFTW.EXHAUSTIVE)

Nonlinear Navier-Stokes operator for three-component Cartesian flows.

Let ``Q=(U_1,U_2,U_3)=(U,V,W)``,
``(D_1,D_2,D_3)=(\partial_x,\partial_y,\partial_z)``, and ``\nu=Re^{-1}``.
The call `eq(t,Q,out)` evaluates

```math
\mathcal N_n(Q)=\nu\nabla_h^2U_n-
\sum_{j=1}^3U_jD_jU_n+\mathcal F_{N,n}(Q),\qquad n=1,2,3.
```

The derivatives are the spectral or finite-difference operators supplied by
the grid. Each nonlinear product is evaluated in physical space, using padded
transforms when dealiasing is enabled. ``\mathcal F_N`` denotes the actual
`force(out,Q,Forward())` action and may couple components.

The grid is expected to map [`ddx!`](@ref), [`ddy!`](@ref), and [`ddz!`](@ref)
to physical ``x``, ``y``, and ``z`` respectively; the low-level constructor
trusts that coordinate contract.

This is a pressure-free spatial residual. Pressure is eliminated only after
wrapping the operator in a [`ProjectedNSE`](@ref) whose basis represents the
intended divergence-free, boundary-compatible space. The argument `t` is
accepted for time-stepper compatibility and ignored; a transformed logical
`t` direction is likewise excluded from all spatial derivatives and from the
Laplacian.

# Fields
- `Re`: Reynolds number
- `plans`: `FFTPlans` for physical↔spectral transforms
- `scache`, `pcache`: spectral- and physical-space scratch `VectorField`s. The
  direct constructor allocates three and four respectively; the generic
  [`construct_equations`](@ref) shares the larger linearised-operator pools.
- `force`: body-force callable with signature `(out, u, mode)`
"""
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
@doc raw"""
    CartesianPrimitive3DLNSE{MODE, T, FFT, S, P, BF}
    CartesianPrimitive3DLNSE(grid, Re; mode=AdjointDiscrete(),
                             force=NoForce(), flags=FFTW.EXHAUSTIVE)

Forward or adjoint linearisation of [`CartesianPrimitive3DNSE`](@ref) about the
cached state ``Q=(U_1,U_2,U_3)``. Write a forward perturbation as
``q=(q_1,q_2,q_3)``, an adjoint field as ``p=(p_1,p_2,p_3)``, set
``(D_1,D_2,D_3)=(\partial_x,\partial_y,\partial_z)``, and let ``\nu=Re^{-1}``.

For `MODE == Forward`, the implemented LNSE action is

```math
(L_Qq)_n=\nu\nabla_h^2q_n-
\sum_{j=1}^3\left(U_jD_jq_n+q_jD_jU_n\right)+\mathcal F_{F,n}(q),
\qquad n=1,2,3.
```

Its hydrodynamic terms are the Jacobian of the nonlinear residual. The force
term is part of that Jacobian only when the selected policy supplies a linear
forward action.

For `MODE == AdjointContinuous`, integration by parts at the PDE level gives

```math
(L_Q^{\dagger,c}p)_j=\nu\nabla^2p_j+
\sum_{\ell=1}^3U_\ell\partial_\ell p_j-
\sum_{n=1}^3p_n\partial_jU_n+\mathcal F_{C,j}(p),
\qquad j=1,2,3.
```

This expression assumes a divergence-free base velocity and boundary
conditions for which the integration-by-parts boundary terms vanish. For
`MODE == AdjointDiscrete`, let ``M_f`` be the configured discrete
multiplication by the real physical field ``f``, including padding and
truncation when dealiasing is enabled. With every ``\dagger`` taken
under the weighted [`dot`](@ref) inner product, the exact numerical transpose is

```math
(L_{h,Q}^{\dagger}p)_j=\nu(\nabla_h^2)^\dagger p_j-
\sum_{\ell=1}^3D_\ell^\dagger M_{U_\ell}p_j-
\sum_{n=1}^3M_{D_jU_n}p_n+\mathcal F_{D,j}(p),
\qquad j=1,2,3.
```

Thus the forward term ``-M_{U_\ell}D_\ell`` becomes
``-D_\ell^\dagger M_{U_\ell}``; multiplication and differentiation cannot in
general be interchanged. Fourier derivatives change sign, finite-difference
directions use the stored quadrature-weighted adjoint matrices, and the
Laplacian uses its stored second-derivative adjoints. The FFTs, dealiased
padding, physical products, and spectral truncation form the corresponding
transpose pipeline. [`AdjointDiscrete`](@ref) derives these statements from
the discrete inner product in detail.

The hydrodynamic identity is exact up to roundoff. A force belongs to it only
when `force(out,p,AdjointDiscrete())` is the weighted transpose of the
`Forward()` action; an affine source has no linear adjoint.

The four-argument call `eq(t,Q,q,out)` transforms and caches ``Q`` and its
gradients before applying the selected action. Further calls `eq(t,q,out)`
reuse that cache. These mutable workspaces make an instance neither reentrant
nor safe for concurrent calls.

# Fields
- `Re`: Reynolds number
- `plans`, `scache`, `pcache`, `force`: same as [`CartesianPrimitive3DNSE`](@ref)
"""
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

    ddx!(dudx, u)
    ddy!(dudy, u)
    ddz!(dudz, u)

    eq.plans(U, u)
    eq.plans(dUdx, dudx); eq.plans(dUdy, dudy); eq.plans(dUdz, dudz)
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
# Four arguments: prepare the base-flow cache, then delegate to the cached three-argument action.
function (eq::CartesianPrimitive3DLNSE)(::Real,
                                       u::VectorField{3, F},
                                       v::VectorField{3, F},
                                     out::VectorField{3, F}) where {F<:FTField}
    dudx = eq.scache[1]; dudy = eq.scache[2]; dudz = eq.scache[3]
    U    = eq.pcache[1]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]

    ddx!(dudx, u)
    ddy!(dudy, u)
    ddz!(dudz, u)

    eq.plans(U, u)
    eq.plans(dUdy, dudy); eq.plans(dUdz, dudz)

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

    ddx!(dvdx, v)
    ddy!(dvdy, v)
    ddz!(dvdz, v)

    eq.plans(V, v)
    eq.plans(dUdx, dudx)
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

    ddx!(dvdx, v)
    ddy!(dvdy, v)
    ddz!(dvdz, v)

    eq.plans(V, v)
    eq.plans(dUdx, dudx)
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

    laplacian!(out, v; adjoint=true)
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
        ddx!(dudx[1], u1v[n]; adjoint=true)
        ddy!(dudx[2], u2v[n]; adjoint=true)
        ddz!(dudx[3], u3v[n]; adjoint=true)
        out[n] .-= dudx[1] .+ dudx[2] .+ dudx[3]
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
