# Concrete Navier–Stokes operators for streamwise-invariant Cartesian flows with three velocity
# components (2D3C). Physical `x` is absent, the flow evolves in the `(y,z)` plane, and the velocity
# state is `(v,w,u)`: wall-normal and spanwise velocity followed by streamwise velocity.
#
# The streamwise-velocity equation is
#
#   ∂u/∂t = Δu/Re − (v∂u/∂y + w∂u/∂z).
#
# The streamwise velocity does not contribute to advection or to the two-dimensional
# incompressibility constraint `∂v/∂y + ∂w/∂z = 0`. Physical coordinates are never relabelled:
# `ddy!` is wall-normal, `ddz!` is spanwise, and `ddx!` remains the absent streamwise derivative.
#
# Variants provided:
#   CartesianPrimitive2D3CNSE  — nonlinear NSE: out = Δu/Re − (u_in·∇)u + force
#   CartesianPrimitive2D3CLNSE{Forward}          — forward linearised operator
#   CartesianPrimitive2D3CLNSE{AdjointContinuous}— continuous adjoint
#   CartesianPrimitive2D3CLNSE{AdjointDiscrete}  — discrete adjoint
#
# All cache layouts and operator conventions follow cartesianprimitive_2d.jl, with the structural
# changes being `VectorField{3}` and loops running to three. In the linearised operators, only the
# wall-normal and spanwise perturbations contribute to base-flow advection; streamwise velocity is
# advected but does not advect.

function _check_cartesian_2d3c_grid(::AbstractGrid{<:Any, <:Any, AXES}) where {AXES}
    x_dim, y_dim, z_dim = AXES[1], AXES[2], AXES[3]
    valid = isnothing(x_dim) && !isnothing(y_dim) && !isnothing(z_dim) && y_dim != z_dim
    valid || throw(ArgumentError("CartesianPrimitive2D3C requires a streamwise-invariant physical " *
                                 "(y,z) grid with x absent; got AXES=$AXES"))
    return nothing
end


# ------------------------- #
# concrete 2D3C NSE struct  #
# ------------------------- #
@doc raw"""
    CartesianPrimitive2D3CNSE{T, FFT, S, P, BF}
    CartesianPrimitive2D3CNSE(grid, Re; force=NoForce(), flags=FFTW.EXHAUSTIVE)

Nonlinear Navier–Stokes operator for streamwise-invariant 2D3C Cartesian
flows. Physical `x` is absent, the active plane is `(y,z)`, and state order is
``Q=(Q_1,Q_2,Q_3)=(V,W,U)``: wall-normal, spanwise, then streamwise velocity.

With ``D_1=\partial_y``, ``D_2=\partial_z``, and ``\nu=Re^{-1}``, the call
`eq(t,Q,out)` evaluates

```math
\mathcal N_n(Q)=\nu\nabla_{yz,h}^2Q_n-VD_1Q_n-WD_2Q_n
                 +\mathcal F_{N,n}(Q),\qquad n=1,2,3.
```

Equivalently, ``V`` and ``W`` advect all three components, whereas the
streamwise component ``U`` never acts as an advecting velocity. Nonlinear
products are formed in physical space and use padded transforms when
dealiasing is enabled. The in-plane incompressibility constraint is
``\partial_yV+\partial_zW=0``.

The argument `t` is ignored, and a transformed logical `t` direction is not a
spatial derivative or part of the Laplacian. This raw operator contains no
pressure solve; an admissible [`ProjectedNSE`](@ref) basis handles the
in-plane pressure gradient. ``\mathcal F_N`` is the actual
`force(out,Q,Forward())` action and may act on all three components.
"""
mutable struct CartesianPrimitive2D3CNSE{T, FFT, S, P, BF}
              Re::T
     const plans::FFT
    const scache::Vector{VectorField{3, S}}
    const pcache::Vector{VectorField{3, P}}
    const  force::BF

    function CartesianPrimitive2D3CNSE(
                Re::T, plans::FFT, scache::Vector{VectorField{3, S}},
                pcache::Vector{VectorField{3, P}}, force::BF) where {T, FFT, S, P, BF}
        _check_cartesian_2d3c_grid(grid(scache[1][1]))
        return new{T, FFT, S, P, BF}(Re, plans, scache, pcache, force)
    end
end

function CartesianPrimitive2D3CNSE(g::G, Re;
                                   force::BF=NoForce(),
                                   flags=FFTW.EXHAUSTIVE) where {T, G<:AbstractGrid{T}, BF}
    _check_cartesian_2d3c_grid(g)
    plans  = FFTPlans(g, flags=flags)
    scache = [VectorField([FTField(g)               for _ in 1:3]...) for _ in 1:2]
    pcache = [VectorField([  Field(g, dealias=true) for _ in 1:3]...) for _ in 1:3]
    return CartesianPrimitive2D3CNSE(T(Re), plans, scache, pcache, force)
end

# ------------------------- #
# concrete 2D3C LNSE struct #
# ------------------------- #
@doc raw"""
    CartesianPrimitive2D3CLNSE{MODE, T, FFT, S, P, BF}
    CartesianPrimitive2D3CLNSE(grid, Re; mode=AdjointDiscrete(),
                               force=NoForce(), flags=FFTW.EXHAUSTIVE)

Forward or adjoint linearisation of [`CartesianPrimitive2D3CNSE`](@ref).
Retain the physical ordering ``Q=(V,W,U)``, write the forward perturbation as
``q=(v,w,u)``, the adjoint field as ``p=(a,b,c)``, and let
``(D_1,D_2)=(\partial_y,\partial_z)`` and ``\nu=Re^{-1}``.

For `MODE == Forward`, every state component is advected, but only ``v`` and
``w`` advect the base:

```math
(L_Qq)_n=\nu\nabla_{yz,h}^2q_n-VD_1q_n-WD_2q_n
          -vD_1Q_n-wD_2Q_n+\mathcal F_{F,n}(q),\qquad n=1,2,3.
```

For `MODE == AdjointContinuous`, integration by parts gives

```math
\begin{aligned}
(L_Q^{\dagger,c}p)_1
 &=\nu\nabla_{yz}^2a+V\partial_ya+W\partial_za
   -\sum_{n=1}^3p_n\partial_yQ_n+\mathcal F_{C,1}(p),\\
(L_Q^{\dagger,c}p)_2
 &=\nu\nabla_{yz}^2b+V\partial_yb+W\partial_zb
   -\sum_{n=1}^3p_n\partial_zQ_n+\mathcal F_{C,2}(p),\\
(L_Q^{\dagger,c}p)_3
 &=\nu\nabla_{yz}^2c+V\partial_yc+W\partial_zc+\mathcal F_{C,3}(p).
\end{aligned}
```

This form assumes ``\partial_yV+\partial_zW=0`` and boundary conditions that
eliminate integration-by-parts terms. For `MODE == AdjointDiscrete`, let
``M_f`` denote the configured discrete multiplication by real ``f``, including
padding and truncation when dealiasing is enabled, and take every ``\dagger``
under the weighted [`dot`](@ref) inner product. The three implemented equations are

```math
\begin{aligned}
(L_{h,Q}^{\dagger}p)_1
 &=\nu(\nabla_{yz,h}^2)^\dagger a-D_1^\dagger M_Va-D_2^\dagger M_Wa
   -\sum_{n=1}^3M_{D_1Q_n}p_n+\mathcal F_{D,1}(p),\\
(L_{h,Q}^{\dagger}p)_2
 &=\nu(\nabla_{yz,h}^2)^\dagger b-D_1^\dagger M_Vb-D_2^\dagger M_Wb
   -\sum_{n=1}^3M_{D_2Q_n}p_n+\mathcal F_{D,2}(p),\\
(L_{h,Q}^{\dagger}p)_3
 &=\nu(\nabla_{yz,h}^2)^\dagger c-D_1^\dagger M_Vc-D_2^\dagger M_Wc
   +\mathcal F_{D,3}(p).
\end{aligned}
```

The absence of a cross-gradient sum in component three is the transpose of
the fact that streamwise perturbation ``u`` does not advect the base. The
ordering ``-D_j^\dagger M_{Q_j}`` reverses the forward
``-M_{Q_j}D_j`` computation, including FFT padding and truncation. See
[`AdjointDiscrete`](@ref) for the complete derivation. A force is part of the
adjoint identity only if its discrete-adjoint action is the weighted transpose
of its forward action.

The four-argument call `eq(t,Q,q,out)` prepares the base cache; later
three-argument calls `eq(t,q,out)` reuse it. The mutable cache makes an
instance neither reentrant nor safe for concurrent calls.
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
                                     force::BF) where {MODE, T, FFT, S, P, BF} = begin
        _check_cartesian_2d3c_grid(grid(scache[1][1]))
        new{MODE, T, FFT, S, P, BF}(Re, plans, scache, pcache, force)
    end
end

function CartesianPrimitive2D3CLNSE(g::G, Re;
                                    mode::Mode=AdjointDiscrete(),
                                    force::BF=NoForce(),
                                    flags=FFTW.EXHAUSTIVE) where {T, G<:AbstractGrid{T}, BF}
    _check_cartesian_2d3c_grid(g)
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
    dudy, dudz = eq.scache
    U, dUdy, dUdz = eq.pcache

    laplacian!(out, u)
    out .*= 1/eq.Re

    ddy!(dudy, u); ddz!(dudz, u)

    eq.plans(U, u); eq.plans(dUdy, dudy); eq.plans(dUdz, dudz)
    for n in 1:3
        @. dUdy[n] = -U[1]*dUdy[n] - U[2]*dUdz[n]
    end
    eq.plans(out, dUdy, add=true)

    eq.force(out, u, Forward())
    return out
end


# -------------- #
# linearised NSE #
# -------------- #
# Four arguments: set up the base-flow cache, then delegate to the cached three-argument action.
function (eq::CartesianPrimitive2D3CLNSE)(::Real,
                                         u::VectorField{3, F},
                                         v::VectorField{3, F},
                                       out::VectorField{3, F}) where {F<:FTField}
    dudy, dudz = eq.scache[1], eq.scache[2]
    U, dUdz = eq.pcache[1], eq.pcache[3]

    ddy!(dudy, u); ddz!(dudz, u)
    eq.plans(U, u); eq.plans(dUdz, dudz)

    eq(0, v, out)
    return out
end

# forward LNSE
function (eq::CartesianPrimitive2D3CLNSE{Forward})(::Real,
                                                   v::VectorField{3, F},
                                                 out::VectorField{3, F}) where {F<:FTField}
    dudy, dvdy, dvdz = eq.scache
    U, dUdy, dUdz, V, dVdy, dVdz = eq.pcache

    laplacian!(out, v)
    out .*= 1/eq.Re

    ddy!(dvdy, v); ddz!(dvdz, v)

    eq.plans(V, v); eq.plans(dUdy, dudy)
    eq.plans(dVdy, dvdy); eq.plans(dVdz, dvdz)
    for n in 1:3
        @. dVdy[n]  = -U[1]*dVdy[n] - U[2]*dVdz[n]
        @. dVdy[n] -=  V[1]*dUdy[n] + V[2]*dUdz[n]
    end
    eq.plans(out, dVdy, add=true)

    eq.force(out, v, Forward())
    return out
end

# continuous adjoint LNSE
function (eq::CartesianPrimitive2D3CLNSE{AdjointContinuous})(::Real,
                                                             v::VectorField{3, F},
                                                           out::VectorField{3, F}) where {F<:FTField}
    dudy, dvdy, dvdz = eq.scache
    U, dUdy, dUdz, V, dVdy, dVdz = eq.pcache

    laplacian!(out, v)
    out .*= 1/eq.Re

    ddy!(dvdy, v); ddz!(dvdz, v)

    eq.plans(V, v); eq.plans(dUdy, dudy)
    eq.plans(dVdy, dvdy); eq.plans(dVdz, dvdz)
    for n in 1:3
        @. dVdy[n] = U[1]*dVdy[n] + U[2]*dVdz[n]
    end
    dVdz .= 0
    for i in 1:3
        @. dVdz[1] -= V[i]*dUdy[i]
        @. dVdz[2] -= V[i]*dUdz[i]
    end
    eq.plans(out, dVdy, add=true); eq.plans(out, dVdz, add=true)

    eq.force(out, v, AdjointContinuous())
    return out
end

# discrete adjoint LNSE
function (eq::CartesianPrimitive2D3CLNSE{AdjointDiscrete})(::Real,
                                                           v::VectorField{3, F},
                                                         out::VectorField{3, F}) where {F<:FTField}
    dudy, vv, wv = eq.scache
    U, dUdy, dUdz, V, VV, WV = eq.pcache

    laplacian!(out, v, adjoint=true)
    out .*= 1/eq.Re

    eq.plans(V, v)
    for n in 1:3
        @. VV[n] = U[1]*V[n]
        @. WV[n] = U[2]*V[n]
    end
    eq.plans(vv, VV); eq.plans(wv, WV)

    eq.plans(dUdy, dudy)
    for n in 1:3
        out[n] .-= ddy!(dudy[1], vv[n], adjoint=true) .+
                   ddz!(dudy[2], wv[n], adjoint=true)
    end
    VV .= 0
    for n in 1:3
        @. VV[1] -= V[n]*dUdy[n]
        @. VV[2] -= V[n]*dUdz[n]
    end
    eq.plans(out, VV, add=true)

    eq.force(out, v, AdjointDiscrete())
    return out
end
