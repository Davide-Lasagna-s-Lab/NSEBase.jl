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

Nonlinear, pressure-free Navier–Stokes right-hand side for a two-dimensional,
three-component state `q = (u, v, w)`. Components one and two are the physical
`x`- and `y`-velocity, while component three is the out-of-plane velocity. With
`∇∥ = (∂ₓ, ∂ᵧ)`, component `n = 1, 2, 3` is

```math
[\\mathcal{N}(\\boldsymbol{q})]_n
  = \\frac{1}{\\mathrm{Re}}\\,\\Delta_{\\parallel}q_n
    - \\sum_{j=1}^{2} q_j\\,\\partial_j q_n
    + [\\mathcal{F}_{\\mathrm{F}}(\\boldsymbol{q})]_n.
```

Thus only `(u, v)` advects, whereas `w` is advected as a scalar-like velocity
component and does not itself advect any state component. The pressure gradient
is not formed here; [`ProjectedNSE`](@ref) removes it when projecting onto a
divergence-free supplied basis. Products are evaluated in the physical-space
cache and transformed back to spectral space, with dealiasing controlled by
[`construct_equations`](@ref).

``\\mathcal{F}_{\\mathrm{F}}`` denotes the additive in-place action
`force(out, q, Forward())`, performed after the viscous and advective terms.
[`NoForce`](@ref) makes it zero. The force callable may act on any of the three
components; it must add to `out` and leave `q` unchanged.

# Call contract

Call the operator as `eq(t, q, out)`. The arguments `q` and `out` must be
distinct `VectorField{3, <:FTField}` values compatible with the grid and FFT
plans used to construct `eq`; aliasing `out` with `q` is unsupported. The time
argument is accepted for solver compatibility but is not used. The call
overwrites `out`, returns that same object, and treats `q` as read-only. Internal
caches are mutated, so one operator instance is not reentrant or safe for
concurrent calls. `Re` must be nonzero.
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

Linearised, pressure-free Navier–Stokes operator for 2D-3C Cartesian flows. The
base state `U = (U₁, U₂, U₃)` and perturbation or adjoint state
`v = (v₁, v₂, v₃)` use in-plane `(x, y)` components first and the
out-of-plane component third.

For `MODE == Forward`, component `n = 1, 2, 3` is

```math
[\\mathcal{L}_{\\boldsymbol{U}}\\boldsymbol{v}]_n
  = \\frac{1}{\\mathrm{Re}}\\,\\Delta_{\\parallel}v_n
    - \\sum_{j=1}^{2} U_j\\,\\partial_j v_n
    - \\sum_{j=1}^{2} v_j\\,\\partial_j U_n
    + [\\mathcal{F}_{\\mathrm{F}}(\\boldsymbol{v})]_n.
```

For `MODE == AdjointContinuous`, integration by parts is performed before
discretisation:

```math
[\\mathcal{L}^{\\dagger}_{\\boldsymbol{U}}\\boldsymbol{v}]_n
  = \\frac{1}{\\mathrm{Re}}\\,\\Delta_{\\parallel}v_n
    + \\sum_{j=1}^{2} U_j\\,\\partial_j v_n
    - \\mathbf{1}_{n\\leq 2}\\sum_{m=1}^{3} v_m\\,\\partial_n U_m
    + [\\mathcal{F}_{\\mathrm{AC}}(\\boldsymbol{v})]_n.
```

The indicator makes the cross-gradient contribution zero in component three:
an out-of-plane perturbation is advected but never acts as an advecting
velocity. This identity assumes `∂ₓU₁ + ∂ᵧU₂ = 0` and boundary conditions for
which the integration-by-parts boundary terms vanish. Forward derivatives are
used deliberately in this mode.

For `MODE == AdjointDiscrete`, let ``D_{j,h}^{+}`` and
``\\Delta_{\\parallel,h}^{+}`` denote the discrete adjoints supplied by the grid:

```math
[\\mathcal{L}^{+}_{\\boldsymbol{U},h}\\boldsymbol{v}]_n
  = \\frac{1}{\\mathrm{Re}}\\,\\Delta_{\\parallel,h}^{+}v_n
    - \\sum_{j=1}^{2}D_{j,h}^{+}(U_j v_n)
    - \\mathbf{1}_{n\\leq 2}\\sum_{m=1}^{3}v_m\\,D_{n,h}U_m
    + [\\mathcal{F}_{\\mathrm{AD}}(\\boldsymbol{v})]_n,
  \\qquad n=1,2,3.
```

This transposes the implemented derivative-and-multiplication sequence rather
than discretising the continuous-adjoint formula. Its adjoint identity requires
each grid [`derivative_matrix`](@ref) to provide the correct
`AdjointDiscrete()` operator.

In every mode, ``\\mathcal{F}`` is the contribution made by
`force(out, v, mode)` after the fluid terms. NSEBase does not automatically
linearise or transpose a custom force: the callable must dispatch on the mode
tag and add the corresponding forward or adjoint action to `out`.

# Call contract

`eq(t, U, v, out)` caches `U` and its forward spatial derivatives, applies the
selected operator to `v`, overwrites `out`, and returns `out`. The shorter call
`eq(t, v, out)` reuses the existing base-state cache and is valid only after a
four-argument call or after a nonlinear operator sharing these caches has been
evaluated at that base state. The time argument is unused.

All field arguments must be grid-compatible `VectorField{3, <:FTField}` values,
and `out` must not alias `v`. Calls mutate the internal caches, so an instance
is neither reentrant nor safe for concurrent use. `MODE` must be `Forward`,
`AdjointContinuous`, or
`AdjointDiscrete`; other `Mode` subtypes can be stored by the constructor but
have no call method. [`construct_equations`](@ref) intentionally accepts only
the two adjoint modes, whereas the direct grid constructor can create all three.
`Re` must be nonzero.
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

    laplacian!(out, v, AdjointDiscrete())
    out .*= 1/eq.Re

    eq.plans(V, v)
    for n in 1:3
        @. U1V[n] = U[1]*V[n]
        @. U2V[n] = U[2]*V[n]
    end
    eq.plans(u1v, U1V); eq.plans(u2v, U2V)

    eq.plans(dUdx, dudx)
    for n in 1:3
        out[n] .-= ddx!(dudx[1], u1v[n], AdjointDiscrete()) .+
                   ddy!(dudx[2], u2v[n], AdjointDiscrete())
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
