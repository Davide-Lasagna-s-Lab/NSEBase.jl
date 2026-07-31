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

Nonlinear, pressure-free Navier–Stokes right-hand side for a planar state
`q = (u, v)`, ordered as the physical `x`- and `y`-velocity components. With
`∇ = (∂ₓ, ∂ᵧ)`, the operator evaluates

```math
\\mathcal{N}(\\boldsymbol{q}) = \\frac{1}{\\mathrm{Re}}\\,\\Delta\\boldsymbol{q}
    - (\\boldsymbol{q}\\!\\cdot\\!\\nabla)\\boldsymbol{q}
    + \\mathcal{F}_{\\mathrm{F}}(\\boldsymbol{q}).
```

The pressure gradient is not formed here. In [`ProjectedNSE`](@ref), projection
onto a divergence-free supplied basis removes the pressure contribution.
Nonlinear products are evaluated in the physical-space cache and transformed
back to spectral space; constructors created through
[`construct_equations`](@ref) control whether that cache is dealiased.

``\\mathcal{F}_{\\mathrm{F}}`` denotes the additive in-place action
`force(out, q, Forward())`, performed after the viscous and advective terms.
[`NoForce`](@ref) makes it zero. A custom force must add its contribution to
`out`, leave `q` unchanged, and implement any required state dependence itself.

# Call contract

Call the operator as `eq(t, q, out)`. The arguments `q` and `out` must be
distinct `VectorField{2, <:FTField}` values compatible with the grid and FFT
plans used to construct `eq`; aliasing `out` with `q` is unsupported. The time
argument is accepted for solver compatibility but is not used. The call
overwrites `out`, returns that same object, and treats `q` as read-only. Internal
caches are mutated, so one operator instance is not reentrant or safe for
concurrent calls. `Re` must be nonzero.
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

Linearised, pressure-free Navier–Stokes operator for planar Cartesian flows. The
base state `U = (U₁, U₂)` and perturbation or adjoint state `v = (v₁, v₂)`
use physical `(x, y)` component order.

For `MODE == Forward`, the implemented linearisation is

```math
\\mathcal{L}_{\\boldsymbol{U}}\\boldsymbol{v}
  = \\frac{1}{\\mathrm{Re}}\\,\\Delta\\boldsymbol{v}
    - (\\boldsymbol{U}\\!\\cdot\\!\\nabla)\\boldsymbol{v}
    - (\\boldsymbol{v}\\!\\cdot\\!\\nabla)\\boldsymbol{U}
    + \\mathcal{F}_{\\mathrm{F}}(\\boldsymbol{v}).
```

For `MODE == AdjointContinuous`, integration by parts is performed before
discretisation:

```math
\\mathcal{L}^{\\dagger}_{\\boldsymbol{U}}\\boldsymbol{v}
  = \\frac{1}{\\mathrm{Re}}\\,\\Delta\\boldsymbol{v}
    + (\\boldsymbol{U}\\!\\cdot\\!\\nabla)\\boldsymbol{v}
    - (\\nabla\\boldsymbol{U})^{\\mathsf{T}}\\boldsymbol{v}
    + \\mathcal{F}_{\\mathrm{AC}}(\\boldsymbol{v}).
```

This continuous-adjoint identity assumes a solenoidal base velocity and
boundary conditions for which the integration-by-parts boundary terms vanish.
The implementation deliberately uses forward derivative operators for this
mode.

For `MODE == AdjointDiscrete`, let ``D_{j,h}^{+}`` and ``\\Delta_h^{+}`` denote the
discrete adjoints supplied by the grid. Component `n = 1, 2` is evaluated as

```math
[\\mathcal{L}^{+}_{\\boldsymbol{U},h}\\boldsymbol{v}]_n
  = \\frac{1}{\\mathrm{Re}}\\,\\Delta_h^{+}v_n
    - \\sum_{j=1}^{2} D_{j,h}^{+}(U_j v_n)
    - \\sum_{m=1}^{2} v_m\\,D_{n,h}U_m
    + [\\mathcal{F}_{\\mathrm{AD}}(\\boldsymbol{v})]_n.
```

Thus the discrete mode transposes the actual derivative-and-multiplication
sequence; it is not obtained by substituting discrete derivatives into the
continuous formula. Its adjoint identity requires each grid
[`derivative_matrix`](@ref) to return the correct `AdjointDiscrete()` operator.

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

All field arguments must be grid-compatible `VectorField{2, <:FTField}` values,
and `out` must not alias `v`. Calls mutate the internal caches, so an instance
is neither reentrant nor safe for concurrent use. `MODE` must be `Forward`,
`AdjointContinuous`, or
`AdjointDiscrete`; other `Mode` subtypes can be stored by the constructor but
have no call method. [`construct_equations`](@ref) intentionally accepts only
the two adjoint modes, whereas the direct grid constructor can create all three.
`Re` must be nonzero.
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

    laplacian!(out, v, AdjointDiscrete())
    out .*= 1/eq.Re

    eq.plans(V, v)
    for n in 1:2
        @. U1V[n] = U[1]*V[n]
        @. U2V[n] = U[2]*V[n]
    end
    eq.plans(u1v, U1V); eq.plans(u2v, U2V)

    eq.plans(dUdx, dudx)
    for n in 1:2
        ddx!(dudx[1], u1v[n], AdjointDiscrete())
        ddy!(dudx[2], u2v[n], AdjointDiscrete())
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
