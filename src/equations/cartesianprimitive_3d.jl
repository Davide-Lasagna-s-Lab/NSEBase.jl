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
"""
    CartesianPrimitive3DNSE{T, FFT, S, P, BF}

Nonlinear, pressure-free Navier–Stokes right-hand side for a three-component
Cartesian state `q = (u, v, w)`, ordered as the physical `x`, `y`, and `z`
velocity components. With `∇ = (∂ₓ, ∂ᵧ, ∂ᶻ)`, the operator evaluates

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
distinct `VectorField{3, <:FTField}` values compatible with the grid and FFT
plans used to construct `eq`; aliasing `out` with `q` is unsupported. The time
argument is accepted for solver compatibility but is not used. The call
overwrites `out`, returns that same object, and treats `q` as read-only. Internal
caches are mutated, so one operator instance is not reentrant or safe for
concurrent calls.

# Fields
- `Re`: Reynolds number; it must be nonzero.
- `plans`: [`FFTPlans`](@ref) for physical↔spectral transforms.
- `scache`, `pcache`: mutable spectral- and physical-space scratch fields.
- `force`: additive callable with signature `(out, q, mode) -> out`.
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
"""
    CartesianPrimitive3DLNSE{MODE, T, FFT, S, P, BF}

Linearised, pressure-free Navier–Stokes operator for three-component Cartesian
flows. The base state `U = (U₁, U₂, U₃)` and perturbation or adjoint state
`v = (v₁, v₂, v₃)` use physical `(x, y, z)` component order.

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
discrete adjoints supplied by the grid. Component `n = 1, 2, 3` is evaluated as

```math
[\\mathcal{L}^{+}_{\\boldsymbol{U},h}\\boldsymbol{v}]_n
  = \\frac{1}{\\mathrm{Re}}\\,\\Delta_h^{+}v_n
    - \\sum_{j=1}^{3} D_{j,h}^{+}(U_j v_n)
    - \\sum_{m=1}^{3} v_m\\,D_{n,h}U_m
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

All field arguments must be grid-compatible `VectorField{3, <:FTField}` values,
and `out` must not alias `v`. Calls mutate the internal caches, so an instance
is neither reentrant nor safe for concurrent use. `MODE` must be `Forward`,
`AdjointContinuous`, or
`AdjointDiscrete`; other `Mode` subtypes can be stored by the constructor but
have no call method. [`construct_equations`](@ref) intentionally accepts only
the two adjoint modes, whereas the direct grid constructor can create all three.

# Fields
- `Re`: Reynolds number; it must be nonzero.
- `plans`, `scache`, `pcache`, `force`: as for [`CartesianPrimitive3DNSE`](@ref).
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
# 3-arg: set up base-flow cache then delegate to 2-arg
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

    laplacian!(out, v, AdjointDiscrete())
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
        ddx!(dudx[1], u1v[n], AdjointDiscrete())
        ddy!(dudx[2], u2v[n], AdjointDiscrete())
        ddz!(dudx[3], u3v[n], AdjointDiscrete())
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
