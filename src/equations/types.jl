# Equation tag and forcing types.
#
# These are deliberately tiny, mostly zero-field types. NSE/LNSE operators carry
# them as type parameters so the compiler specialises on mode and advection form
# without runtime branches in the operator kernels.


# ============================================================================ #
# Mode Tags                                                                    #
# ============================================================================ #
"""
    Mode

Abstract supertype for equation-mode tags.  Concrete subtypes select which
variant of the linearised Navier-Stokes operator is applied.
"""
abstract type Mode end

"""
    Forward <: Mode

Tag selecting the forward linearised operator L·v, i.e. the linearisation of
the NSE around the current base flow in the forward-time direction.
"""
struct Forward           <: Mode end

"""
    AdjointDiscrete <: Mode

Tag selecting the discrete adjoint of the forward linearised operator.
The discrete adjoint is derived by transposing the discrete operator exactly —
it satisfies ⟨L·v, w⟩ = ⟨v, L*·w⟩ with respect to the discrete inner product
used in [`dot`](@ref).
"""
struct AdjointDiscrete   <: Mode end

"""
    AdjointContinuous <: Mode

Tag selecting the continuous adjoint of the linearised operator.  The
continuous adjoint is derived by integration-by-parts before discretisation,
which gives a different operator from the discrete adjoint for finite
resolution.
"""
struct AdjointContinuous <: Mode end


# ============================================================================ #
# Advection-Form Tags                                                          #
# ============================================================================ #
# The nonlinear advection term of the Navier-Stokes equations can be written in
# several mathematically equivalent forms that differ in transform count and in
# their discrete aliasing / conservation properties. In a pseudo-spectral /
# FD-hybrid solver the FFTs dominate, so the transform count is often the useful
# figure of merit.
#
#   Advective    N_i = u_j ∂_j u_i
#   Divergence   N_i = ∂_j (u_i u_j)
#   Rotational   N   = ω × u  (+ ∇(½|u|²) dropped)
#
# The divergence form also applies to passively-advected scalars because
# ∂_j u_j = 0. The rotational form applies only to velocity self-advection;
# scalar / out-of-plane components fall back to divergence treatment inside the
# rotational advection helper.
"""
    AdvectionForm

Abstract supertype for advection-form tags. Concrete subtypes ([`Advective`](@ref),
[`Divergence`](@ref), [`Rotational`](@ref)) select how the nonlinear advection
term is evaluated. The choice is a type parameter on each NSE/LNSE operator, so
the form is fixed at compile time with no runtime dispatch in the hot loop.
"""
abstract type AdvectionForm end

"""
    Advective <: AdvectionForm

Convective form of the advection term, `N_i = u_j ∂_j u_i`. This is the textbook
form and the default. It transforms the velocity and all of its first
derivatives to physical space.
"""
struct Advective <: AdvectionForm end

"""
    Divergence <: AdvectionForm

Conservative form of the advection term, `N_i = ∂_j(u_i u_j)`. The velocity is
transformed to physical space once, the quadratic products are formed there and
transformed back, and the divergence is taken with the cheap spectral / FD
derivatives. Fewer transforms than [`Advective`](@ref); valid for
divergence-free fields and for passively-advected scalars.
"""
struct Divergence <: AdvectionForm end

"""
    Rotational <: AdvectionForm

Rotational (Lamb) form of the advection term, `N = ω × u`, using the identity
`(u·∇)u = ω × u + ∇(½|u|²)` and dropping the gradient term (removed by the
projection). Fewer transforms than [`Advective`](@ref). Applies to the velocity
self-advection block only; scalar / out-of-plane components use the divergence
treatment.
"""
struct Rotational <: AdvectionForm end


# ============================================================================ #
# Body Forces                                                                  #
# ============================================================================ #
"""
    NoForce

Default body-force callable that applies no forcing.  Its call signature is
`(out, u, mode) -> out`; it returns `out` unchanged.

Pass `NoForce()` to any NSE constructor that accepts a `force` keyword when no
body force is needed.
"""
struct NoForce end
(::NoForce)(out, _, _) = out


"""
    CompoundForcing(forces...)

A body force that applies each of `forces` in sequence. Use this to combine
multiple body-force terms, e.g.:

```julia
CompoundForcing(ConstantForcing(), CoriolisForce(Ro))
```
"""
struct CompoundForcing{N, F<:NTuple{N, Any}}
    forces::F
end
CompoundForcing(forces...) = CompoundForcing{length(forces), typeof(forces)}(forces)

# A plain `for f in cf.forces` loop iterates via `iterate(::Tuple, ::Int)`, which
# returns a Union of all element types and causes dynamic dispatch for each call.
# N is a type parameter so @nexprs can unroll the calls into N statically-typed
# statements at compile time, keeping dispatch fully specialised.
@generated function (cf::CompoundForcing{N})(out, u, mode) where {N}
    return quote
        Base.Cartesian.@nexprs $N i -> cf.forces[i](out, u, mode)
        return out
    end
end
