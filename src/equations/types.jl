# Equation-mode tags, the no-op body force, and the compound force combinator.
#
# Mode tags control which flavour of the linearised Navier-Stokes operator is
# applied.  Each concrete NSE/LNSE operator is parameterised on a mode type so
# that the compiler generates specialised, zero-overhead method bodies for each
# adjoint variant rather than branching at runtime.
#
# `NoForce` is the default body-force callable: it receives `(out, u, mode)` and
# returns `out` unchanged, imposing no additional forcing.
#
# `CompoundForcing` chains multiple body-force callables in sequence.  The call
# is unrolled at compile time via `@nexprs` so that dynamic dispatch is avoided
# even when the individual force types differ.

# ---------- #
# mode types #
# ---------- #
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


# -------------------------- #
# default body force (no-op) #
# -------------------------- #
"""
    NoForce

Default body-force callable that applies no forcing.  Its call signature is
`(out, u, mode) -> out`; it returns `out` unchanged.

Pass `NoForce()` to any NSE constructor that accepts a `force` keyword when no
body force is needed.
"""
struct NoForce end
(::NoForce)(out, _, _) = out


# ------------------------------------- #
# compound force: sequence of forces    #
# ------------------------------------- #
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
