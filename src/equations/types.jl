# Shared equation-mode types and the default no-op body-force functor.
# These are generic across all NSE formulations (Cartesian, polar, …).

# ---------- #
# mode types #
# ---------- #
abstract type               Mode end
struct Forward           <: Mode end
struct AdjointDiscrete   <: Mode end
struct AdjointContinuous <: Mode end


# -------------------------- #
# default body force (no-op) #
# -------------------------- #
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
