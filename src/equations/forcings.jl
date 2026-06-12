# Body forces: callables with signature `(out, u, mode) -> out` that add a
# forcing term to the operator residual. The operator's `force` field holds one
# of these; downstream packages add flow-specific forces (mean pressure gradient,
# Coriolis, …) the same way.

"""
    NoForce

Default body-force callable that applies no forcing. Its call signature is
`(out, u, mode) -> out`; it returns `out` unchanged.
"""
struct NoForce end
(::NoForce)(out, _, _) = out


"""
    CompoundForcing(forces...)

A body force that applies each of `forces` in sequence, e.g.
`CompoundForcing(ConstantForcing(), CoriolisForce(Ro))`.
"""
struct CompoundForcing{N, F<:NTuple{N, Any}}
    forces::F
end
CompoundForcing(forces...) = CompoundForcing{length(forces), typeof(forces)}(forces)

# N is a type parameter so @nexprs unrolls the calls into N statically-typed
# statements, keeping dispatch fully specialised (a plain `for` loop over the
# tuple would dynamically dispatch on the Union of element types).
@generated function (cf::CompoundForcing{N})(out, u, mode) where {N}
    return quote
        Base.Cartesian.@nexprs $N i -> cf.forces[i](out, u, mode)
        return out
    end
end
