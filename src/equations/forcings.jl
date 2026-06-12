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


"""
    BuoyancyForce(Ri, grav, temp)

Boussinesq buoyancy coupling expressed as a body force: the temperature
component `temp` of the state drives the velocity component `grav` (the gravity
direction), scaled by the Richardson number `Ri`. This is what makes a `{3,4}`
velocity+temperature operator a Boussinesq operator — the coupling is a force,
not part of the advection.

Forward: `out[grav] += Ri·u[temp]`; the (continuous and discrete) adjoint is its
transpose, `out[temp] += Ri·u[grav]`.
"""
struct BuoyancyForce{T}
      Ri::T
    grav::Int
    temp::Int
end

(f::BuoyancyForce)(out, u, ::Forward) = (out[f.grav] .+= f.Ri .* u[f.temp]; out)
(f::BuoyancyForce)(out, u, ::Union{AdjointContinuous, AdjointDiscrete}) =
    (out[f.temp] .+= f.Ri .* u[f.grav]; out)
