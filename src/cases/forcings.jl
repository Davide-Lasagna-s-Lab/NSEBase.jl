# //////////////////////////////////////////////////////////////////////////// #
# ///                    No-force type and application                     /// #
# //////////////////////////////////////////////////////////////////////////// #

"""
    NoForce

Default body-force callable that applies no forcing. Its call signature is
`(out, u, mode) -> out`; it returns `out` unchanged.

Pass `NoForce()` to any NSE constructor that accepts a `force` keyword when no
body force is needed.
"""
struct NoForce end
(::NoForce)(out, _, _) = out

# //////////////////////////////////////////////////////////////////////////// #
# ///                Compound forcing type and construction                /// #
# //////////////////////////////////////////////////////////////////////////// #

"""
    CompoundForcing(forces...)

A body force that applies each of `forces` in sequence. Use this to combine
multiple body-force terms, for example:

```julia
CompoundForcing(ConstantBodyForce(), CoriolisForce(Ro))
```
"""
struct CompoundForcing{N, F<:NTuple{N, Any}}
    forces::F
end

CompoundForcing(forces...) = CompoundForcing{length(forces), typeof(forces)}(forces)

# Iterating over a heterogeneous tuple would make each force a union type. The
# tuple length is a type parameter, so generated unrolling keeps every call
# statically dispatched.
@generated function (cf::CompoundForcing{N})(out, u, mode) where {N}
    return quote
        Base.Cartesian.@nexprs $N i -> cf.forces[i](out, u, mode)
        return out
    end
end

# //////////////////////////////////////////////////////////////////////////// #
# ///                    Coriolis type and construction                    /// #
# //////////////////////////////////////////////////////////////////////////// #

"""
    CoriolisForce(rotation_number; components=(1, 2))

Coriolis body force represented by a two-component skew-symmetric coupling.
For `components=(i, j)`, the forward operator applies
`out[i] += Ro*u[j]` and `out[j] -= Ro*u[i]`; both signs reverse for continuous
and discrete adjoints.

The default `components=(1, 2)` is rotation about the third Cartesian axis for
velocity ordered as `(u, v, w)`. RPCF uses `components=(3, 1)` because its 2D3C
state is ordered as wall-normal, spanwise, and streamwise velocity.

`rotation_number` is the nondimensional coefficient multiplying the Coriolis
acceleration. In the channel and RPCF conventions it is
`Ro = 2Ωh/U_ref`, where `Ω` is angular velocity, `h` is the channel half-height,
and `U_ref` is the reference velocity (the wall speed `U_w` for RPCF). The
component tuple fixes the projected sign convention explicitly, avoiding any
ambiguity introduced by a case's velocity ordering.

# Fields

- `Ro`: stored rotation number.
- `components`: distinct velocity-component indices `(i, j)` coupled by the
  skew-symmetric operator. Both indices must lie in `1:3`.

Because the forward coupling is skew-symmetric, its discrete and continuous
adjoints are its negative; no grid-dependent adjoint data are required. The
force acts on a three-component [`VectorField`](@ref).
"""
struct CoriolisForce{T, C<:NTuple{2, Int}}
    Ro::T
    components::C
end

function CoriolisForce(rotation_number; components=(1, 2))
    i, j = components
    i != j || throw(ArgumentError("Coriolis force must couple two distinct components"))
    all(component -> 1 <= component <= 3, components) || throw(ArgumentError("Coriolis components must lie in 1:3"))
    return CoriolisForce(rotation_number, components)
end

function (f::CoriolisForce)(out::VectorField{3}, v::VectorField{3}, ::Forward)
    i, j = f.components
    @. out[i] += f.Ro * v[j]
    @. out[j] -= f.Ro * v[i]
    return out
end

function (f::CoriolisForce)(out::VectorField{3}, v::VectorField{3}, ::Union{AdjointDiscrete, AdjointContinuous})
    i, j = f.components
    @. out[i] -= f.Ro * v[j]
    @. out[j] += f.Ro * v[i]
    return out
end

# //////////////////////////////////////////////////////////////////////////// #
# ///              Constant body force type and construction               /// #
# //////////////////////////////////////////////////////////////////////////// #

"""
    ConstantBodyForce(value=1; component=1)

Spatially constant force applied to one velocity component at the zero Fourier
mode. `component` follows the field's velocity ordering; for three-dimensional
Cartesian cases, `1 => x`, `2 => y`, and `3 => z`. The forcing is applied for
every [`Mode`](@ref), making it suitable for a constant pressure gradient in
channel and duct cases.

The selected mean-mode slice still spans every finite-difference direction,
so the same value is added at every wall-normal channel point or every `(x,y)`
duct collocation point. It is applied in nonlinear, forward-linearised, and
adjoint-linearised equation modes, matching the forcing convention of the
original flow packages.

# Fields

- `value`: forcing amplitude.
- `component`: one-based velocity component receiving the force; it must lie in
  `1:N` when applied to an `N`-component field.

# Pressure-gradient normalisation

For plane Poiseuille flow nondimensionalised with friction velocity `u_τ` and
channel half-height `h`, `Re = Re_τ = u_τ h/ν` and the conventional streamwise
forcing is `ConstantBodyForce(1; component=1)`. For a duct using the analogous
friction scaling and half-width `h`, use `component=3`. The case constructors
[`PlanePoiseuilleFlow`](@ref) and [`SquareDuctFlow`](@ref) select these components
automatically and default to amplitude one.
"""
struct ConstantBodyForce{T}
    value::T
    component::Int
end

ConstantBodyForce(value=1; component=1) = ConstantBodyForce(value, component)

function (f::ConstantBodyForce)(out::VectorField{N, <:FTField}, _, ::Mode) where {N}
    1 <= f.component <= N || throw(ArgumentError("force component must lie in 1:$N"))
    g = grid(out[f.component])
    # WaveNumberVector follows FFT_DIMS order, so one zero is needed per transform.
    mean_mode = WaveNumberVector(ntuple(_ -> 0, length(fft_storage_dims(g))))
    mean = out[f.component][mean_mode]
    mean .+= f.value
    return out
end
