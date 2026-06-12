# Three-component (u, v, w) Cartesian NSE / LNSE operators.
#
# The operator structs carry an `AdvectionForm` type parameter (default
# `Advective`) and the call methods are thin skeletons:
#
#     viscous term  →  form-dispatched advection helper  →  body force
#
# Only the advection helpers (in advection.jl) depend on the form. The cache
# pools are sized for the advective form (the largest); the leaner divergence /
# rotational forms use a subset of the same pool.
#
#   CartesianPrimitive3DNSE                       — out = Δu/Re − (u·∇)u + force
#   CartesianPrimitive3DLNSE{Forward}             — forward linearised operator
#   CartesianPrimitive3DLNSE{AdjointContinuous}   — continuous adjoint
#   CartesianPrimitive3DLNSE{AdjointDiscrete}     — discrete adjoint (advective only, Phase A)


# ----------------------- #
# concrete 3D NSE struct  #
# ----------------------- #
"""
    CartesianPrimitive3DNSE{FORM, T, FFT, S, P, BF}

Nonlinear Navier-Stokes operator for three-component Cartesian flows.

Evaluates `out = Δu/Re − (u·∇)u + force(out, u, Forward())` in spectral space,
using dealiased physical-space products for the nonlinear term. The `FORM` type
parameter ([`Advective`](@ref), [`Divergence`](@ref), [`Rotational`](@ref))
selects how the advection term is computed.

# Fields
- `Re`: Reynolds number
- `plans`: `FFTPlans` for physical↔spectral transforms
- `scache`: spectral-space scratch `VectorField`s
- `pcache`: physical-space scratch `VectorField`s
- `force`: body-force callable with signature `(out, u, mode)`
"""
mutable struct CartesianPrimitive3DNSE{FORM<:AdvectionForm, T, FFT, S, P, BF}
              Re::T
     const plans::FFT
    const scache::Vector{VectorField{3, S}}
    const pcache::Vector{VectorField{3, P}}
    const  force::BF

    CartesianPrimitive3DNSE{FORM}(Re::T,
                               plans::FFT,
                              scache::Vector{VectorField{3, S}},
                              pcache::Vector{VectorField{3, P}},
                               force::BF) where {FORM, T, FFT, S, P, BF} =
        new{FORM, T, FFT, S, P, BF}(Re, plans, scache, pcache, force)
end

function CartesianPrimitive3DNSE(g::G, Re;
                               form::AdvectionForm=Advective(),
                              force::BF           =NoForce(),
                              flags               =FFTW.EXHAUSTIVE) where {T, G<:AbstractGrid{T}, BF}
    plans          = FFTPlans(g, flags=flags)
    scache, pcache = alloc_caches(g, 3, 3, 4)
    return CartesianPrimitive3DNSE{typeof(form)}(T(Re), plans, scache, pcache, force)
end

# Positional constructor used by `construct_equations` (shared caches). Defaults
# to the advective form; `construct_equations` passes a form via the keyword path.
CartesianPrimitive3DNSE(Re, plans, scache::Vector{<:VectorField{3}},
                        pcache::Vector{<:VectorField{3}}, force) =
    CartesianPrimitive3DNSE{Advective}(Re, plans, scache, pcache, force)

# ----------------------- #
# concrete 3D LNSE struct #
# ----------------------- #
"""
    CartesianPrimitive3DLNSE{MODE, FORM, T, FFT, S, P, BF}

Linearised Navier-Stokes operator for three-component Cartesian flows,
parameterised on the adjoint `MODE <: Mode` and the advection `FORM`.

The three-argument call `eq(t, u, v, out)` caches the physical base-flow
quantities needed by `FORM` from `u`, then delegates to the two-argument form
`eq(t, v, out)` which applies the chosen linearised operator to `v`.

# Fields
- Same as [`CartesianPrimitive3DNSE`](@ref).
"""
mutable struct CartesianPrimitive3DLNSE{MODE, FORM<:AdvectionForm, T, FFT, S, P, BF}
              Re::T
     const plans::FFT
    const scache::Vector{VectorField{3, S}}
    const pcache::Vector{VectorField{3, P}}
    const  force::BF

    CartesianPrimitive3DLNSE{MODE, FORM}(Re::T,
                                      plans::FFT,
                                     scache::Vector{VectorField{3, S}},
                                     pcache::Vector{VectorField{3, P}},
                                      force::BF) where {MODE, FORM, T, FFT, S, P, BF} =
        new{MODE, FORM, T, FFT, S, P, BF}(Re, plans, scache, pcache, force)
end

function CartesianPrimitive3DLNSE(g::G, Re;
                                mode::Mode         =AdjointDiscrete(),
                                form::AdvectionForm=Advective(),
                               force::BF           =NoForce(),
                               flags               =FFTW.EXHAUSTIVE) where {T, G<:AbstractGrid{T}, BF}
    plans          = FFTPlans(g, flags=flags)
    scache, pcache = alloc_caches(g, 3, 4, 8)
    return CartesianPrimitive3DLNSE{typeof(mode), typeof(form)}(T(Re), plans, scache, pcache, force)
end

# Positional constructor used by `construct_equations` (mode applied, shared
# caches). Defaults to the advective form.
CartesianPrimitive3DLNSE{MODE}(Re, plans, scache::Vector{<:VectorField{3}},
                               pcache::Vector{<:VectorField{3}}, force) where {MODE} =
    CartesianPrimitive3DLNSE{MODE, Advective}(Re, plans, scache, pcache, force)


# ------------- #
# nonlinear NSE #
# ------------- #
function (eq::CartesianPrimitive3DNSE{FORM})(::Real,
                                            u::VectorField{3, F},
                                          out::VectorField{3, F}) where {FORM, F<:FTField}
    laplacian!(out, u)
    out .*= 1/eq.Re
    _nse_advection!(out, u, eq, FORM())
    eq.force(out, u, Forward())
    return out
end


# -------------- #
# linearised NSE #
# -------------- #
# 3-arg: cache the base-flow quantities needed by FORM, then delegate to 2-arg.
function (eq::CartesianPrimitive3DLNSE{MODE, FORM})(::Real,
                                                   u::VectorField{3, F},
                                                   v::VectorField{3, F},
                                                 out::VectorField{3, F}) where {MODE, FORM, F<:FTField}
    _lnse_setup!(u, eq, FORM())
    eq(0, v, out)
    return out
end

# forward
function (eq::CartesianPrimitive3DLNSE{Forward, FORM})(::Real,
                                                       v::VectorField{3, F},
                                                     out::VectorField{3, F}) where {FORM, F<:FTField}
    laplacian!(out, v)
    out .*= 1/eq.Re
    _fwd_advection!(out, v, eq, FORM())
    eq.force(out, v, Forward())
    return out
end

# continuous adjoint
function (eq::CartesianPrimitive3DLNSE{AdjointContinuous, FORM})(::Real,
                                                                 v::VectorField{3, F},
                                                               out::VectorField{3, F}) where {FORM, F<:FTField}
    laplacian!(out, v)
    out .*= 1/eq.Re
    _adjcont_advection!(out, v, eq, FORM())
    eq.force(out, v, AdjointContinuous())
    return out
end

# discrete adjoint (advective only in Phase A)
function (eq::CartesianPrimitive3DLNSE{AdjointDiscrete, FORM})(::Real,
                                                               v::VectorField{3, F},
                                                             out::VectorField{3, F}) where {FORM, F<:FTField}
    laplacian!(out, v, adjoint=true)
    out .*= 1/eq.Re
    _adjdisc_advection!(out, v, eq, FORM())
    eq.force(out, v, AdjointDiscrete())
    return out
end
