# Generic velocity-only Cartesian primitive NSE / LNSE operators.
#
# `NDIM` is the number of spatial derivative directions. `NCOMP` is the number
# of advected velocity components. This covers the bundled cases:
#
#   NDIM=2, NCOMP=2  -> planar 2D velocity
#   NDIM=2, NCOMP=3  -> 2D-3C velocity, with w passively advected by (u, v)
#   NDIM=3, NCOMP=3  -> fully three-dimensional velocity
#
# The call skeletons live in abstract_nse.jl; form-specific advection helpers
# live in advection.jl. The old public CartesianPrimitive2D/2D3C/3D operator
# names are aliases below.


# ---------------------- #
# generic nonlinear NSE  #
# ---------------------- #
"""
    CartesianPrimitiveNSE{NDIM, NCOMP, FORM, T, FFT, S, P, BF}

Nonlinear velocity-only Cartesian primitive-variable Navier-Stokes operator.

Evaluates `out = Δu/Re − advection(u) + force(out, u, Forward())` in spectral
space. `NDIM` is the number of spatial derivative directions, `NCOMP` is the
number of advected components, and `FORM` selects the advection form.
"""
mutable struct CartesianPrimitiveNSE{NDIM, NCOMP, FORM<:AdvectionForm, T, FFT, S, P, BF} <: AbstractNSE{FORM}
              Re::T
     const plans::FFT
    const scache::Vector{VectorField{NCOMP, S}}
    const pcache::Vector{VectorField{NCOMP, P}}
    const  force::BF

    CartesianPrimitiveNSE{NDIM, NCOMP, FORM}(Re::T,
                                          plans::FFT,
                                         scache::Vector{VectorField{NCOMP, S}},
                                         pcache::Vector{VectorField{NCOMP, P}},
                                          force::BF) where {NDIM, NCOMP, FORM, T, FFT, S, P, BF} =
        new{NDIM, NCOMP, FORM, T, FFT, S, P, BF}(Re, plans, scache, pcache, force)
end

function CartesianPrimitiveNSE{NDIM, NCOMP}(g::G, Re;
                                            form::AdvectionForm=Advective(),
                                            force::BF=NoForce(),
                                            flags=FFTW.EXHAUSTIVE) where {NDIM, NCOMP, T, G<:AbstractGrid{T}, BF}
    return _construct_cartesian_primitive_nse(Val(NDIM), Val(NCOMP), g, Re;
                                              form=form, force=force, flags=flags)
end

CartesianPrimitiveNSE{NDIM, NCOMP}(Re, plans,
                                   scache::Vector{<:VectorField{NCOMP}},
                                   pcache::Vector{<:VectorField{NCOMP}},
                                   force) where {NDIM, NCOMP} =
    CartesianPrimitiveNSE{NDIM, NCOMP, Advective}(Re, plans, scache, pcache, force)


# ---------------------- #
# generic linearised NSE #
# ---------------------- #
"""
    CartesianPrimitiveLNSE{NDIM, NCOMP, MODE, FORM, T, FFT, S, P, BF}

Linearised velocity-only Cartesian primitive-variable Navier-Stokes operator.

The three-argument call `eq(t, u, v, out)` caches the base-flow quantities
needed by `FORM` from `u`, then delegates to the two-argument form
`eq(t, v, out)`. `MODE` selects forward, continuous-adjoint, or discrete-adjoint
action.
"""
mutable struct CartesianPrimitiveLNSE{NDIM, NCOMP, MODE<:Mode, FORM<:AdvectionForm, T, FFT, S, P, BF} <: AbstractLNSE{MODE, FORM}
              Re::T
     const plans::FFT
    const scache::Vector{VectorField{NCOMP, S}}
    const pcache::Vector{VectorField{NCOMP, P}}
    const  force::BF

    CartesianPrimitiveLNSE{NDIM, NCOMP, MODE, FORM}(Re::T,
                                                 plans::FFT,
                                                scache::Vector{VectorField{NCOMP, S}},
                                                pcache::Vector{VectorField{NCOMP, P}},
                                                 force::BF) where {NDIM, NCOMP, MODE, FORM, T, FFT, S, P, BF} =
        new{NDIM, NCOMP, MODE, FORM, T, FFT, S, P, BF}(Re, plans, scache, pcache, force)
end

function CartesianPrimitiveLNSE{NDIM, NCOMP}(g::G, Re;
                                             mode::Mode=AdjointDiscrete(),
                                             form::AdvectionForm=Advective(),
                                             force::BF=NoForce(),
                                             flags=FFTW.EXHAUSTIVE) where {NDIM, NCOMP, T, G<:AbstractGrid{T}, BF}
    return _construct_cartesian_primitive_lnse(Val(NDIM), Val(NCOMP), g, Re;
                                               mode=mode, form=form, force=force, flags=flags)
end

CartesianPrimitiveLNSE{NDIM, NCOMP, MODE}(Re, plans,
                                          scache::Vector{<:VectorField{NCOMP}},
                                          pcache::Vector{<:VectorField{NCOMP}},
                                          force) where {NDIM, NCOMP, MODE} =
    CartesianPrimitiveLNSE{NDIM, NCOMP, MODE, Advective}(Re, plans, scache, pcache, force)


# ---------------------- #
# public compatibility   #
# ---------------------- #
const CartesianPrimitive2DNSE{FORM, T, FFT, S, P, BF} =
    CartesianPrimitiveNSE{2, 2, FORM, T, FFT, S, P, BF}
const CartesianPrimitive2D3CNSE{FORM, T, FFT, S, P, BF} =
    CartesianPrimitiveNSE{2, 3, FORM, T, FFT, S, P, BF}
const CartesianPrimitive3DNSE{FORM, T, FFT, S, P, BF} =
    CartesianPrimitiveNSE{3, 3, FORM, T, FFT, S, P, BF}

const CartesianPrimitive2DLNSE{MODE, FORM, T, FFT, S, P, BF} =
    CartesianPrimitiveLNSE{2, 2, MODE, FORM, T, FFT, S, P, BF}
const CartesianPrimitive2D3CLNSE{MODE, FORM, T, FFT, S, P, BF} =
    CartesianPrimitiveLNSE{2, 3, MODE, FORM, T, FFT, S, P, BF}
const CartesianPrimitive3DLNSE{MODE, FORM, T, FFT, S, P, BF} =
    CartesianPrimitiveLNSE{3, 3, MODE, FORM, T, FFT, S, P, BF}


# Grid constructors for the compatibility aliases above. A bare alias like
# `CartesianPrimitive2DNSE(g, Re)` does not automatically dispatch to the
# partial constructor `CartesianPrimitiveNSE{2,2}(g, Re)`, so keep these small
# forwarding methods for downstream code.
CartesianPrimitive2DNSE(g::AbstractGrid, Re; kwargs...) =
    _construct_cartesian_primitive_nse(Val(2), Val(2), g, Re; kwargs...)
CartesianPrimitive2D3CNSE(g::AbstractGrid, Re; kwargs...) =
    _construct_cartesian_primitive_nse(Val(2), Val(3), g, Re; kwargs...)
CartesianPrimitive3DNSE(g::AbstractGrid, Re; kwargs...) =
    _construct_cartesian_primitive_nse(Val(3), Val(3), g, Re; kwargs...)

CartesianPrimitive2DNSE{FORM}(g::AbstractGrid, Re; force::BF=NoForce(),
                              flags=FFTW.EXHAUSTIVE) where {FORM<:AdvectionForm, BF} =
    _construct_cartesian_primitive_nse(Val(2), Val(2), g, Re;
                                       form=FORM(), force=force, flags=flags)
CartesianPrimitive2D3CNSE{FORM}(g::AbstractGrid, Re; force::BF=NoForce(),
                                flags=FFTW.EXHAUSTIVE) where {FORM<:AdvectionForm, BF} =
    _construct_cartesian_primitive_nse(Val(2), Val(3), g, Re;
                                       form=FORM(), force=force, flags=flags)
CartesianPrimitive3DNSE{FORM}(g::AbstractGrid, Re; force::BF=NoForce(),
                              flags=FFTW.EXHAUSTIVE) where {FORM<:AdvectionForm, BF} =
    _construct_cartesian_primitive_nse(Val(3), Val(3), g, Re;
                                       form=FORM(), force=force, flags=flags)

CartesianPrimitive2DNSE(Re, plans, scache::Vector{<:VectorField{2}},
                        pcache::Vector{<:VectorField{2}}, force) =
    CartesianPrimitiveNSE{2, 2, Advective}(Re, plans, scache, pcache, force)
CartesianPrimitive2D3CNSE(Re, plans, scache::Vector{<:VectorField{3}},
                          pcache::Vector{<:VectorField{3}}, force) =
    CartesianPrimitiveNSE{2, 3, Advective}(Re, plans, scache, pcache, force)
CartesianPrimitive3DNSE(Re, plans, scache::Vector{<:VectorField{3}},
                        pcache::Vector{<:VectorField{3}}, force) =
    CartesianPrimitiveNSE{3, 3, Advective}(Re, plans, scache, pcache, force)

CartesianPrimitive2DLNSE(g::AbstractGrid, Re; kwargs...) =
    _construct_cartesian_primitive_lnse(Val(2), Val(2), g, Re; kwargs...)
CartesianPrimitive2D3CLNSE(g::AbstractGrid, Re; kwargs...) =
    _construct_cartesian_primitive_lnse(Val(2), Val(3), g, Re; kwargs...)
CartesianPrimitive3DLNSE(g::AbstractGrid, Re; kwargs...) =
    _construct_cartesian_primitive_lnse(Val(3), Val(3), g, Re; kwargs...)

CartesianPrimitive2DLNSE{MODE}(g::AbstractGrid, Re; form::AdvectionForm=Advective(),
                               force::BF=NoForce(), flags=FFTW.EXHAUSTIVE) where {MODE<:Mode, BF} =
    _construct_cartesian_primitive_lnse(Val(2), Val(2), g, Re;
                                        mode=MODE(), form=form, force=force, flags=flags)
CartesianPrimitive2D3CLNSE{MODE}(g::AbstractGrid, Re; form::AdvectionForm=Advective(),
                                 force::BF=NoForce(), flags=FFTW.EXHAUSTIVE) where {MODE<:Mode, BF} =
    _construct_cartesian_primitive_lnse(Val(2), Val(3), g, Re;
                                        mode=MODE(), form=form, force=force, flags=flags)
CartesianPrimitive3DLNSE{MODE}(g::AbstractGrid, Re; form::AdvectionForm=Advective(),
                               force::BF=NoForce(), flags=FFTW.EXHAUSTIVE) where {MODE<:Mode, BF} =
    _construct_cartesian_primitive_lnse(Val(3), Val(3), g, Re;
                                        mode=MODE(), form=form, force=force, flags=flags)

CartesianPrimitive2DLNSE{MODE}(Re, plans, scache::Vector{<:VectorField{2}},
                               pcache::Vector{<:VectorField{2}}, force) where {MODE<:Mode} =
    CartesianPrimitiveLNSE{2, 2, MODE, Advective}(Re, plans, scache, pcache, force)
CartesianPrimitive2D3CLNSE{MODE}(Re, plans, scache::Vector{<:VectorField{3}},
                                 pcache::Vector{<:VectorField{3}}, force) where {MODE<:Mode} =
    CartesianPrimitiveLNSE{2, 3, MODE, Advective}(Re, plans, scache, pcache, force)
CartesianPrimitive3DLNSE{MODE}(Re, plans, scache::Vector{<:VectorField{3}},
                               pcache::Vector{<:VectorField{3}}, force) where {MODE<:Mode} =
    CartesianPrimitiveLNSE{3, 3, MODE, Advective}(Re, plans, scache, pcache, force)

CartesianPrimitive2DLNSE{MODE, FORM}(g::AbstractGrid, Re; force::BF=NoForce(),
                                     flags=FFTW.EXHAUSTIVE) where {MODE<:Mode, FORM<:AdvectionForm, BF} =
    _construct_cartesian_primitive_lnse(Val(2), Val(2), g, Re;
                                        mode=MODE(), form=FORM(), force=force, flags=flags)
CartesianPrimitive2D3CLNSE{MODE, FORM}(g::AbstractGrid, Re; force::BF=NoForce(),
                                       flags=FFTW.EXHAUSTIVE) where {MODE<:Mode, FORM<:AdvectionForm, BF} =
    _construct_cartesian_primitive_lnse(Val(2), Val(3), g, Re;
                                        mode=MODE(), form=FORM(), force=force, flags=flags)
CartesianPrimitive3DLNSE{MODE, FORM}(g::AbstractGrid, Re; force::BF=NoForce(),
                                     flags=FFTW.EXHAUSTIVE) where {MODE<:Mode, FORM<:AdvectionForm, BF} =
    _construct_cartesian_primitive_lnse(Val(3), Val(3), g, Re;
                                        mode=MODE(), form=FORM(), force=force, flags=flags)


# ---------------------- #
# construction utilities #
# ---------------------- #
function _construct_cartesian_primitive_nse(::Val{NDIM}, ::Val{NCOMP}, g::G, Re;
                                            form::AdvectionForm=Advective(),
                                            force::BF=NoForce(),
                                            flags=FFTW.EXHAUSTIVE) where {NDIM, NCOMP, T, G<:AbstractGrid{T}, BF}
    ((NDIM == 2 && (NCOMP == 2 || NCOMP == 3)) || (NDIM == 3 && NCOMP == 3)) ||
        throw(ArgumentError("unsupported Cartesian primitive dimensions: NDIM=$NDIM, NCOMP=$NCOMP"))

    plans = FFTPlans(g, flags=flags)
    scache, pcache = alloc_caches(g, NCOMP, NDIM, NDIM + 1)
    return CartesianPrimitiveNSE{NDIM, NCOMP, typeof(form)}(T(Re), plans, scache, pcache, force)
end

function _construct_cartesian_primitive_lnse(::Val{NDIM}, ::Val{NCOMP}, g::G, Re;
                                             mode::Mode=AdjointDiscrete(),
                                             form::AdvectionForm=Advective(),
                                             force::BF=NoForce(),
                                             flags=FFTW.EXHAUSTIVE) where {NDIM, NCOMP, T, G<:AbstractGrid{T}, BF}
    ((NDIM == 2 && (NCOMP == 2 || NCOMP == 3)) || (NDIM == 3 && NCOMP == 3)) ||
        throw(ArgumentError("unsupported Cartesian primitive dimensions: NDIM=$NDIM, NCOMP=$NCOMP"))

    plans = FFTPlans(g, flags=flags)
    scache, pcache = alloc_caches(g, NCOMP, NDIM + 1, 2 * (NDIM + 1))
    return CartesianPrimitiveLNSE{NDIM, NCOMP, typeof(mode), typeof(form)}(T(Re), plans, scache, pcache, force)
end
