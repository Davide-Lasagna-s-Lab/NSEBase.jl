# NSE formulation tag types and the `construct_equations` factory function.
#
# Each formulation is a plain singleton struct (no fields) that acts as a
# compile-time tag, selecting which operator types and cache sizes to use.
# Downstream packages can add new formulations (polar, cylindrical, …) by
# defining a new struct and implementing the four interface methods below.
#
# `construct_equations` is the single entry point for building a `ProjectedNSE`
# object: it allocates shared cache pools, creates FFTW plans, and wires
# everything together without the caller needing to know the internal operator
# types.


# -------------------------- #
# Navier-Stokes formulations #
# -------------------------- #
"""
    CartesianPrimitive3D

Tag selecting the three-component (u, v, w) Cartesian primitive-variable NSE
formulation.  Pass `CartesianPrimitive3D()` to [`construct_equations`](@ref) to
build [`CartesianPrimitive3DNSE`](@ref) and [`CartesianPrimitive3DLNSE`](@ref)
operators with three velocity components, four spectral-space caches, and eight
physical-space caches.
"""
struct CartesianPrimitive3D end

"""
    CartesianPrimitive2D

Tag selecting the two-component (u, v) planar Cartesian primitive-variable NSE
formulation.  Pass `CartesianPrimitive2D()` to [`construct_equations`](@ref) to
build [`CartesianPrimitive2DNSE`](@ref) and [`CartesianPrimitive2DLNSE`](@ref)
operators with two velocity components, three spectral-space caches, and six
physical-space caches.
"""
struct CartesianPrimitive2D end

"""
    CartesianPrimitive2D3C

Tag selecting the 2D-3C (u, v, w) Cartesian primitive-variable NSE
formulation.  Pass `CartesianPrimitive2D3C()` to [`construct_equations`](@ref)
to build [`CartesianPrimitive2D3CNSE`](@ref) and [`CartesianPrimitive2D3CLNSE`](@ref)
operators with three velocity components on a two-dimensional spatial grid.
The in-plane pair (u, v) satisfies the standard incompressible 2D NSE; the
out-of-plane component w is advected by (u, v) without contributing to the
pressure or to its own advection.
"""
struct CartesianPrimitive2D3C end

"""
    PolarPrimitive

Tag reserved for a polar (cylindrical) primitive-variable NSE formulation.
Not yet implemented — all interface methods throw an error.
"""
struct PolarPrimitive end


# ------------------------------------------------------------------ #
# Formulation interface                                               #
# ------------------------------------------------------------------ #
# New formulations must implement all four methods below for their tag
# type.  The `CartesianPrimitive3D` and `CartesianPrimitive2D` methods
# are defined immediately after these stubs and serve as canonical examples.

"""
    ncomp(formulation) -> Int

Return the number of velocity components for `formulation` (e.g. 3 for a
three-component flow, 2 for a planar flow).  Used by [`construct_equations`](@ref)
to size the cache `VectorField`s.
"""
function ncomp end

"""
    cache_length(formulation, ::Type{<:FTField}) -> Int
    cache_length(formulation, ::Type{<:Field})   -> Int

Return the number of pre-allocated `VectorField` scratch arrays needed by the
spectral and physical cache pools for `formulation`.  Used by
[`construct_equations`](@ref) to allocate `scache` and `pcache`.
"""
function cache_length end

"""
    nonlinear_operator(formulation) -> Type

Return the concrete nonlinear NSE operator type for `formulation`.
[`construct_equations`](@ref) calls this to instantiate the nonlinear operator.
"""
function nonlinear_operator end

"""
    linearised_operator(formulation, mode) -> Type

Return the concrete linearised NSE operator type for `formulation` and `mode`.
The returned type is parametric on `mode` (e.g. `CartesianPrimitive3DLNSE{M}`).
[`construct_equations`](@ref) calls this to instantiate the linearised operator.
"""
function linearised_operator end

# CartesianPrimitive3D
ncomp(              ::CartesianPrimitive3D)                    = 3
cache_length(       ::CartesianPrimitive3D, ::Type{<:FTField}) = 4
cache_length(       ::CartesianPrimitive3D, ::Type{<:Field})   = 8
nonlinear_operator( ::CartesianPrimitive3D)                    = CartesianPrimitive3DNSE
linearised_operator(::CartesianPrimitive3D, ::M) where {M}     = CartesianPrimitive3DLNSE{M}

# CartesianPrimitive2D
ncomp(              ::CartesianPrimitive2D)                    = 2
cache_length(       ::CartesianPrimitive2D, ::Type{<:FTField}) = 3
cache_length(       ::CartesianPrimitive2D, ::Type{<:Field})   = 6
nonlinear_operator( ::CartesianPrimitive2D)                    = CartesianPrimitive2DNSE
linearised_operator(::CartesianPrimitive2D, ::M) where {M}     = CartesianPrimitive2DLNSE{M}

# CartesianPrimitive2D3C
ncomp(              ::CartesianPrimitive2D3C)                    = 3
cache_length(       ::CartesianPrimitive2D3C, ::Type{<:FTField}) = 3
cache_length(       ::CartesianPrimitive2D3C, ::Type{<:Field})   = 6
nonlinear_operator( ::CartesianPrimitive2D3C)                    = CartesianPrimitive2D3CNSE
linearised_operator(::CartesianPrimitive2D3C, ::M) where {M}     = CartesianPrimitive2D3CLNSE{M}

# PolarPrimitive (not yet implemented)
ncomp(              ::PolarPrimitive)                    = throw(error("polar primitive Navier-Stokes formulation has not been implemented"))
cache_length(       ::PolarPrimitive, ::Type{<:FTField}) = throw(error("polar primitive Navier-Stokes formulation has not been implemented"))
cache_length(       ::PolarPrimitive, ::Type{<:Field})   = throw(error("polar primitive Navier-Stokes formulation has not been implemented"))
nonlinear_operator( ::PolarPrimitive)                    = throw(error("polar primitive Navier-Stokes formulation has not been implemented"))
linearised_operator(::PolarPrimitive, ::M) where {M}     = throw(error("polar primitive Navier-Stokes formulation has not been implemented"))


"""
    construct_equations(grid::AbstractGrid, Re, base, formulation=CartesianPrimitive3D();
                        force=NoForce(), mode=AdjointDiscrete(),
                        form=Advective(), flags=FFTW.EXHAUSTIVE,
                        dealias=true)

Construct a [`ProjectedNSE`](@ref) from a grid, Reynolds number, and NSE
formulation tag, pre-allocating all operator caches and FFTW plans.

# Arguments
- `grid`: computational grid defining the domain geometry and spectral structure.
- `Re`: Reynolds number.
- `base`: laminar base flow as a tuple with one entry per velocity component.
  Each entry is either a vector of values at the inhomogeneous grid points or
  `nothing` when that component has no base flow (e.g. `(U, nothing, nothing)`
  for streamwise-only Couette/Poiseuille flow).
- `formulation`: NSE formulation tag, defaulting to `CartesianPrimitive3D()`.

# Keyword arguments
- `force`: body forcing callable with signature `(out, u, mode)`.
  Defaults to [`NoForce`](@ref).
- `mode`: adjoint mode; must be [`AdjointDiscrete`](@ref) or
  [`AdjointContinuous`](@ref). Defaults to `AdjointDiscrete()`.
- `form`: advection form tag. Defaults to [`Advective`](@ref). The bundled
  velocity-only Cartesian formulations support [`Advective`](@ref),
  [`Divergence`](@ref), and [`Rotational`](@ref); unsupported formulations
  accept only the default advective form.
- `flags`: FFTW planner flags, e.g. `FFTW.MEASURE` or `FFTW.ESTIMATE` to
  reduce plan-construction time during development. Defaults to `FFTW.EXHAUSTIVE`.
- `dealias`: if `true`, physical-space caches use the 3/2-rule padded grid.
  Defaults to `true`.

# Returns
A [`ProjectedNSE`](@ref) bundling the nonlinear operator, linearised operator,
and base flow, ready for use in optimisation or time-stepping routines.

# Cache layout
Two shared scratch pools are allocated (and reused by both operators):

- `scache`: `cache_length(formulation, FTField)` spectral-space
  [`VectorField`](@ref)s on the undealiased grid.
- `pcache`: `cache_length(formulation, Field)` physical-space
  [`VectorField`](@ref)s on the (optionally dealiased) grid.

# Example
```julia
grid = ChannelGrid(y, Nx, Nz, Nt, α, β, D₁, D₂, ws)
obj  = construct_equations(grid, 1000.0, (U, nothing, nothing); flags=FFTW.MEASURE)
```
"""
function construct_equations(grid::AbstractGrid{T},
                               Re,
                             base,
                      formulation=CartesianPrimitive3D();
                            force=NoForce(),
                             mode=AdjointDiscrete(),
                             form::AdvectionForm=Advective(),
                            flags=FFTW.EXHAUSTIVE,
                          dealias=true) where {T}
    mode isa Union{AdjointContinuous, AdjointDiscrete} || throw(ArgumentError("linearised operator has to operate in adjoint mode"))
    plans          = FFTPlans(grid; flags=flags)
    scache, pcache = alloc_caches(grid, ncomp(formulation),
                                  cache_length(formulation, FTField),
                                  cache_length(formulation, Field);
                                  dealias=dealias)
    nl = _construct_nonlinear_operator(formulation, typeof(form), T(Re), plans, scache, pcache, force)
    ln = _construct_linearised_operator(formulation, mode, typeof(form), T(Re), plans, scache, pcache, force)
    return ProjectedNSE(grid, ncomp(formulation), nl, ln, base)
end

function _construct_nonlinear_operator(formulation, ::Type{FORM}, Re, plans, scache, pcache, force) where {FORM<:AdvectionForm}
    FORM === Advective && return nonlinear_operator(formulation)(Re, plans, scache, pcache, force)
    throw(ArgumentError("$(typeof(formulation)) does not support advection form $(FORM)"))
end

function _construct_linearised_operator(formulation, mode, ::Type{FORM}, Re, plans, scache, pcache, force) where {FORM<:AdvectionForm}
    FORM === Advective && return linearised_operator(formulation, mode)(Re, plans, scache, pcache, force)
    throw(ArgumentError("$(typeof(formulation)) does not support advection form $(FORM)"))
end

_construct_nonlinear_operator( ::CartesianPrimitive3D, ::Type{FORM}, Re, plans, scache, pcache, force) where {FORM<:AdvectionForm} =
    CartesianPrimitive3DNSE{FORM}(Re, plans, scache, pcache, force)
_construct_linearised_operator(::CartesianPrimitive3D, mode, ::Type{FORM}, Re, plans, scache, pcache, force) where {FORM<:AdvectionForm} =
    CartesianPrimitive3DLNSE{typeof(mode), FORM}(Re, plans, scache, pcache, force)

_construct_nonlinear_operator( ::CartesianPrimitive2D, ::Type{FORM}, Re, plans, scache, pcache, force) where {FORM<:AdvectionForm} =
    CartesianPrimitive2DNSE{FORM}(Re, plans, scache, pcache, force)
_construct_linearised_operator(::CartesianPrimitive2D, mode, ::Type{FORM}, Re, plans, scache, pcache, force) where {FORM<:AdvectionForm} =
    CartesianPrimitive2DLNSE{typeof(mode), FORM}(Re, plans, scache, pcache, force)

_construct_nonlinear_operator( ::CartesianPrimitive2D3C, ::Type{FORM}, Re, plans, scache, pcache, force) where {FORM<:AdvectionForm} =
    CartesianPrimitive2D3CNSE{FORM}(Re, plans, scache, pcache, force)
_construct_linearised_operator(::CartesianPrimitive2D3C, mode, ::Type{FORM}, Re, plans, scache, pcache, force) where {FORM<:AdvectionForm} =
    CartesianPrimitive2D3CLNSE{typeof(mode), FORM}(Re, plans, scache, pcache, force)
