# Shared interface for different formulations of the Navier-Stokes equations
# that allows the easier construction of ProjectedNSE operators.


# -------------------------- #
# Navier-Stokes formulations #
# -------------------------- #
abstract type                NSEFormulation end
struct CartesianPrimitive <: NSEFormulation end
struct PolarPrimitive     <: NSEFormulation end

ncomp(              ::CartesianPrimitive)                    = 3
cache_length(       ::CartesianPrimitive, ::Type{<:FTField}) = 4
cache_length(       ::CartesianPrimitive, ::Type{<:Field})   = 8
nonlinear_operator( ::CartesianPrimitive)                    = CartesianPrimitiveNSE
linearised_operator(::CartesianPrimitive, ::M) where {M}     = CartesianPrimitiveLNSE{M}

ncomp(              ::PolarPrimitive)                    = throw(error("polar primitive Navier-Stokes formulation has not been implemented"))
cache_length(       ::PolarPrimitive, ::Type{<:FTField}) = throw(error("polar primitive Navier-Stokes formulation has not been implemented"))
cache_length(       ::PolarPrimitive, ::Type{<:Field})   = throw(error("polar primitive Navier-Stokes formulation has not been implemented"))
nonlinear_operator( ::PolarPrimitive)                    = throw(error("polar primitive Navier-Stokes formulation has not been implemented"))
linearised_operator(::PolarPrimitive, ::M) where {M}     = throw(error("polar primitive Navier-Stokes formulation has not been implemented"))


"""
    construct_equations(grid::AbstractGrid, Re, formulation::NSEFormulation=CartesianPrimitive(),
                        base; force=NoForce(), mode=AdjointDiscrete(), flags=FFTW.EXHAUSTIVE,
                        dealias=true)

Construct a [`ProjectedNSE`](@ref) objective from a grid, Reynolds number, and NSE
formulation, pre-allocating all operator caches and FFTW plans.

# Arguments
- `grid`: computational grid defining the domain geometry and spectral structure.
- `Re`: Reynolds number.
- `formulation`: NSE formulation determining the state-space representation and
  operator structure. Defaults to [`CartesianPrimitive`](@ref).
- `base`: base flow used to recover the flow field after [`expand!`](@ref) call.

# Keyword arguments
- `force`: body forcing term. Defaults to [`NoForce`](@ref).
- `mode`: adjoint mode controlling whether the linearised operator uses a
  continuous or discrete adjoint. Defaults to [`AdjointDiscrete`](@ref).
- `flags`: FFTW planner flags passed to [`FFTPlans`](@ref). Defaults to
  `FFTW.EXHAUSTIVE`; use `FFTW.MEASURE` or `FFTW.ESTIMATE` to reduce
  plan construction time during development.
- `dealias`: if `true`, physical-space caches are allocated on the dealiased
  grid. Defaults to `true`.

# Returns
A [`ProjectedNSE`](@ref) bundling the nonlinear operator, linearised operator,
and base flow, ready for use in optimisation or time-stepping routines.

# Cache layout
Two cache pools are allocated and shared between the nonlinear and linearised
operators:

- `scache`: `cache_length(formulation, FTField)` spectral-space
  [`VectorField`](@ref)s, each with `ncomp(formulation)` [`FTField`](@ref)
  components on the undealiased grid.
- `pcache`: `cache_length(formulation, Field)` physical-space
  [`VectorField`](@ref)s, each with `ncomp(formulation)` [`FTField`](@ref)
  components on the (optionally dealiased) grid.

# Example
```julia
grid  = MyChannelGrid(Ny=64, Nx=32, Nz=32)
base  = read_base_flow(grid)
obj   = construct_equations(grid, 1000.0, CartesianPrimitive(), base;
                            force=ConstantForce(grid),
                            mode=AdjointDiscrete(),
                            flags=FFTW.MEASURE)
```
"""
function construct_equations(grid::AbstractGrid{T},
                               Re,
                             base,
                      formulation::NSEFormulation=CartesianPrimitive();
                            force=NoForce(),
                             mode=AdjointDiscrete(),
                            flags=FFTW.EXHAUSTIVE,
                          dealias=true) where {T}
    mode isa Union{AdjointContinuous, AdjointDiscrete} || throw(ArgumentError("linearised operator has to operate in adjoint mode"))
    plans = FFTPlans(grid; flags=flags)
    scache = [VectorField([FTField(grid)                  for _ in 1:ncomp(formulation)]...) for _ in 1:cache_length(formulation, FTField)]
    pcache = [VectorField([  Field(grid; dealias=dealias) for _ in 1:ncomp(formulation)]...) for _ in 1:cache_length(formulation, Field)]
    nl = nonlinear_operator(formulation)(T(Re), plans, scache, pcache, force)
    ln = linearised_operator(formulation, mode)(T(Re), plans, scache, pcache, force)
    return ProjectedNSE(grid, ncomp(formulation), nl, ln, base)
end
