# NSE formulation tag types and the `construct_equations` factory function.
#
# The velocity-only formulations are singleton tags that select operator types
# and cache sizes through the four interface methods below. Parameterised
# formulations such as the 2D and 3D Boussinesq tags carry physical
# coefficients and specialise `construct_equations` directly.
#
# `construct_equations` is the single entry point for building a `ProjectedNSE`
# object: it allocates shared cache pools, creates FFTW plans, and wires
# everything together without the caller needing to know the internal operator
# types.


# -------------------------- #
# Navier-Stokes formulations #
# -------------------------- #
@doc raw"""
    CartesianPrimitive3D

Tag selecting the three-component (u, v, w) Cartesian primitive-variable NSE
formulation.  Pass `CartesianPrimitive3D()` to [`construct_equations`](@ref) to
build [`CartesianPrimitive3DNSE`](@ref) and [`CartesianPrimitive3DLNSE`](@ref)
operators with three velocity components, four spectral-space caches, and eight
physical-space caches.

For ``\boldsymbol u=(u,v,w)`` and ``\nabla=(\partial_x,\partial_y,\partial_z)``,
the nonlinear spatial residual is

```math
\mathcal N_i(\boldsymbol u)=\frac{1}{Re}\nabla^2u_i
-\sum_{j=1}^3u_j\partial_ju_i+\mathcal F_i(\boldsymbol u),
\qquad i=1,2,3.
```

The raw primitive operator does not compute pressure. In a
[`ProjectedNSE`](@ref), expansion into an admissible divergence-free basis and
projection of this residual remove the pressure contribution.
"""
struct CartesianPrimitive3D end

@doc raw"""
    CartesianPrimitive2D

Tag selecting the two-component (u, v) planar Cartesian primitive-variable NSE
formulation.  Pass `CartesianPrimitive2D()` to [`construct_equations`](@ref) to
build [`CartesianPrimitive2DNSE`](@ref) and [`CartesianPrimitive2DLNSE`](@ref)
operators with two velocity components, three spectral-space caches, and six
physical-space caches.

For ``\boldsymbol u=(u,v)`` and ``\nabla=(\partial_x,\partial_y)``, the
implemented residual is

```math
\mathcal N_i(\boldsymbol u)=\frac{1}{Re}\nabla^2u_i
-\sum_{j=1}^2u_j\partial_ju_i+\mathcal F_i(\boldsymbol u),
\qquad i=1,2.
```

Pressure is omitted from the raw operator and is eliminated by the admissible
basis when the equation is wrapped in [`ProjectedNSE`](@ref).
"""
struct CartesianPrimitive2D end

@doc raw"""
    CartesianPrimitive2D3C

Tag selecting the streamwise-invariant Cartesian primitive-variable formulation with three
velocity components. Pass `CartesianPrimitive2D3C()` to [`construct_equations`](@ref) to build
[`CartesianPrimitive2D3CNSE`](@ref) and [`CartesianPrimitive2D3CLNSE`](@ref).

Physical `x` must be absent and the active plane is always `(y,z)`. The state is `(v,w,u)`: the
wall-normal and spanwise velocities satisfy the incompressible two-dimensional equations, while
the streamwise velocity is advected without contributing to pressure or advection. Ordinary
two-dimensional `(x,y)` channels use [`CartesianPrimitive2D`](@ref), not this formulation.

Writing the state as ``\boldsymbol q=(v,w,u)`` and
``\nabla_{yz}=(\partial_y,\partial_z)``, the three residual equations are

```math
\begin{aligned}
\mathcal N_v &= Re^{-1}\nabla_{yz}^2v-v\partial_yv-w\partial_zv+\mathcal F_v,\\
\mathcal N_w &= Re^{-1}\nabla_{yz}^2w-v\partial_yw-w\partial_zw+\mathcal F_w,\\
\mathcal N_u &= Re^{-1}\nabla_{yz}^2u-v\partial_yu-w\partial_zu+\mathcal F_u.
\end{aligned}
```

Thus ``u`` is advected but never acts as an advecting velocity. Pressure is
absent from the raw residual and is handled by projection of the in-plane
``(v,w)`` dynamics.
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
# Formulations using the generic factory implement all four methods below for
# their tag type. `CartesianPrimitive3D` and `CartesianPrimitive2D` are the
# canonical examples. A formulation whose operator constructor needs extra
# coefficients may instead provide a specialised `construct_equations` method.

"""
    ncomp(formulation) -> Int

Return the number of state components for a formulation handled by the generic
factory (e.g. 3 for a three-component velocity flow, 2 for a planar flow).
Used by [`construct_equations`](@ref) to size cache `VectorField`s.
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


@doc raw"""
    construct_equations(grid::AbstractGrid, Re, base, formulation=CartesianPrimitive3D();
                        force=NoForce(), mode=AdjointDiscrete(), flags=FFTW.EXHAUSTIVE,
                        dealias=true)

Construct a [`ProjectedNSE`](@ref) from a grid, Reynolds number, and NSE
formulation tag, pre-allocating all operator caches and FFTW plans.

# Arguments
- `grid`: computational grid defining the domain geometry and spectral structure.
- `Re`: Reynolds number.
- `base`: laminar base state as a tuple with one entry per state component.
  Each entry is either an array on the inhomogeneous grid or `nothing` when
  that component has no base flow: a vector for a channel profile, a matrix for
  a duct profile, and so on. For example, `(U, nothing, nothing)` represents a
  streamwise-only Couette or Poiseuille base flow. Parameterised Boussinesq
  overloads append temperature as their final state component.
- `formulation`: NSE formulation tag, defaulting to `CartesianPrimitive3D()`.

# Keyword arguments
- `force`: body forcing callable with signature `(out, u, mode)`.
  Defaults to [`NoForce`](@ref).
- `mode`: adjoint mode; must be [`AdjointDiscrete`](@ref) or
  [`AdjointContinuous`](@ref). Defaults to `AdjointDiscrete()`.
- `flags`: FFTW planner flags, e.g. `FFTW.MEASURE` or `FFTW.ESTIMATE` to
  reduce plan-construction time during development. Defaults to `FFTW.EXHAUSTIVE`.
- `dealias`: if `true`, physical-space caches use the 3/2-rule padded grid.
  Defaults to `true`.

# Returns
A [`ProjectedNSE`](@ref) bundling the nonlinear operator, linearised operator,
and base flow, ready for use in optimisation or time-stepping routines.

# Projected equations

Let ``E`` denote [`expand!`](@ref), ``P`` denote [`project!`](@ref), and
``\boldsymbol U_b`` denote the inhomogeneous base tuple inserted by
[`add_base_flow!`](@ref). For modal coefficients ``a``, the nonlinear call
computes

```math
R(a)=P\,\mathcal N(Ea+\boldsymbol U_b).
```

The corresponding reduced forward map, whose adjoint is requested from this
adjoint-oriented factory, is

```math
J(a)b=P\,L_{Ea+\boldsymbol U_b}\,Eb
```

After the nonlinear call has cached ``Ea+\boldsymbol U_b``, the returned
linearised call computes ``P L^\dagger_{h,Ea+\boldsymbol U_b}Eb`` for
`AdjointDiscrete()`, or the analogous discretised continuous-adjoint action
for `AdjointContinuous()`. Since ``P=E^\dagger`` under the full and projected
discrete inner products, the discrete expression is the adjoint of the
reduced forward map whenever the force policy implements matching linear
forward and discrete-adjoint actions. See [`AdjointDiscrete`](@ref) for the
complete derivation.

# Cache layout
Two shared scratch pools are allocated (and reused by both operators):

- `scache`: `cache_length(formulation, FTField)` spectral-space
  [`VectorField`](@ref)s on the undealiased grid.
- `pcache`: `cache_length(formulation, Field)` physical-space
  [`VectorField`](@ref)s on the (optionally dealiased) grid.

# Example
```julia
grid = ChannelGrid(y, Nx, Nz, Nt, α, β, D₁, D₂, D₁⁺, D₂⁺, ws)
obj  = construct_equations(grid, 1000.0, (U, nothing, nothing); flags=FFTW.MEASURE)
```
"""
function construct_equations(grid::AbstractGrid{T},
                               Re,
                             base,
                      formulation=CartesianPrimitive3D();
                            force=NoForce(),
                             mode=AdjointDiscrete(),
                            flags=FFTW.EXHAUSTIVE,
                          dealias=true) where {T}
    mode isa Union{AdjointContinuous, AdjointDiscrete} || throw(ArgumentError("linearised operator has to operate in adjoint mode"))
    plans = FFTPlans(grid; flags=flags, dealias)
    scache = [VectorField([FTField(grid)                  for _ in 1:ncomp(formulation)]...) for _ in 1:cache_length(formulation, FTField)]
    pcache = [VectorField([  Field(grid; dealias=dealias) for _ in 1:ncomp(formulation)]...) for _ in 1:cache_length(formulation, Field)]
    nl = nonlinear_operator(formulation)(T(Re), plans, scache, pcache, force)
    ln = linearised_operator(formulation, mode)(T(Re), plans, scache, pcache, force)
    return ProjectedNSE(grid, ncomp(formulation), nl, ln, base)
end
