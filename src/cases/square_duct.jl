# //////////////////////////////////////////////////////////////////////////// #
# ///              Square-duct grids and pressure-driven flow              /// #
# //////////////////////////////////////////////////////////////////////////// #
#
# This file defines the complete square-duct case: its layout constants and contracts, the bundled
# `SquareDuctGrid` factory, and the pressure-driven `SquareDuctFlow` equation constructor.
#
# Physical case and equations
#
# The domain has bounded cross-section `(x,y)∈[0,1]²`, periodic streamwise direction
# `z∈[0,2π/α)`, and a unit-period time or phase coordinate `t`; `Nt=1` represents no dependence
# on that coordinate. The Cartesian velocity state is `(u,v,w)=(u_x,u_y,u_z)`, so component three
# `w` is streamwise. `SquareDuctFlow` constructs the three-dimensional primitive equations with
# viscous coefficient `1/Re`.
#
# Its default base `(nothing,nothing,nothing)` means that no reference flow is prescribed. To
# linearise about a laminar duct solution, place its cross-section values in the third entry of
# `base=(U,V,W)`. The pressure-gradient amplitude `f` is a spatially constant force on component
# three at the zero `(z,t)` Fourier mode and every cross-section point. Nondimensionalisation remains
# the caller's choice; with the documented friction scaling, `Re=Reτ` and `f=1` is conventional.
#
# Numerical layout
#
# `SquareDuctGrid` is a `RectangularGrid{2}` stored as `(x,y,z,t)`. Both bounded directions use one
# deliberately shared FDGrids discretisation: points, first and second derivatives, quadrature
# weights, and weighted adjoints have identical object identity in `x` and `y`. The cached product
# quadrature satisfies `weights(g)[i,j] = g.ws[1][i] * g.ws[2][j]`, while derivatives remain
# one-dimensional matrix products. Streamwise `z` is the real-to-complex transform with wavenumbers
# `kα`; `t` is a complex transform with wavenumbers `2πn`. Dealiasing changes only Fourier sizes.
#
# Boundary conditions
#
# The grid owns geometry, quadrature, and differentiation operators, not boundary conditions. The
# default Gauss–Lobatto distribution includes the four walls. The caller must supply any base profile
# with the required wall values; the basis or residual formulation imposes the perturbation or full
# residual boundary conditions appropriate to the chosen representation.
#
# Typical use
#
#     grid = SquareDuctGrid(49, 63, 1, 0.5; width=7)
#     equations = SquareDuctFlow(grid, 2000; f=1, fftw_flags=FFTW.ESTIMATE)
#
# To supply a streamwise reference profile:
#
#     x, y = grid.xs
#     W = (x .* (1 .- x)) * transpose(y .* (1 .- y))
#     equations = SquareDuctFlow(grid, 2000; base=(nothing, nothing, W), f=1,
#                                fftw_flags=FFTW.ESTIMATE)
#
# Extending the case
#
# The `AbstractSquareDuctGrid` alias is the exact numerical-layout contract used by the flow
# constructor. A compatible serial or distributed grid must subtype
# `AbstractGrid{T,4,SQUARE_DUCT_AXES,SQUARE_DUCT_FFT_ORDER}` and implement its interface. Equal side
# lengths and shared operator identity are factory guarantees, not abstract-contract requirements. A
# related duct case can retain the layout while changing its base or forcing policy:
#
#     function MyDuctFlow(g::AbstractSquareDuctGrid, Re;
#                         base=(nothing, nothing, nothing), force=NoForce(),
#                         mode=AdjointDiscrete(), fftw_flags=FFTW.EXHAUSTIVE, dealias=true)
#         return construct_equations(g, Re, base, CartesianPrimitive3D(); force, mode,
#                                    flags=fftw_flags, dealias)
#     end
#
# //////////////////////////////////////////////////////////////////////////// #
# ///                     Square-duct layout constants                     /// #
# //////////////////////////////////////////////////////////////////////////// #

"""
    SQUARE_DUCT_AXES = (1, 2, 3, 4)

Map physical coordinates `(x, y, z, t)` to square-duct storage dimensions.
Physical and storage order coincide: dimensions 1 and 2 form the bounded
cross-section, dimension 3 is the periodic streamwise direction, and dimension
4 is the optional time or phase direction.

Keeping the two finite-difference directions first makes their matrix products
act directly along the leading array dimensions. See also
[`SQUARE_DUCT_FFT_ORDER`](@ref) and [`SQUARE_DUCT_INHOMOGENEOUS_DIMS`](@ref).
"""
const SQUARE_DUCT_AXES = (1, 2, 3, 4)

"""
    SQUARE_DUCT_FFT_ORDER = (3, 4)

Square-duct Fourier storage dimensions in transform order. Streamwise `z`
occupies dimension 3 and uses the real-to-complex transform; unit-period time
occupies dimension 4 and uses a complex transform.
"""
const SQUARE_DUCT_FFT_ORDER = (3, 4)

"""
    SQUARE_DUCT_INHOMOGENEOUS_DIMS = (1, 2)

The two square-duct storage dimensions discretised with FDGrids, corresponding
to the bounded cross-section coordinates `(x, y)`.
"""
const SQUARE_DUCT_INHOMOGENEOUS_DIMS = (1, 2)

# //////////////////////////////////////////////////////////////////////////// #
# ///              Square-duct layout contract and type alias              /// #
# //////////////////////////////////////////////////////////////////////////// #

"""
    AbstractSquareDuctGrid{T}

Alias for any four-dimensional [`AbstractGrid`](@ref) with real scalar type
`T`, storage order `(x, y, z, t)`, and Fourier directions `(z, t)`.

This is a numerical layout contract, not a separate grid implementation.
[`SquareDuctGrid`](@ref) is the bundled serial implementation; MPI-decomposed
and downstream grids can preserve the same contract and therefore use
[`SquareDuctFlow`](@ref). The abstract alias cannot encode equal cross-section
sizes or shared operator identity, which are guarantees of the bundled
constructor rather than requirements of the differential operators.
"""
const AbstractSquareDuctGrid{T} = AbstractGrid{T, 4, SQUARE_DUCT_AXES, SQUARE_DUCT_FFT_ORDER}

"""
    SquareDuctGrid

Case alias for a four-dimensional [`RectangularGrid{2}`](@ref) stored as
`(x, y, z, t)`. The constructor uses one FDGrids discretisation for both sides
of the square cross-section, so corresponding tuple entries deliberately share
object identity:

```julia
g.xs[1] === g.xs[2]
g.ws[1] === g.ws[2]
g.D₁[1] === g.D₁[2]
g.D₂[1] === g.D₂[2]
g.D₁⁺[1] === g.D₁⁺[2]
g.D₂⁺[1] === g.D₂⁺[2]
```

Like the other case aliases, `SquareDuctGrid` identifies a numerical layout;
the type system cannot express equal cross-section sizes or shared object
identity. Those stronger square-grid guarantees belong to the bundled
[`SquareDuctGrid`](@ref) constructor. A manually assembled `RectangularGrid`
with the same axis map and FFT order can therefore satisfy this alias without
satisfying those constructor guarantees.

The stored size is `(N, N, Nz, Nt)`. [`weights`](@ref) is the tensor product
`g.ws[1] * transpose(g.ws[2])`, while `g.scales == (T(α), T(2π))` stores
the streamwise and unit-period temporal wavenumber scales.
"""
const SquareDuctGrid{T, S, XS, D1, D2, A1, A2, WS, WP} =
    RectangularGrid{2, T, S, 4, SQUARE_DUCT_AXES, SQUARE_DUCT_FFT_ORDER, XS, D1, D2, A1, A2, WS, WP, 2}

# //////////////////////////////////////////////////////////////////////////// #
# ///                       Square-duct grid factory                       /// #
# //////////////////////////////////////////////////////////////////////////// #

"""
    SquareDuctGrid(N, Nz, Nt, α; dist=FDGrids.GaussLobattoGrid(), width=5, T=Float64)
    SquareDuctGrid(N, width, Nz, Nt, α; kwargs...)

Construct the grid for pressure-driven flow through a square
`[0,1] × [0,1]` cross-section. `N` is the number of collocation points in
each bounded direction, `Nz` and `Nt` are the pre-dealiasing streamwise and
temporal Fourier resolutions, and `α=2π/Lz` is the streamwise wavenumber
scale. The second signature preserves the historical ReSolver-SquareDuct.jl
positional stencil width; new code should use the first signature and the
`width` keyword.

# Keywords

- `dist`: FDGrids distribution used in both cross-section directions. The
  default Gauss–Lobatto grid includes both walls and supplies Clenshaw–Curtis
  quadrature weights.
- `width`: stencil width for first- and second-derivative matrices. It must be
  odd and at least three, with `N > 2width` for the compact weighted adjoints.
- `T`: real scalar type used by points, weights, operators, scales, and fields.

`Nz` and `Nt` must be positive and odd. The constructor supplies geometry,
quadrature, derivatives, and weighted adjoints but does not impose no-slip wall
values; boundary conditions belong to the basis or residual formulation.

# Example

```julia
g = SquareDuctGrid(49, 63, 1, 0.5; width=7)
```
"""
function SquareDuctGrid(N::Int, Nz::Int, Nt::Int, α::Real;
                        dist=FDGrids.GaussLobattoGrid(), width=5, T::Type{<:Real}=Float64)
    grid = FDGrids.grid(N, 0, 1, dist)
    xs, ws = Vector{T}(grid.xs), Vector{T}(grid.ws)
    D₁ = FDGrids.DiffMatrix(xs, width, 1; eltype=T)
    D₂ = FDGrids.DiffMatrix(xs, width, 2; eltype=T)
    D₁⁺, D₂⁺ = LinearAlgebra.adjoint(D₁, ws), LinearAlgebra.adjoint(D₂, ws)
    return RectangularGrid((xs, xs), (D₁, D₁), (D₂, D₂), (D₁⁺, D₁⁺), (D₂⁺, D₂⁺),
                           (ws, ws), (α, 2π), (N, N, Nz, Nt), SQUARE_DUCT_AXES, SQUARE_DUCT_FFT_ORDER, T)
end

SquareDuctGrid(N::Int, width::Int, Nz::Int, Nt::Int, α::Real; kwargs...) =
    SquareDuctGrid(N, Nz, Nt, α; width, kwargs...)

# //////////////////////////////////////////////////////////////////////////// #
# ///                           Square-duct flow                           /// #
# //////////////////////////////////////////////////////////////////////////// #

"""
    SquareDuctFlow(g, Re; base=(nothing, nothing, nothing), f=1,
                   mode=AdjointDiscrete(), fftw_flags=FFTW.EXHAUSTIVE,
                   dealias=true) -> ProjectedNSE

Construct the three-component Cartesian primitive-variable equations for
pressure-driven square-duct flow at Reynolds number `Re`.

# Keywords

- `base`: Cartesian tuple `(U, V, W)`. Each entry is a cross-section array or
  `nothing`. The default has no base flow. To linearise about a streamwise
  laminar solution, place its `N × N` values in `W`, for example
  `(nothing, nothing, W)`.
- `f`: amplitude of a [`ConstantBodyForce`](@ref) applied to the third,
  streamwise velocity component at zero `(z,t)` wavenumber and every
  cross-section point. The conventional friction-scaled amplitude is `f=1`.
- `mode`: [`AdjointDiscrete`](@ref) or [`AdjointContinuous`](@ref) for the
  linearised operator. The quadrature-consistent discrete adjoint is the
  default.
- `fftw_flags`: FFTW planner flags, defaulting to `FFTW.EXHAUSTIVE`.
- `dealias`: when `true` (default), physical caches use the 3/2-rule padded
  streamwise and temporal grid while spectral caches retain the resolved size.

The constant force is present in nonlinear, forward-linearised, and adjoint
equation modes.

# Example

```julia
g = SquareDuctGrid(49, 63, 1, 0.5; width=7)
eq = SquareDuctFlow(g, 2000; f=1, fftw_flags=FFTW.ESTIMATE)
```
"""
function SquareDuctFlow(g::AbstractSquareDuctGrid, Re; base=(nothing, nothing, nothing), f=1,
                        mode=AdjointDiscrete(), fftw_flags=FFTW.EXHAUSTIVE, dealias=true)
    force = ConstantBodyForce(eltype(g)(f); component=3)
    construct_equations(g, Re, base, CartesianPrimitive3D(); force, mode, flags=fftw_flags, dealias)
end
