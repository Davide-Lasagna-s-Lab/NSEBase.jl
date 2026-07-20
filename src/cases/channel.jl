# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Channel grids and parallel-plate flow cases                                                // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #
#
# This file defines the channel-case layer: three numerical layouts, their concrete
# `RectangularGrid` aliases and factories, canonical wall-normal profiles, and equation constructors
# for flows between parallel walls. Physical `x` is streamwise, `y` is wall-normal, and `z` is
# spanwise.
#
# Physical cases
#
# - Plane Couette flow, driven by opposite wall velocities. Its default streamwise base profile is
#   `U(η)=η`, where the wall interval is normalised to `η∈[-1,1]`; `Ro` optionally adds rotation
#   about the spanwise axis.
# - Plane Poiseuille flow, driven by a constant streamwise pressure-gradient force `f`. Its default
#   base is `U(η)=1-η²`, and it may use the same spanwise-axis rotation.
# - Rayleigh–Bénard convection between isothermal plates. The two-dimensional state is `(u,v,θ)`
#   and the full state is `(u,v,w,θ)`. Both use the default conduction temperature
#   `Θ(η)=(1-η)/2` and buoyancy coefficient `Ri=Ra/(Re²Pr)`.
#
# Here `η=2(y-y₋)/(y₊-y₋)-1`, so each profile retains its canonical wall values for any
# `lim=(y₋,y₊)`. A base tuple follows the case's state order; an entry may be an array on the
# wall-normal grid or `nothing` for a zero component. When the `ProjectedNSE` wrapper is evaluated,
# it adds these profiles only to the zero Fourier mode before calling the equation operator.
#
# Numerical layouts
#
# The full 3D layout is stored as `(y,x,z,t)` and constructed by
# `ChannelGrid(Nx, Ny, Nz; Nt, α, β, ...)`, where `α=2π/Lx` and `β=2π/Lz`. Hydrodynamic cases use
# state `(u,v,w)` and Boussinesq flow adds temperature as `(u,v,w,θ)`.
#
# `StreamwiseInvariantChannelGrid(Ny, Nz; Nt, β, ...)` stores `(y,z,t)` and retains all three
# velocity components in the 2D3C order `(v,w,u)`. Its physical axis map records that `x` is absent,
# so `ddy!` is always wall-normal and `ddz!` is always spanwise. The 2D3C operator is restricted to
# this streamwise-invariant physical `(y,z)` formulation.
#
# `TwoDimensionalChannelGrid(Nx, Ny; Nt, α, ...)` is specifically the physical two-dimensional
# `(x,y)` channel. It stores `(y,x,t)` and has no spanwise dependence. Hydrodynamic cases use
# `(u,v)` and Boussinesq flow uses `(u,v,θ)`. Its physical naming is direct: `ddx!` is the
# streamwise Fourier derivative and `ddy!` is the wall-normal finite-difference derivative.
#
# All three factories return a `RectangularGrid{1}`. FDGrids supplies wall-normal points, quadrature,
# first and second derivatives, and their quadrature-weighted adjoints; the default is a
# Gauss–Lobatto distribution on `(-1,1)`. The full layout Fourier-transforms `(x,z,t)`, whereas the
# streamwise-invariant and two-dimensional layouts transform `(z,t)` and `(x,t)`, respectively.
# The unit-period `t` coordinate represents time or phase; `Nt=1` retains only its zero mode.
# Full-layout constructors accepting precomputed wall-normal points, matrices, adjoints, and weights
# are also provided.
#
# Boundary conditions
#
# A channel grid owns geometry and differentiation, and a flow constructor selects a base state,
# formulation, and forcing. Neither layer removes wall degrees of freedom or imposes no-slip,
# moving-wall, or fixed-temperature values. Those conditions belong to the basis or residual
# formulation. The canonical profiles carry the usual nonzero base values but are not themselves
# boundary-condition machinery.
#
# Typical use
#
#     streamwise_invariant = StreamwiseInvariantChannelGrid(65, 63; β=0.5, width=7)
#     rotating_couette = PlaneCouetteFlow(streamwise_invariant, 400;
#                                         Ro=0.1, fftw_flags=FFTW.ESTIMATE)
#
#     two_dimensional = TwoDimensionalChannelGrid(63, 65; α=0.5, width=7)
#     poiseuille_2d = PlanePoiseuilleFlow(two_dimensional, 1000;
#                                        f=1, fftw_flags=FFTW.ESTIMATE)
#     convection_2d = RayleighBenardFlow(two_dimensional, 1, 0.7, 1e5;
#                                        fftw_flags=FFTW.ESTIMATE)
#
#     full = ChannelGrid(63, 65, 63; α=0.5, β=1, width=7)
#     couette = PlaneCouetteFlow(full, 400; fftw_flags=FFTW.ESTIMATE)
#     poiseuille = PlanePoiseuilleFlow(full, 1000; f=1, fftw_flags=FFTW.ESTIMATE)
#     convection = RayleighBenardFlow(full, 100, 0.7, 1e5; fftw_flags=FFTW.ESTIMATE)
#
# Extending the cases
#
# `AbstractStreamwiseInvariantChannelGrid`, `AbstractTwoDimensionalChannelGrid`, and
# `AbstractChannel3DGrid` are exact layout contracts, not concrete grid implementations. A compatible
# serial or distributed grid must subtype `AbstractGrid` with the corresponding axis-map and
# FFT-order parameters and implement its interface. Profiles shared by all layouts should dispatch
# on `AbstractChannelGrid`; layout-specific methods should use the narrowest contract. A related 3D
# flow can reuse the geometry and primitive-variable operators:
#
#     my_channel_base(g::AbstractChannelGrid) = 1 .- plane_couette_base(g) .^ 4
#
#     function MyChannelFlow(g::AbstractChannel3DGrid, Re;
#                            base=(my_channel_base(g), nothing, nothing), force=NoForce(),
#                            mode=AdjointDiscrete(), fftw_flags=FFTW.EXHAUSTIVE, dealias=true)
#         return construct_equations(g, Re, base, CartesianPrimitive3D(); force, mode,
#                                    flags=fftw_flags, dealias)
#     end
#
# A streamwise-invariant 2D3C factory uses `CartesianPrimitive2D3C()` and base order `(V,W,U)`; a
# two-dimensional factory uses `CartesianPrimitive2D()` and base order `(U,V)`. Thermal factories
# use `CartesianPrimitive2DBoussinesq` with `(U,V,Θ)` or `CartesianPrimitive3DBoussinesq` with
# `(U,V,W,Θ)` according to the channel layout.

# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Streamwise-invariant channel layout                                                        // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

"""
    STREAMWISE_INVARIANT_CHANNEL_AXES = (nothing, 1, 2, 3)

Map physical coordinates `(x,y,z,t)` to streamwise-invariant channel storage order `(y,z,t)`.
Streamwise `x` is absent, wall-normal `y` occupies storage dimension 1, spanwise `z` dimension 2,
and time dimension 3.

Coordinate names retain their physical meaning: `ddx!` is the derivative along the absent
streamwise direction, `ddy!` applies the wall-normal finite-difference matrix, and `ddz!` applies
the spanwise Fourier derivative. [`CartesianPrimitive2D3C`](@ref) is defined specifically on this
physical `(y,z)` plane with velocity state `(v,w,u)`.
"""
const STREAMWISE_INVARIANT_CHANNEL_AXES = (nothing, 1, 2, 3)

"""
    STREAMWISE_INVARIANT_CHANNEL_FFT_ORDER = (2, 3)

Fourier storage dimensions for a streamwise-invariant channel. Spanwise
storage dimension 2 uses the real-to-complex transform and unit-period time in
dimension 3 uses a complex transform.
"""
const STREAMWISE_INVARIANT_CHANNEL_FFT_ORDER = (2, 3)

"""
    STREAMWISE_INVARIANT_CHANNEL_INHOMOGENEOUS_DIMS = (1,)

The streamwise-invariant channel's wall-normal finite-difference storage
dimension.
"""
const STREAMWISE_INVARIANT_CHANNEL_INHOMOGENEOUS_DIMS = (1,)

# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Two-dimensional channel layout                                                             // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

"""
    TWO_DIMENSIONAL_CHANNEL_AXES = (2, 1, nothing, 3)

Map physical coordinates `(x,y,z,t)` to two-dimensional channel storage order `(y,x,t)`.
Streamwise `x` occupies storage dimension 2, wall-normal `y` dimension 1, physical `z` is absent,
and time occupies dimension 3. Hydrodynamic equations use [`CartesianPrimitive2D`](@ref), while
two-dimensional thermal flow uses [`CartesianPrimitive2DBoussinesq`](@ref).
"""
const TWO_DIMENSIONAL_CHANNEL_AXES = (2, 1, nothing, 3)

"""
    TWO_DIMENSIONAL_CHANNEL_FFT_ORDER = (2, 3)

Fourier storage dimensions for a two-dimensional channel. Streamwise dimension 2 uses the
real-to-complex transform and unit-period time in dimension 3 uses a complex transform.
"""
const TWO_DIMENSIONAL_CHANNEL_FFT_ORDER = (2, 3)

"""
    TWO_DIMENSIONAL_CHANNEL_INHOMOGENEOUS_DIMS = (1,)

The two-dimensional channel's wall-normal finite-difference storage dimension.
"""
const TWO_DIMENSIONAL_CHANNEL_INHOMOGENEOUS_DIMS = (1,)

# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // 3D channel layout                                                                          // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

"""
    CHANNEL_3D_AXES = (2, 1, 3, 4)

Map physical coordinates `(x,y,z,t)` to full-channel storage order
`(y,x,z,t)`. Wall-normal `y` occupies dimension 1, streamwise `x` dimension 2,
spanwise `z` dimension 3, and time dimension 4.
"""
const CHANNEL_3D_AXES = (2, 1, 3, 4)

"""
    CHANNEL_3D_FFT_ORDER = (2, 3, 4)

Fourier storage dimensions for a full three-dimensional channel, in
streamwise, spanwise, and temporal order. Streamwise dimension 2 uses the
real-to-complex transform.
"""
const CHANNEL_3D_FFT_ORDER = (2, 3, 4)

"""
    CHANNEL_3D_INHOMOGENEOUS_DIMS = (1,)

The full channel's wall-normal finite-difference storage dimension.
"""
const CHANNEL_3D_INHOMOGENEOUS_DIMS = (1,)

# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Channel layout contracts                                                                   // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

"""
    AbstractStreamwiseInvariantChannelGrid{T}

Layout contract for serial, MPI-decomposed, and downstream streamwise-invariant channel grids with
real scalar type `T`. Arrays are stored as `(y,z,t)` and use the physical axis map documented by
[`STREAMWISE_INVARIANT_CHANNEL_AXES`](@ref), in which streamwise `x` is absent.
"""
const AbstractStreamwiseInvariantChannelGrid{T} = AbstractGrid{
    T, 3, STREAMWISE_INVARIANT_CHANNEL_AXES, STREAMWISE_INVARIANT_CHANNEL_FFT_ORDER,
}

"""
    AbstractTwoDimensionalChannelGrid{T}

Layout contract for serial, MPI-decomposed, and downstream two-dimensional channel grids with real
scalar type `T`, physical plane `(x,y)`, and storage order `(y,x,t)`. The grid contract is
independent of whether the equation state is hydrodynamic `(u,v)` or Boussinesq `(u,v,θ)`.
"""
const AbstractTwoDimensionalChannelGrid{T} = AbstractGrid{
    T, 3, TWO_DIMENSIONAL_CHANNEL_AXES, TWO_DIMENSIONAL_CHANNEL_FFT_ORDER,
}

"""
    AbstractChannel3DGrid{T}

Layout contract for serial, MPI-decomposed, and downstream full-channel grids with real scalar type
`T` and storage order `(y,x,z,t)`.
"""
const AbstractChannel3DGrid{T} = AbstractGrid{T, 4, CHANNEL_3D_AXES, CHANNEL_3D_FFT_ORDER}

"""
    AbstractChannelGrid{T}

Union of every supported channel layout contract. Shared geometry and profile utilities dispatch on
this union; equation constructors dispatch on an exact layout whenever component ordering or
formulation differs.
"""
const AbstractChannelGrid{T} = Union{
    AbstractStreamwiseInvariantChannelGrid{T}, AbstractTwoDimensionalChannelGrid{T},
    AbstractChannel3DGrid{T},
}

# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Concrete channel-grid aliases                                                              // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

"""
    StreamwiseInvariantChannelGrid

Case alias for a [`RectangularGrid{1}`](@ref) stored as `(y,z,t)`, with the exact axis and Fourier
contracts of [`AbstractStreamwiseInvariantChannelGrid`](@ref). Construct it from resolutions with
[`StreamwiseInvariantChannelGrid`](@ref).
"""
const StreamwiseInvariantChannelGrid{T, S, XS, D1, D2, A1, A2, WS, WP} = RectangularGrid{
    1, T, S, 3, STREAMWISE_INVARIANT_CHANNEL_AXES, STREAMWISE_INVARIANT_CHANNEL_FFT_ORDER,
    XS, D1, D2, A1, A2, WS, WP, 2,
}

"""
    TwoDimensionalChannelGrid

Case alias for a strictly two-dimensional [`RectangularGrid{1}`](@ref) stored as `(y,x,t)`, with
the exact axis and Fourier contracts of [`AbstractTwoDimensionalChannelGrid`](@ref). Construct it
from resolutions with [`TwoDimensionalChannelGrid`](@ref).
"""
const TwoDimensionalChannelGrid{T, S, XS, D1, D2, A1, A2, WS, WP} = RectangularGrid{
    1, T, S, 3, TWO_DIMENSIONAL_CHANNEL_AXES, TWO_DIMENSIONAL_CHANNEL_FFT_ORDER,
    XS, D1, D2, A1, A2, WS, WP, 2,
}

"""
    ChannelGrid

Case alias for a full [`RectangularGrid{1}`](@ref) stored as `(y,x,z,t)`, with the exact axis and
Fourier contracts of [`AbstractChannel3DGrid`](@ref). Construct it from resolutions with
[`ChannelGrid`](@ref), or from precomputed wall-normal data with one of the
lower-level constructors documented below.
"""
const ChannelGrid{T, S, XS, D1, D2, A1, A2, WS, WP} = RectangularGrid{
    1, T, S, 4, CHANNEL_3D_AXES, CHANNEL_3D_FFT_ORDER, XS, D1, D2, A1, A2, WS, WP, 3,
}

# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Wall-normal finite-difference discretisation                                               // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

"""
    _channel_fd(Ny, lim, dist, width, T) -> (y, Dy, Dy2, Dya, Dy2a, ws)

Build the wall-normal finite-difference discretisation shared by every channel
layout.

# Arguments

- `Ny`: number of wall-normal collocation points.
- `lim`: physical wall locations `(y₋,y₊)`.
- `dist`: FDGrids point distribution on `lim`.
- `width`: stencil width passed to `FDGrids.DiffMatrix`.
- `T`: real scalar type of the returned points, weights, and operators.

# Returns

The collocation points `y`, first- and second-derivative operators `Dy` and
`Dy2`, their quadrature-weighted adjoints `Dya` and `Dy2a`, and the quadrature
weights `ws`. The ordering is the same as the corresponding fields of
[`RectangularGrid`](@ref).

FDGrids validates the interval, point count, distribution, and stencil width.
The adjoints satisfy the weighted inner-product convention used by NSEBase's
discrete-adjoint operators.
"""
function _channel_fd(Ny::Int, lim::NTuple{2, <:Real}, dist::FDGrids.AbstractGridDistribution,
                     width::Int, ::Type{T}) where {T<:Real}
    grid = FDGrids.grid(Ny, lim[1], lim[2], dist)
    y, ws = Vector{T}(grid.xs), Vector{T}(grid.ws)
    Dy = FDGrids.DiffMatrix(y, width, 1; eltype=T)
    Dy2 = FDGrids.DiffMatrix(y, width, 2; eltype=T)
    Dya, Dy2a = LinearAlgebra.adjoint(Dy, ws), LinearAlgebra.adjoint(Dy2, ws)
    return y, Dy, Dy2, Dya, Dy2a, ws
end

# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // High-level channel-grid factories                                                          // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

"""
    StreamwiseInvariantChannelGrid(Ny, Nz; Nt=1, β=1, lim=(-1,1),
                                    dist=FDGrids.GaussLobattoGrid(), width=5, T=Float64)
    StreamwiseInvariantChannelGrid(Ny, Nz, Nt, width, β; kwargs...)

Construct a streamwise-invariant channel [`RectangularGrid{1}`](@ref). Fields are stored as
`(y,z,t)` with size `(Ny,Nz,Nt)`, and `β=2π/Lz` is the spanwise Fourier wavenumber scale. The
unit-period temporal scale is `2π`.

This layout retains all three velocity components while omitting streamwise dependence. It is
therefore paired with [`CartesianPrimitive2D3C`](@ref), whose state order is `(v,w,u)` on the
physical `(y,z)` plane. Coordinate names are never relabelled: `ddx!` is inactive, `ddy!`
differentiates wall-normal `y`, and `ddz!` differentiates spanwise `z`.

# Keywords

- `Nt`: positive odd unit-period temporal resolution, defaulting to `1`.
- `β`: spanwise Fourier wavenumber scale, defaulting to `1`.
- `lim`: wall-normal interval, defaulting to `(-1,1)`.
- `dist`: wall-normal FDGrids distribution. The default Gauss–Lobatto grid includes both walls and
  supplies Clenshaw–Curtis quadrature weights.
- `width`: odd finite-difference stencil width of at least three. The compact weighted adjoints
  require `Ny > 2width`.
- `T`: real scalar type for points, operators, weights, scales, and fields.

The compact positional form contains the same information without keywords: resolutions first,
then `width`, then `β`. Fourier resolutions must be positive and odd. The grid supplies wall points
and operators but does not impose velocity boundary values.

# Example

```julia
g = StreamwiseInvariantChannelGrid(65, 63; Nt=1, β=0.5, width=7)
```
"""
function StreamwiseInvariantChannelGrid(Ny::Int, Nz::Int; Nt::Int=1, β::Real=1,
                                        lim::NTuple{2, <:Real}=(-1, 1),
                                        dist::FDGrids.AbstractGridDistribution=FDGrids.GaussLobattoGrid(),
                                        width::Int=5, T::Type{<:Real}=Float64)
    y, Dy, Dy2, Dya, Dy2a, ws = _channel_fd(Ny, lim, dist, width, T)
    return RectangularGrid((y,), (Dy,), (Dy2,), (Dya,), (Dy2a,), (ws,), (β, 2π),
                           (Ny, Nz, Nt), STREAMWISE_INVARIANT_CHANNEL_AXES,
                           STREAMWISE_INVARIANT_CHANNEL_FFT_ORDER, T)
end

StreamwiseInvariantChannelGrid(Ny::Int, Nz::Int, Nt::Int, width::Int, β::Real; kwargs...) =
    StreamwiseInvariantChannelGrid(Ny, Nz; Nt, width, β, kwargs...)

"""
    TwoDimensionalChannelGrid(Nx, Ny; Nt=1, α=1, lim=(-1,1),
                                  dist=FDGrids.GaussLobattoGrid(), width=5, T=Float64)
    TwoDimensionalChannelGrid(Nx, Ny, Nt, width, α; kwargs...)

Construct a physical two-dimensional `(x,y)` channel [`RectangularGrid{1}`](@ref). Fields are stored
as `(y,x,t)` with size `(Ny,Nx,Nt)`, and `α=2π/Lx` is the streamwise Fourier wavenumber scale. The
unit-period temporal scale is `2π`; physical `z` and the spanwise velocity are absent.

This is the ordinary two-dimensional channel layout. Physical and computational names coincide:
`ddx!` is the streamwise Fourier derivative and `ddy!` is the wall-normal finite-difference
derivative. It supports plane Couette and Poiseuille flow with [`CartesianPrimitive2D`](@ref) state
`(u,v)`, and Rayleigh–Bénard convection with [`CartesianPrimitive2DBoussinesq`](@ref) state
`(u,v,θ)`.

The keywords have the same meaning as for [`StreamwiseInvariantChannelGrid`](@ref), with `α`
replacing the absent spanwise scale `β`. The compact positional form orders resolutions first, then
`width`, then `α`. Fourier resolutions must be positive and odd, and boundary values remain the
responsibility of the basis or residual formulation.

# Example

```julia
g = TwoDimensionalChannelGrid(63, 65; Nt=1, α=0.5, width=7)
equations = PlanePoiseuilleFlow(g, 1000; f=1, fftw_flags=FFTW.ESTIMATE)
convection = RayleighBenardFlow(g, 1, 0.71, 1e4; fftw_flags=FFTW.ESTIMATE)
```
"""
function TwoDimensionalChannelGrid(Nx::Int, Ny::Int; Nt::Int=1, α::Real=1,
                                      lim::NTuple{2, <:Real}=(-1, 1),
                                      dist::FDGrids.AbstractGridDistribution=FDGrids.GaussLobattoGrid(),
                                      width::Int=5, T::Type{<:Real}=Float64)
    y, Dy, Dy2, Dya, Dy2a, ws = _channel_fd(Ny, lim, dist, width, T)
    return RectangularGrid((y,), (Dy,), (Dy2,), (Dya,), (Dy2a,), (ws,), (α, 2π),
                           (Ny, Nx, Nt), TWO_DIMENSIONAL_CHANNEL_AXES,
                           TWO_DIMENSIONAL_CHANNEL_FFT_ORDER, T)
end

TwoDimensionalChannelGrid(Nx::Int, Ny::Int, Nt::Int, width::Int, α::Real; kwargs...) =
    TwoDimensionalChannelGrid(Nx, Ny; Nt, width, α, kwargs...)

"""
    ChannelGrid(Nx, Ny, Nz; Nt=1, α=1, β=1, lim=(-1,1),
                dist=FDGrids.GaussLobattoGrid(), width=5, T=Float64)
    ChannelGrid(Nx, Ny, Nz, Nt, width, α, β; kwargs...)

Construct a full three-dimensional channel [`RectangularGrid{1}`](@ref), stored as `(y,x,z,t)` with
size `(Ny,Nx,Nz,Nt)`. The Fourier scales are `(α,β,2π)`, where `α=2π/Lx` and `β=2π/Lz`. Wall-normal
points and operators use the same keywords and defaults as the reduced channel factories.

The compact positional form orders all resolutions first, followed by `width`, `α`, and `β`.
Fourier resolutions must be positive and odd. For reduced layouts, use the explicitly named
[`StreamwiseInvariantChannelGrid`](@ref) or [`TwoDimensionalChannelGrid`](@ref), respectively.

# Example

```julia
g = ChannelGrid(63, 65, 63; Nt=1, α=0.5, β=1, width=7)
```
"""
function ChannelGrid(Nx::Int, Ny::Int, Nz::Int; Nt::Int=1, α::Real=1, β::Real=1,
                     lim::NTuple{2, <:Real}=(-1, 1),
                     dist::FDGrids.AbstractGridDistribution=FDGrids.GaussLobattoGrid(),
                     width::Int=5, T::Type{<:Real}=Float64)
    y, Dy, Dy2, Dya, Dy2a, ws = _channel_fd(Ny, lim, dist, width, T)
    return RectangularGrid((y,), (Dy,), (Dy2,), (Dya,), (Dy2a,), (ws,), (α, β, 2π),
                           (Ny, Nx, Nz, Nt), CHANNEL_3D_AXES, CHANNEL_3D_FFT_ORDER, T)
end

ChannelGrid(Nx::Int, Ny::Int, Nz::Int, Nt::Int, width::Int, α::Real, β::Real; kwargs...) =
    ChannelGrid(Nx, Ny, Nz; Nt, width, α, β, kwargs...)

# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Precomputed-data channel-grid constructors                                                 // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

"""
    ChannelGrid(y, Nx, Nz, Nt, α, β, Dy, Dy2, Dya, Dy2a, ws, T=Float64)
    ChannelGrid(y, Nx, Nz, Nt, α, β, Dy, Dy2, ws, T=Float64; adjoint_diff=true)

Construct a full 3D channel from caller-supplied wall-normal data. The first
form stores the supplied weighted adjoints. The second is the original
ReSolver-ChannelFlow.jl interface: with `adjoint_diff=true` it derives `Dya`
and `Dy2a` from `(Dy,Dy2,ws)`; with `false`, the forward matrices are also used
for adjoint differentiation. All data are converted to `T` when necessary.
"""
function ChannelGrid(y::AbstractVector, Nx::Int, Nz::Int, Nt::Int, α::Real, β::Real,
                     Dy::AbstractMatrix, Dy2::AbstractMatrix, Dya::AbstractMatrix,
                     Dy2a::AbstractMatrix, ws::AbstractVector, ::Type{T}=Float64) where {T<:Real}
    Ny = length(y)
    return RectangularGrid((y,), (Dy,), (Dy2,), (Dya,), (Dy2a,), (ws,), (α, β, 2π),
                           (Ny, Nx, Nz, Nt), CHANNEL_3D_AXES, CHANNEL_3D_FFT_ORDER, T)
end

function ChannelGrid(y::AbstractVector, Nx::Int, Nz::Int, Nt::Int, α::Real, β::Real,
                     Dy::AbstractMatrix, Dy2::AbstractMatrix, ws::AbstractVector,
                     ::Type{T}=Float64; adjoint_diff::Bool=true) where {T<:Real}
    yT, wsT = y isa Vector{T} ? y : Vector{T}(y), ws isa Vector{T} ? ws : Vector{T}(ws)
    DyT, Dy2T = eltype(Dy) === T ? Dy : T.(Dy), eltype(Dy2) === T ? Dy2 : T.(Dy2)
    Dya = adjoint_diff ? LinearAlgebra.adjoint(DyT, wsT) : DyT
    Dy2a = adjoint_diff ? LinearAlgebra.adjoint(Dy2T, wsT) : Dy2T
    return ChannelGrid(yT, Nx, Nz, Nt, α, β, DyT, Dy2T, Dya, Dy2a, wsT, T)
end

# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Reference profiles                                                                         // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

"""
    plane_couette_base(g::AbstractChannelGrid) -> Vector

Return the canonical Couette profile `U(η)=η`. The wall-normal points are
mapped from the grid's interval `[y₋,y₊]` to `η∈[-1,1]`, so the two walls have
velocities `-1` and `1` for every supported channel layout and any valid `lim`.

For a decomposed grid, the MPI extension uses the undecomposed parent's global
wall locations while evaluating the profile only at the rank-local points.
"""
function plane_couette_base(g::AbstractChannelGrid{T}) where {T}
    dim = only(inhomogeneous_storage_dims(g))
    y = copy(vec(points(g)[dim]))
    y₋, y₊ = extrema(y)
    y₊ > y₋ || throw(ArgumentError("channel walls must have distinct coordinates"))
    return @. T(2) * (y - y₋) / (y₊ - y₋) - one(T)
end

"""
    plane_poiseuille_base(g::AbstractChannelGrid) -> Vector

Return `U(η)=1-η²` using the normalized wall coordinate defined by
[`plane_couette_base`](@ref). The profile is zero at both walls and reaches
one at the channel centre independently of the physical wall interval.
"""
plane_poiseuille_base(g::AbstractChannelGrid) = one(eltype(g)) .- plane_couette_base(g) .^ 2

"""
    rbc_base_temperature(g::AbstractChannelGrid) -> Vector

Return the Rayleigh–Bénard conduction profile `Θ(η)=(1-η)/2`, using the same
normalized wall coordinate as the channel velocity profiles. The lower and
upper plates therefore have temperatures one and zero for any valid `lim`.
The same profile is used by the two-dimensional and full three-dimensional
Rayleigh–Bénard cases.
"""
rbc_base_temperature(g::AbstractChannelGrid) = (one(eltype(g)) .- plane_couette_base(g)) ./ 2

# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Plane Couette flow                                                                         // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

"""
    PlaneCouetteFlow(g, Re; Ro=0, base=..., mode=AdjointDiscrete(),
                     fftw_flags=FFTW.EXHAUSTIVE, dealias=true) -> ProjectedNSE

Construct plane Couette equations on any supported channel layout.

- A full 3D grid selects [`CartesianPrimitive3D`](@ref), state `(u,v,w)`,
  default base `(U,nothing,nothing)`, and Coriolis components `(1,2)`.
- A streamwise-invariant grid selects [`CartesianPrimitive2D3C`](@ref), state `(v,w,u)`, default
  base `(nothing,nothing,U)`, and Coriolis components `(3,1)`.
- A two-dimensional grid selects [`CartesianPrimitive2D`](@ref), state `(u,v)`, default base
  `(U,nothing)`, and Coriolis components `(1,2)`.

Here `U=plane_couette_base(g)`. `Ro=2Ωh/Uw` is the rotation number about the
spanwise axis; zero installs [`NoForce`](@ref). `mode` selects the continuous
or discrete adjoint, `fftw_flags` controls FFTW planning, and `dealias=true`
pads only Fourier directions for nonlinear products. Boundary values remain
the responsibility of the basis or residual formulation.
"""
function PlaneCouetteFlow(g::AbstractChannel3DGrid, Re; Ro=0,
                          base=(plane_couette_base(g), nothing, nothing), mode=AdjointDiscrete(),
                          fftw_flags=FFTW.EXHAUSTIVE, dealias=true)
    length(base) == 3 || throw(ArgumentError("a 3D channel base must contain (U, V, W)"))
    force = iszero(Ro) ? NoForce() : CoriolisForce(eltype(g)(Ro))
    return construct_equations(g, Re, base, CartesianPrimitive3D(); force, mode, flags=fftw_flags, dealias)
end

function PlaneCouetteFlow(g::AbstractStreamwiseInvariantChannelGrid, Re; Ro=0,
                          base=(nothing, nothing, plane_couette_base(g)), mode=AdjointDiscrete(),
                          fftw_flags=FFTW.EXHAUSTIVE, dealias=true)
    length(base) == 3 || throw(ArgumentError("a streamwise-invariant channel base must contain (V, W, U)"))
    force = iszero(Ro) ? NoForce() : CoriolisForce(eltype(g)(Ro); components=(3, 1))
    return construct_equations(g, Re, base, CartesianPrimitive2D3C(); force, mode, flags=fftw_flags, dealias)
end

function PlaneCouetteFlow(g::AbstractTwoDimensionalChannelGrid, Re; Ro=0,
                          base=(plane_couette_base(g), nothing), mode=AdjointDiscrete(),
                          fftw_flags=FFTW.EXHAUSTIVE, dealias=true)
    length(base) == 2 || throw(ArgumentError("a two-dimensional channel base must contain (U, V)"))
    force = iszero(Ro) ? NoForce() : CoriolisForce(eltype(g)(Ro))
    return construct_equations(g, Re, base, CartesianPrimitive2D(); force, mode, flags=fftw_flags, dealias)
end

# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Plane Poiseuille flow                                                                      // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

"""
    PlanePoiseuilleFlow(g, Re; Ro=0, f=1, base=..., mode=AdjointDiscrete(),
                        fftw_flags=FFTW.EXHAUSTIVE, dealias=true) -> ProjectedNSE

Construct pressure-driven plane Poiseuille equations on any supported channel layout. The default
base is the normalized laminar profile `U(η)=1-η²`, placed in the streamwise velocity component.
`f` applies a [`ConstantBodyForce`](@ref) to that component at the zero Fourier mode. Nonzero `Ro`
adds spanwise-axis Coriolis forcing.

- Full 3D uses [`CartesianPrimitive3D`](@ref), state `(u,v,w)`, base `(U,nothing,nothing)`, pressure
  component `1`, and Coriolis components `(1,2)`.
- Streamwise invariant uses [`CartesianPrimitive2D3C`](@ref), state `(v,w,u)`, base
  `(nothing,nothing,U)`, pressure component `3`, and Coriolis components `(3,1)`.
- Two-dimensional uses [`CartesianPrimitive2D`](@ref), state `(u,v)`, base `(U,nothing)`, pressure
  component `1`, and Coriolis components `(1,2)`. This is the standard two-dimensional plane
  Poiseuille case.
"""
function PlanePoiseuilleFlow(g::AbstractChannel3DGrid, Re; Ro=0, f=1,
                             base=(plane_poiseuille_base(g), nothing, nothing), mode=AdjointDiscrete(),
                             fftw_flags=FFTW.EXHAUSTIVE, dealias=true)
    length(base) == 3 || throw(ArgumentError("a 3D channel base must contain (U, V, W)"))
    pressure = ConstantBodyForce(eltype(g)(f); component=1)
    force = iszero(Ro) ? pressure : CompoundForcing(pressure, CoriolisForce(eltype(g)(Ro)))
    return construct_equations(g, Re, base, CartesianPrimitive3D(); force, mode, flags=fftw_flags, dealias)
end

function PlanePoiseuilleFlow(g::AbstractStreamwiseInvariantChannelGrid, Re; Ro=0, f=1,
                             base=(nothing, nothing, plane_poiseuille_base(g)), mode=AdjointDiscrete(),
                             fftw_flags=FFTW.EXHAUSTIVE, dealias=true)
    length(base) == 3 || throw(ArgumentError("a streamwise-invariant channel base must contain (V, W, U)"))
    pressure = ConstantBodyForce(eltype(g)(f); component=3)
    force = iszero(Ro) ? pressure : CompoundForcing(pressure, CoriolisForce(eltype(g)(Ro); components=(3, 1)))
    return construct_equations(g, Re, base, CartesianPrimitive2D3C(); force, mode, flags=fftw_flags, dealias)
end

function PlanePoiseuilleFlow(g::AbstractTwoDimensionalChannelGrid, Re; Ro=0, f=1,
                             base=(plane_poiseuille_base(g), nothing), mode=AdjointDiscrete(),
                             fftw_flags=FFTW.EXHAUSTIVE, dealias=true)
    length(base) == 2 || throw(ArgumentError("a two-dimensional channel base must contain (U, V)"))
    pressure = ConstantBodyForce(eltype(g)(f); component=1)
    force = iszero(Ro) ? pressure : CompoundForcing(pressure, CoriolisForce(eltype(g)(Ro)))
    return construct_equations(g, Re, base, CartesianPrimitive2D(); force, mode, flags=fftw_flags, dealias)
end

# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Rayleigh–Bénard convection                                                                 // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

"""
    RayleighBenardFlow(g::AbstractTwoDimensionalChannelGrid, Re, Pr, Ra;
                       base=(nothing,nothing,rbc_base_temperature(g)),
                       mode=AdjointDiscrete(), fftw_flags=FFTW.EXHAUSTIVE,
                       dealias=true) -> ProjectedNSE
    RayleighBenardFlow(g::AbstractChannel3DGrid, Re, Pr, Ra;
                       base=(nothing,nothing,nothing,rbc_base_temperature(g)),
                       mode=AdjointDiscrete(), fftw_flags=FFTW.EXHAUSTIVE,
                       dealias=true) -> ProjectedNSE

Construct two- or three-dimensional Boussinesq equations for Rayleigh–Bénard convection between
isothermal channel walls. A [`TwoDimensionalChannelGrid`](@ref) selects
[`CartesianPrimitive2DBoussinesq`](@ref) with state/base order `(u,v,θ)` / `(U,V,Θ)`. A full
[`ChannelGrid`](@ref) selects [`CartesianPrimitive3DBoussinesq`](@ref) with order `(u,v,w,θ)` /
`(U,V,W,Θ)`.

Both layouts preserve physical coordinate names: `x` is horizontal/streamwise, `y` is
vertical/wall-normal, and buoyancy acts on velocity component two. The Richardson coefficient is
`Ri=Ra/(Re²Pr)`; setting `Re=1` gives `Ri=Ra/Pr`, while `Ra=0` reduces temperature to a passive
scalar. The default base is motionless conduction with lower- and upper-plate temperatures one and
zero, returned by [`rbc_base_temperature`](@ref).

The grid and case constructor do not impose no-slip or fixed-temperature boundary conditions.
Those remain the responsibility of the basis or residual formulation.
"""
function RayleighBenardFlow(g::AbstractTwoDimensionalChannelGrid, Re, Pr, Ra;
                            base=(nothing, nothing, rbc_base_temperature(g)),
                            mode=AdjointDiscrete(), fftw_flags=FFTW.EXHAUSTIVE, dealias=true)
    length(base) == 3 || throw(ArgumentError("a 2D Rayleigh–Bénard base must contain (U, V, Θ)"))
    formulation = CartesianPrimitive2DBoussinesq(Pr, Ra / (Re^2 * Pr); grav=2)
    return construct_equations(g, Re, base, formulation; mode, flags=fftw_flags, dealias)
end

function RayleighBenardFlow(g::AbstractChannel3DGrid, Re, Pr, Ra;
                            base=(nothing, nothing, nothing, rbc_base_temperature(g)),
                            mode=AdjointDiscrete(), fftw_flags=FFTW.EXHAUSTIVE, dealias=true)
    length(base) == 4 || throw(ArgumentError("a 3D Rayleigh–Bénard base must contain (U, V, W, Θ)"))
    formulation = CartesianPrimitive3DBoussinesq(Pr, Ra / (Re^2 * Pr); grav=2)
    return construct_equations(g, Re, base, formulation; mode, flags=fftw_flags, dealias)
end
