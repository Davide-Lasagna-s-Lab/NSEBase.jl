# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Channel grids and flows                                                                    // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // 2D3C channel layout                                                                        // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

"""
    CHANNEL_2D3C_AXES = (1, 2, nothing, 3)

Map the computational coordinates used by [`CartesianPrimitive2D3C`](@ref) to
the streamwise-independent channel storage order `(y,z,t)`. Computational `x`
is physical wall-normal `y`, computational `y` is physical spanwise `z`, the
third spatial coordinate is absent, and time occupies storage dimension 3.

This deliberate computational relabelling lets `ddx!` apply the wall-normal
finite-difference matrix and `ddy!` apply the spanwise Fourier derivative. The
velocity state is `(v,w,u)`, with streamwise velocity as the third component.
"""
const CHANNEL_2D3C_AXES = (1, 2, nothing, 3)

"""
    CHANNEL_2D3C_FFT_ORDER = (2, 3)

Fourier storage dimensions for a streamwise-independent channel. Spanwise
storage dimension 2 uses the real-to-complex transform and unit-period time in
dimension 3 uses a complex transform.
"""
const CHANNEL_2D3C_FFT_ORDER = (2, 3)

"""
    CHANNEL_2D3C_INHOMOGENEOUS_DIMS = (1,)

The streamwise-independent channel's wall-normal finite-difference storage
dimension.
"""
const CHANNEL_2D3C_INHOMOGENEOUS_DIMS = (1,)

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
    AbstractChannel2D3CGrid{T}

Layout contract for serial, MPI-decomposed, and downstream
streamwise-independent channel grids with real scalar type `T`. Arrays are
stored as `(y,z,t)` and use the computational axis convention documented by
[`CHANNEL_2D3C_AXES`](@ref).
"""
const AbstractChannel2D3CGrid{T} =
    AbstractGrid{T, 3, CHANNEL_2D3C_AXES, CHANNEL_2D3C_FFT_ORDER}

"""
    AbstractChannel3DGrid{T}

Layout contract for serial, MPI-decomposed, and downstream full-channel grids
with real scalar type `T` and storage order `(y,x,z,t)`.
"""
const AbstractChannel3DGrid{T} = AbstractGrid{T, 4, CHANNEL_3D_AXES, CHANNEL_3D_FFT_ORDER}

"""
    AbstractChannelGrid{T}

Union of the supported streamwise-independent 2D3C and full 3D channel layout
contracts. Shared geometry and profile utilities dispatch on this union;
equation constructors dispatch on an exact layout whenever component ordering
or formulation differs.
"""
const AbstractChannelGrid{T} = Union{AbstractChannel2D3CGrid{T}, AbstractChannel3DGrid{T}}

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
# // High-level channel-grid factory                                                            // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

"""
    ChannelGrid(Ny, Nz; Nt=1, β=1, lim=(-1,1),
                dist=FDGrids.GaussLobattoGrid(), width=5, T=Float64)
    ChannelGrid(Nx, Ny, Nz; Nt=1, α=1, β=1, lim=(-1,1),
                dist=FDGrids.GaussLobattoGrid(), width=5, T=Float64)
    ChannelGrid(Ny, Nz, Nt, width, β; kwargs...)
    ChannelGrid(Nx, Ny, Nz, Nt, width, α, β; kwargs...)

Construct a streamwise-independent 2D3C or full 3D channel
[`RectangularGrid{1}`](@ref). As for [`LidDrivenCavityGrid`](@ref), the number
of positional resolutions identifies the spatial layout and `Nt` is a keyword.

The two-resolution form stores `(y,z,t)`, uses Fourier scale `β=2π/Lz`, and is
intended for the 2D3C streamwise-independent Couette formulation. The
three-resolution form stores `(y,x,z,t)` and uses `α=2π/Lx` and `β=2π/Lz`.
Both forms build exactly the same wall-normal FDGrids points, first and second
derivatives, quadrature weights, and quadrature-weighted adjoints.

The compact positional forms contain the same information without keywords.
Every resolution comes first, followed by `width` and then the Fourier
scale or scales. No historical width-second or wall-normal-first ordering is
supported.

# Keywords

- `Nt`: positive odd unit-period temporal resolution, defaulting to `1`.
- `α`, `β`: Fourier wavenumber scales; both default to `1`.
- `lim`: wall-normal interval, defaulting to `(-1,1)`.
- `dist`: wall-normal FDGrids distribution. The default Gauss–Lobatto grid
  includes both walls and supplies Clenshaw–Curtis quadrature weights.
- `width`: odd finite-difference stencil width of at least three. The compact
  weighted adjoints require `Ny > 2width`.
- `T`: real scalar type for points, operators, weights, scales, and fields.

Fourier resolutions must be positive and odd. The grid supplies wall points
and numerical operators but does not impose velocity or thermal boundary
values.

# Examples

```julia
g2 = ChannelGrid(65, 63; Nt=1, β=0.5, width=7)
g3 = ChannelGrid(63, 65, 63; Nt=1, α=0.5, β=1, width=7)
```
"""
function ChannelGrid(Ny::Int, Nz::Int; Nt::Int=1, β::Real=1,
                     lim::NTuple{2, <:Real}=(-1, 1),
                     dist::FDGrids.AbstractGridDistribution=FDGrids.GaussLobattoGrid(),
                     width::Int=5, T::Type{<:Real}=Float64)
    y, Dy, Dy2, Dya, Dy2a, ws = _channel_fd(Ny, lim, dist, width, T)
    return RectangularGrid((y,), (Dy,), (Dy2,), (Dya,), (Dy2a,), (ws,), (β, 2π),
                           (Ny, Nz, Nt), CHANNEL_2D3C_AXES, CHANNEL_2D3C_FFT_ORDER, T)
end

function ChannelGrid(Nx::Int, Ny::Int, Nz::Int; Nt::Int=1, α::Real=1, β::Real=1,
                     lim::NTuple{2, <:Real}=(-1, 1),
                     dist::FDGrids.AbstractGridDistribution=FDGrids.GaussLobattoGrid(),
                     width::Int=5, T::Type{<:Real}=Float64)
    y, Dy, Dy2, Dya, Dy2a, ws = _channel_fd(Ny, lim, dist, width, T)
    return RectangularGrid((y,), (Dy,), (Dy2,), (Dya,), (Dy2a,), (ws,), (α, β, 2π),
                           (Ny, Nx, Nz, Nt), CHANNEL_3D_AXES, CHANNEL_3D_FFT_ORDER, T)
end

ChannelGrid(Ny::Int, Nz::Int, Nt::Int, width::Int, β::Real; kwargs...) =
    ChannelGrid(Ny, Nz; Nt, width, β, kwargs...)

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
velocities `-1` and `1` for either channel layout and any valid `lim`.

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
    rpcf_base(g::AbstractChannel2D3CGrid) -> Vector

Compatibility name for the shared normalized [`plane_couette_base`](@ref) on
a streamwise-independent channel grid.
"""
rpcf_base(g::AbstractChannel2D3CGrid) = plane_couette_base(g)

"""
    rbc_base_temperature(g::AbstractChannel3DGrid) -> Vector

Return the Rayleigh–Bénard conduction profile `Θ(η)=(1-η)/2`, using the same
normalized wall coordinate as the channel velocity profiles. The lower and
upper plates therefore have temperatures one and zero for any valid `lim`.
"""
rbc_base_temperature(g::AbstractChannel3DGrid) = (one(eltype(g)) .- plane_couette_base(g)) ./ 2

# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Plane Couette flow                                                                         // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

"""
    PlaneCouetteFlow(g, Re; Ro=0, base=..., mode=AdjointDiscrete(),
                     fftw_flags=FFTW.EXHAUSTIVE, dealias=true) -> ProjectedNSE

Construct plane Couette equations on either channel layout.

- A full 3D grid selects [`CartesianPrimitive3D`](@ref), state `(u,v,w)`,
  default base `(U,nothing,nothing)`, and Coriolis components `(1,2)`.
- A streamwise-independent grid selects [`CartesianPrimitive2D3C`](@ref),
  state `(v,w,u)`, default base `(nothing,nothing,U)`, and Coriolis components
  `(3,1)`. This is the former RPCF case.

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

function PlaneCouetteFlow(g::AbstractChannel2D3CGrid, Re; Ro=0,
                          base=(nothing, nothing, plane_couette_base(g)), mode=AdjointDiscrete(),
                          fftw_flags=FFTW.EXHAUSTIVE, dealias=true)
    length(base) == 3 || throw(ArgumentError("a 2D3C channel base must contain (V, W, U)"))
    force = iszero(Ro) ? NoForce() : CoriolisForce(eltype(g)(Ro); components=(3, 1))
    return construct_equations(g, Re, base, CartesianPrimitive2D3C(); force, mode, flags=fftw_flags, dealias)
end

# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Plane Poiseuille flow                                                                      // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

"""
    PlanePoiseuilleFlow(g::AbstractChannel3DGrid, Re; Ro=0, f=1,
                        base=(plane_poiseuille_base(g),nothing,nothing),
                        mode=AdjointDiscrete(), fftw_flags=FFTW.EXHAUSTIVE,
                        dealias=true) -> ProjectedNSE

Construct pressure-driven plane Poiseuille equations on a full 3D channel.
`f` applies a [`ConstantBodyForce`](@ref) to streamwise component one at the
zero Fourier mode. Nonzero `Ro` composes it with spanwise-axis Coriolis
forcing. The default base is the normalized laminar profile `U(η)=1-η²`.
"""
function PlanePoiseuilleFlow(g::AbstractChannel3DGrid, Re; Ro=0, f=1,
                             base=(plane_poiseuille_base(g), nothing, nothing), mode=AdjointDiscrete(),
                             fftw_flags=FFTW.EXHAUSTIVE, dealias=true)
    length(base) == 3 || throw(ArgumentError("a 3D channel base must contain (U, V, W)"))
    pressure = ConstantBodyForce(eltype(g)(f); component=1)
    force = iszero(Ro) ? pressure : CompoundForcing(pressure, CoriolisForce(eltype(g)(Ro)))
    return construct_equations(g, Re, base, CartesianPrimitive3D(); force, mode, flags=fftw_flags, dealias)
end

# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Rayleigh–Bénard convection                                                              // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

"""
    RayleighBenardFlow(g::AbstractChannel3DGrid, Re, Pr, Ra;
                       base=(nothing,nothing,nothing,rbc_base_temperature(g)),
                       mode=AdjointDiscrete(), fftw_flags=FFTW.EXHAUSTIVE,
                       dealias=true) -> ProjectedNSE

Construct the full 3D Boussinesq equations for Rayleigh–Bénard convection on a
channel grid. The state is `(u,v,w,θ)`, buoyancy acts on wall-normal component
two, and the formulation uses `Ri=Ra/(Re²Pr)`. The default base is motionless
conduction with plate temperatures one and zero under the normalized channel
coordinate.
"""
function RayleighBenardFlow(g::AbstractChannel3DGrid, Re, Pr, Ra;
                            base=(nothing, nothing, nothing, rbc_base_temperature(g)),
                            mode=AdjointDiscrete(), fftw_flags=FFTW.EXHAUSTIVE, dealias=true)
    length(base) == 4 || throw(ArgumentError("a Rayleigh–Bénard base must contain (U, V, W, Θ)"))
    formulation = CartesianPrimitive3DBoussinesq(Pr, Ra / (Re^2 * Pr); grav=2)
    return construct_equations(g, Re, base, formulation; mode, flags=fftw_flags, dealias)
end
