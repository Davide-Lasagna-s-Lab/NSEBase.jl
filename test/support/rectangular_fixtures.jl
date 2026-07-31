# Shared production-grid constructors for NSEBase's contract tests. Every
# helper returns a ReSolverRectangularGrids grid; this module deliberately
# defines no AbstractGrid subtype and adds no method to NSEBase's interface.

module TestRectangularFixtures

using ReSolverRectangularGrids

export line_grid, planar_grid, channel_grid, cavity2d_grid, cavity3d_grid,
       square_duct_grid, steady_channel_grid, flipped_grid, spacetime_grid

"""
    channel_grid(; Nx=7, Ny=15, Nz=9, Nt=5, α=1, β=1, width=3, T=Float64)

Return a production channel grid stored as `(y, x, z, t)`. The default Fourier
resolutions are odd, and the bounded resolution is large enough for the
default FDGrids stencil and its discrete adjoint.
"""
function channel_grid(; Nx::Int=7, Ny::Int=15, Nz::Int=9, Nt::Int=5,
                      α::Real=1, β::Real=1, width::Int=3,
                      T::Type{<:Real}=Float64)
    return ChannelGrid(Nx, Ny, Nz; Nt, α, β, width, T)
end

"""
    cavity2d_grid(; N=9, Nt=5, lim=(0, 1), width=3, T=Float64)

Return the production square two-dimensional cavity grid stored as `(x,y,t)`.
"""
function cavity2d_grid(; N::Int=9, Nt::Int=5,
                       lim::NTuple{2, <:Real}=(0, 1), width::Int=3,
                       T::Type{<:Real}=Float64)
    return LidDrivenCavity2DGrid(N; Nt, lim, width, T)
end

"""
    cavity3d_grid(; N=9, Nt=5, lim=(0, 1), width=3, T=Float64)

Return the production cubic three-dimensional cavity grid stored as
`(x,y,z,t)`.
"""
function cavity3d_grid(; N::Int=9, Nt::Int=5,
                       lim::NTuple{2, <:Real}=(0, 1), width::Int=3,
                       T::Type{<:Real}=Float64)
    return LidDrivenCavity3DGrid(N; Nt, lim, width, T)
end

"""
    square_duct_grid(; N=9, Nz=9, Nt=5, α=1, width=3, T=Float64)

Return the production square-duct grid stored as `(x,y,z,t)`.
"""
function square_duct_grid(; N::Int=9, Nz::Int=9, Nt::Int=5,
                          α::Real=1, width::Int=3,
                          T::Type{<:Real}=Float64)
    return SquareDuctGrid(N, Nz; Nt, α, width, T)
end

"""
    line_grid(; Nb=9, Nh=9, scale=1, width=3, T=Float64)

Return a two-dimensional production `RectangularGrid` with storage layout
`(bounded, Fourier)`, Cartesian axes `(x,y)`, and one real-to-complex Fourier
transform. This compact layout is useful for scalar-field and one-wavenumber
contracts without introducing a test grid type.
"""
function line_grid(; Nb::Int=9, Nh::Int=9, scale::Real=1, width::Int=3,
                   T::Type{<:Real}=Float64)
    donor = ChannelGrid(Nh, Nb, 1; Nt=1, α=scale, width, T)
    return RectangularGrid(donor.xs, donor.D₁, donor.D₂, donor.D₁⁺,
                           donor.D₂⁺, donor.ws, (scale,), (Nb, Nh),
                           (1, 2, nothing, nothing), (2,), T)
end

"""Alias for [`line_grid`](@ref), emphasizing its bounded-periodic plane."""
planar_grid(; kwargs...) = line_grid(; kwargs...)

"""
    steady_channel_grid(; Nx=9, Ny=9, Nz=7, α=1, β=1, width=3, T=Float64)

Return a three-dimensional production `RectangularGrid` stored as `(y,x,z)`.
It has the spatial channel layout but omits the singleton time dimension.
"""
function steady_channel_grid(; Nx::Int=9, Ny::Int=9, Nz::Int=7,
                             α::Real=1, β::Real=1, width::Int=3,
                             T::Type{<:Real}=Float64)
    donor = ChannelGrid(Nx, Ny, Nz; Nt=1, α, β, width, T)
    return RectangularGrid(donor.xs, donor.D₁, donor.D₂, donor.D₁⁺,
                           donor.D₂⁺, donor.ws, (α, β), (Ny, Nx, Nz),
                           (2, 1, 3, nothing), (2, 3), T)
end

"""
    flipped_grid(; Nx=9, Nz=7, Ny=9, α=1, β=1, width=3, T=Float64)

Return a three-dimensional production `RectangularGrid` stored as `(x,z,y)`,
with both Fourier dimensions before the bounded dimension. It exercises valid
storage permutations through the production grid implementation.
"""
function flipped_grid(; Nx::Int=9, Nz::Int=7, Ny::Int=9,
                      α::Real=1, β::Real=1, width::Int=3,
                      T::Type{<:Real}=Float64)
    donor = ChannelGrid(Nx, Ny, Nz; Nt=1, α, β, width, T)
    return RectangularGrid(donor.xs, donor.D₁, donor.D₂, donor.D₁⁺,
                           donor.D₂⁺, donor.ws, (α, β), (Nx, Nz, Ny),
                           (1, 3, 2, nothing), (1, 2), T)
end

"""
    spacetime_grid(; Nx=9, Ny=9, Nt=7, α=1, width=3, T=Float64)

Return a three-dimensional production `RectangularGrid` stored as `(y,x,t)`.
Physical `z` is absent, while both `x` and `t` are Fourier transformed.
"""
function spacetime_grid(; Nx::Int=9, Ny::Int=9, Nt::Int=7,
                        α::Real=1, width::Int=3,
                        T::Type{<:Real}=Float64)
    donor = ChannelGrid(Nx, Ny, 1; Nt, α, width, T)
    return RectangularGrid(donor.xs, donor.D₁, donor.D₂, donor.D₁⁺,
                           donor.D₂⁺, donor.ws, (α, 2π), (Ny, Nx, Nt),
                           (2, 1, nothing, 3), (2, 3), T)
end
end
