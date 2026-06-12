# Helper grid for the generic-interface test suites.
#
# `FakeGrid` (in fake.jl) is a 2-D grid with one inhomogeneous and one rfft
# dimension.  Several tests want a richer layout with multiple homogeneous
# dimensions of mixed kind (rfft + signed FFT) so we can check the
# convention-sensitive code paths in shifts, norms, weighting and
# derivatives.  `TripleGrid` provides exactly that.


# 3-D grid layout:
#
#   array dim 1: inhomogeneous (y), length Ny — wall-normal-like.
#   array dim 2: rfft           (x), length Nx, period 2π/α  (FFT_DIMS_ORDER[1]).
#   array dim 3: signed FFT     (z), length Nz, period 2π/β  (FFT_DIMS_ORDER[2]).
#
# `AXES = (2, 1, 3, nothing)` means coordinate 1 (x) → array dim 2,
# coordinate 2 (y) → array dim 1, coordinate 3 (z) → array dim 3, and
# coordinate 4 (t) is absent.  Cartesian grid test helpers keep all four
# coordinate slots so wrappers such as `ddx_4!` can dispatch to a no-op for
# grids with no time direction.
#
# `points(g)` still returns only `D == 3` arrays, in storage order:
# `(y, x, z)`.
struct TripleGrid <: AbstractGrid{Float64, 3, (2, 1, 3, nothing), (2, 3)}
    Ny :: Int
    Nx :: Int
    Nz :: Int
    α  :: Float64        # 2π/Lx
    β  :: Float64        # 2π/Lz
    ws :: Vector{Float64}
end

# Construct with unit weights by default — sufficient for tests that don't
# depend on a particular quadrature rule.
function TripleGrid(Ny, Nx, Nz; α=1.0, β=1.0)
    TripleGrid(Ny, Nx, Nz, α, β, ones(Ny))
end

Base.size(g::TripleGrid)                                  = (g.Ny, g.Nx, g.Nz)
NSEBase.weights(g::TripleGrid)                            = g.ws
NSEBase.wavenumber_scale(g::TripleGrid, dim::Int)         = dim == 2 ? g.α :
                                                            dim == 3 ? g.β :
                                                            one(g.α)
Base.convert(::Type{Float64}, g::TripleGrid)              = g

# `growto` increases the homogeneous resolution while keeping the inhomogeneous
# size (Ny) and the wavenumber scales (α, β) fixed.
function NSEBase.growto(g::TripleGrid, target_size::NTuple{2, Int})
    Nx_new, Nz_new = target_size
    return TripleGrid(g.Ny, Nx_new, Nz_new, g.α, g.β, g.ws)
end

# Broadcastable coordinate arrays in storage order.  The inhomogeneous
# direction is exposed as a custom non-uniform `g.y` vector; the two
# homogeneous directions cover their full periods with equally-spaced points.
NSEBase.points(g::TripleGrid; dealias=false) = begin
    Ny, Nx, Nz = size(g)
    y = reshape(range(-1, 1, length=Ny) |> collect, Ny, 1, 1)
    x = reshape((0:Nx-1) * (2π/g.α/Nx),             1, Nx, 1)
    z = reshape((0:Nz-1) * (2π/g.β/Nz),             1,  1, Nz)
    return (y, x, z)
end


# 2-D analytic fixture:
#
#   array dim 1: inhomogeneous y, represented by arbitrary collocation points.
#   array dim 2: rfft x, periodic with length Lx.
#
# The derivative matrices are Lagrange-collocation differentiation matrices.
# They differentiate polynomials of degree < Ny exactly at the grid points,
# which lets tests compare `ddx_2!` and `laplacian!` against analytic
# derivatives without depending on ChannelFlow's Chebyshev grid.
struct PolynomialGrid <: AbstractGrid{Float64, 2, (2, 1, nothing, nothing), (2,)}
    y  :: Vector{Float64}
    Nx :: Int
    Lx :: Float64
    D1 :: Matrix{Float64}
    D2 :: Matrix{Float64}
    ws :: Vector{Float64}
end

function PolynomialGrid(y::AbstractVector{<:Real}, Nx::Integer, Lx::Real=2π)
    yv = Float64.(collect(y))
    D1 = _lagrange_derivative_matrix(yv)
    D2 = D1 * D1
    return PolynomialGrid(yv, Int(Nx), Float64(Lx), D1, D2, ones(length(yv)))
end

function _lagrange_derivative_matrix(x::AbstractVector{<:Real})
    N = length(x)
    λ = ones(Float64, N)
    for j in 1:N, m in 1:N
        j != m && (λ[j] /= x[j] - x[m])
    end

    D = zeros(Float64, N, N)
    for i in 1:N, j in 1:N
        i != j && (D[i, j] = λ[j] / (λ[i] * (x[i] - x[j])))
    end
    for i in 1:N
        D[i, i] = -sum(D[i, j] for j in 1:N if j != i)
    end
    return D
end

Base.size(g::PolynomialGrid)                         = (length(g.y), g.Nx)
NSEBase.weights(g::PolynomialGrid)                   = g.ws
NSEBase.wavenumber_scale(g::PolynomialGrid, dim::Int) = dim == 2 ? 2π / g.Lx : one(g.Lx)
Base.convert(::Type{Float64}, g::PolynomialGrid)     = g

NSEBase.points(g::PolynomialGrid; dealias=false) = begin
    Nx = dealias ? NSEBase.get_padded_size(size(g), NSEBase.fft_storage_dims(g))[2] : g.Nx
    y = reshape(g.y, :, 1)
    x = reshape((0:Nx-1) * (g.Lx / Nx), 1, :)
    return (y, x)
end

function NSEBase.dd!(out::FTField{PolynomialGrid},
                     u::FTField{PolynomialGrid},
                     ::Val{1};
                     adjoint::Bool=false)
    LinearAlgebra.mul!(parent(out), adjoint ? grid(u).D1' : grid(u).D1, parent(u))
    return out
end

function NSEBase.inhomogeneous_laplacian!(out::FTField{PolynomialGrid},
                                          u::FTField{PolynomialGrid};
                                          adjoint::Bool=false)
    LinearAlgebra.mul!(parent(out), adjoint ? grid(u).D2' : grid(u).D2, parent(u))
    return out
end


# Mock channel grid: a real 3-D channel layout (x, z Fourier; y wall-normal)
# with a genuine Lagrange-collocation wall-normal derivative, used to test the
# advection-form variants. Unlike FakeGrid/TripleGrid (which fake the wall-normal
# derivative as identity), this distinguishes the forms — they differ precisely
# in their finite-difference treatment of the inhomogeneous direction.
#
#   array dim 1: inhomogeneous (y), length Ny — real collocation derivative.
#   array dim 2: rfft           (x), length Nx, period 2π/α.
#   array dim 3: signed FFT     (z), length Nz, period 2π/β.
#
# Unit weights ⇒ the discrete adjoint of the wall-normal derivative is the plain
# matrix transpose (D'), so discrete-adjoint identities hold to machine precision.
struct ChannelMockGrid <: AbstractGrid{Float64, 3, (2, 1, 3, nothing), (2, 3)}
    y  :: Vector{Float64}
    Nx :: Int
    Nz :: Int
    α  :: Float64
    β  :: Float64
    D1 :: Matrix{Float64}
    D2 :: Matrix{Float64}
    ws :: Vector{Float64}
end

function ChannelMockGrid(Ny, Nx, Nz; α=1.0, β=1.0)
    # Chebyshev-Gauss-Lobatto nodes on [-1, 1]: well-conditioned collocation
    # derivative (the natural channel wall-normal grid).
    y  = [-cospi((j - 1) / (Ny - 1)) for j in 1:Ny]
    D1 = _lagrange_derivative_matrix(y)
    D2 = D1 * D1
    return ChannelMockGrid(y, Nx, Nz, α, β, D1, D2, ones(Ny))
end

Base.size(g::ChannelMockGrid)                        = (length(g.y), g.Nx, g.Nz)
NSEBase.weights(g::ChannelMockGrid)                  = g.ws
NSEBase.wavenumber_scale(g::ChannelMockGrid, d::Int) = d == 2 ? g.α : d == 3 ? g.β : one(g.α)
Base.convert(::Type{Float64}, g::ChannelMockGrid)    = g
NSEBase.growto(g::ChannelMockGrid, (Nx, Nz)::NTuple{2, Int}) =
    ChannelMockGrid(g.y, Nx, Nz, g.α, g.β, g.D1, g.D2, g.ws)

NSEBase.points(g::ChannelMockGrid; dealias=false) = begin
    Ny, Nx, Nz = size(g)
    if dealias
        ps = NSEBase.get_padded_size(size(g), NSEBase.fft_storage_dims(g))
        Nx, Nz = ps[2], ps[3]
    end
    y = reshape(g.y,                       Ny, 1, 1)
    x = reshape((0:Nx-1) .* (2π/g.α/Nx),   1, Nx, 1)
    z = reshape((0:Nz-1) .* (2π/g.β/Nz),   1, 1, Nz)
    return (y, x, z)
end

# Apply the dense wall-normal matrix along storage dim 1 by reshaping all
# trailing dims into columns (dim 1 is contiguous, so the reshape is free).
function _mul_dim1!(out, D, u)
    Ny = size(parent(u), 1)
    LinearAlgebra.mul!(reshape(parent(out), Ny, :), D, reshape(parent(u), Ny, :))
    return out
end
NSEBase.dd!(out::FTField{<:ChannelMockGrid}, u::FTField{<:ChannelMockGrid},
            ::Val{1}; adjoint::Bool=false) =
    _mul_dim1!(out, adjoint ? grid(u).D1' : grid(u).D1, u)
NSEBase.inhomogeneous_laplacian!(out::FTField{<:ChannelMockGrid}, u::FTField{<:ChannelMockGrid};
                                 adjoint::Bool=false) =
    _mul_dim1!(out, adjoint ? grid(u).D2' : grid(u).D2, u)


# Minimal grids used by several generic tests.  They intentionally implement
# only `size`; tests that use them exercise code paths that need no concrete
# coordinate arrays, weights, or derivative extensions.
struct SpectralTestGrid{S, D, AXES, FFT_DIMS_ORDER} <: AbstractGrid{Float64, D, AXES, FFT_DIMS_ORDER} end
Base.size(::SpectralTestGrid{S}) where {S} = S

struct GalerkinGrid{S} <: AbstractGrid{Float64, 2, (1, 2, nothing, nothing), (2,)}
    ws::Vector{Float64}
end

Base.size(::GalerkinGrid{S}) where {S} = S
NSEBase.weights(g::GalerkinGrid) = g.ws
NSEBase.wavenumber_scale(::GalerkinGrid, ::Int) = 1.0


# "Flipped" 3-D grid: FFT dimensions come FIRST in storage, inhomogeneous LAST.
#
#   array dim 1: rfft           (x), length Nx  — FFT_DIMS_ORDER[1]
#   array dim 2: signed FFT     (z), length Nz  — FFT_DIMS_ORDER[2]
#   array dim 3: inhomogeneous  (y), length Ny
#
# AXES = (1, 3, 2, nothing): coord x → dim 1, coord y → dim 3, coord z → dim 2.
# FFT_DIMS_ORDER = (1, 2).
#
# The `transform_size` of this grid is ((Nx>>1)+1, Nz, Ny), so FTField storage
# is (rfft, z, y) — opposite of TripleGrid.  Used to test that `homogeneous_axes`
# and `inhomogeneous_axes` work correctly regardless of which dimensions come first.
struct FlippedGrid <: AbstractGrid{Float64, 3, (1, 3, 2, nothing), (1, 2)}
    Nx :: Int
    Nz :: Int
    Ny :: Int
    ws :: Vector{Float64}
end

function FlippedGrid(Nx, Nz, Ny)
    FlippedGrid(Nx, Nz, Ny, ones(Ny))
end

# size(g) returns (Nx, Nz, Ny) — storage-order sizes.
Base.size(g::FlippedGrid) = (g.Nx, g.Nz, g.Ny)
NSEBase.weights(g::FlippedGrid)              = g.ws
NSEBase.wavenumber_scale(::FlippedGrid, ::Int) = 1.0
