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
# `AXES = (2, 1, 3)` means coordinate 1 (x) → array dim 2, coordinate 2 (y) →
# array dim 1, coordinate 3 (z) → array dim 3.  Therefore `points(g)` returns
# `(y, x, z)` — the storage-order traversal of the coordinate triple.
# TODO: AXES should be a tuple of 4 elements
struct TripleGrid <: AbstractGrid{Float64, 3, (2, 1, 3), (2, 3)}
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
