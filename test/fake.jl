# ------------- #
# concrete grid #
# ------------- #
struct FakeGrid <: AbstractGrid{Float64, 2, (2,)}
    x::Vector{Float64}
    N::Int
    L::Float64
end

 NSEBase.similar(g::FakeGrid, ::Type{T}=Float64) where {T} = FakeGrid(g.x, g.N, g.L)
    Base.size(g::FakeGrid)                                 = (length(g.x), g.N)
  NSEBase.points(g::FakeGrid; dealias=false)               = dealias ? _grid_dealiased_points(g) : _grid_points(g)
NSEBase.fft_norm(g::FakeGrid)                              = g.N

_grid_points(g)           = (reshape(g.x, :, 1), reshape(collect(range(0, g.L*(1 - 1/g.N), length=g.N)), 1, :))
_grid_dealiased_points(g) = (reshape(g.x, :, 1), reshape(collect(range(0, g.L*(1 - 1/(ceil(Int, 1.5*g.N))), length=ceil(Int, 1.5*g.N))), 1, :))


NSEBase.add_base!(u::VectorField{N, <:FTField{FakeGrid}}, base) where {N} = u


# ----------------- #
# field dot methods #
# ----------------- #
function LinearAlgebra.dot(u::FTField{FakeGrid}, v::FTField{FakeGrid})
    Nx, Ny = size(grid(u))
    sum = 0.0
    for nx in 1:Nx
        sum += 0.5*real(dot(u[nx, 1], v[nx, 1]))
        for ny in 2:((Ny >> 1) + 1)
            sum += real(dot(u[nx, ny], v[nx, ny]))
        end
    end
    return sum
end


# --------- #
# operators #
# --------- #
struct NLE; end
function (eq::NLE)(::Real,
                  u::VectorField{3, <:FTField{FakeGrid}},
                out::VectorField{3, <:FTField{FakeGrid}})
    @. out = 2*u
    return out
end

struct LNE; end
function (eq::LNE)(::Real,
                  v::VectorField{3, <:FTField{FakeGrid}},
                out::VectorField{3, <:FTField{FakeGrid}})
    @. out = 4*v
    return out
end

NSEBase.ProjectedNSE(grid::FakeGrid) = ProjectedNSE(grid, 3, NLE(), LNE(), nothing)


# ----------------------- #
# projected field methods #
# ----------------------- #
NSEBase.no_of_modes(modes::Vector{Array{Complex{T}, 3}}) where {T} = size(modes[1], 2)

function NSEBase.project!(a::ProjectedField{FakeGrid}, u::VectorField{N, <:FTField{FakeGrid}}) where {N}
    a .= 0
    for n in 1:N, ny in axes(a, 2), m in axes(a, 1)
        a[m, ny] += dot(modes(a)[n][:, m, ny], u[n][:, ny])
    end
    return a
end

function NSEBase.expand!(u::VectorField{N, <:FTField{FakeGrid}}, a::ProjectedField{FakeGrid}) where {N}
    u .= 0
    for n in 1:N, ny in axes(a, 2), m in axes(a, 1)
        u[n][:, ny] .+= a[m, ny].*@view(modes(a)[n][:, m, ny])
    end
    return u
end

function LinearAlgebra.dot(a::ProjectedField{FakeGrid}, b::ProjectedField{FakeGrid})
    M, Ny = size(a)
    sum = 0.0
    for m in 1:M
        sum += 0.5*real(dot(a[m, 1], b[m, 1]))
        for ny in 2:((Ny >> 1) + 1)
            sum += real(dot(a[m, ny], b[m, ny]))
        end
    end
    return sum
end
