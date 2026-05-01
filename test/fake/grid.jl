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
