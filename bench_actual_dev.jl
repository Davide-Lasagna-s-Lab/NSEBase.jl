using BenchmarkTools
using LinearAlgebra
using Printf
using NSEBase

include("test/test_grids.jl")

# Simple 3D grid for testing
struct TestGrid <: AbstractGrid{Float64, 4, (2, 1, 3, 4), (2, 3, 4)}
    Ny::Int; Nx::Int; Nz::Int; Nt::Int; ws::Vector{Float64}
end
Base.size(g::TestGrid) = (g.Ny, g.Nx, g.Nz, g.Nt)
NSEBase.weights(g::TestGrid) = g.ws
NSEBase.wavenumber_scale(::TestGrid, ::Int) = 1.0

Ny, Nx, Nz, Nt = 16, 32, 24, 16
g = TestGrid(Ny, Nx, Nz, Nt, ones(Ny))

u = FTField(g)
parent(u) .= randn(ComplexF64, size(parent(u)))
v = FTField(g)
parent(v) .= randn(ComplexF64, size(parent(v)))

rfft_sz = (Nx >> 1) + 1
Nm = 3
ms = ntuple(_ -> randn(ComplexF64, Nm, Ny, rfft_sz, Nz, Nt), 1)[1]
a = ProjectedField(g, ms)
b = ProjectedField(g, randn(ComplexF64, size(ms)))

# Warm up
for _ in 1:3
    dot(u, v)
    shift!(copy(u), (0.1, 0.1, 0.1))
    dot(a, b)
end

println("Benchmarking dev branch implementations...")
println("Grid size: Ny=$Ny, Nx=$Nx, Nz=$Nz, Nt=$Nt")
println()

println("dot(FTField):")
@time dot(u, v)

println("shift!(FTField):")
@time shift!(copy(u), (0.1, 0.1, 0.1))

println("dot(ProjectedField):")
@time dot(a, b)

println("Done")
