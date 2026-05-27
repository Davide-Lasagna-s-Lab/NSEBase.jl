# Benchmark: dot(u, v) for FTField and ProjectedField on a 4-D channel-flow
# layout, comparing two loop strategies:
#
#   A  CartesianIndices  — single loop over the full FTField parent array;
#                          column-major (stride-1) access, one branch per rfft index.
#                          Current implementation on better-spectral-loops.
#
#   B  hand-written split loop — rfft branch lifted out; dim-1 (j) innermost
#                          for stride-1 access.
#
# Grid layout (mirrors ChannelGrid from ReSolver-ChannelFlow.jl):
#   physical axes  (x, y, z, t)  →  storage dims  (2, 1, 3, 4)
#   i.e. storage order: (y, x, z, t)  =  (Ny, Nx_half, Nz, Nt)
#   FFT dims: (2=rfft, 3=signed FFT, 4=signed FFT)
#
# Run: julia --project=benchmark benchmark/ftfield_dot.jl

using LinearAlgebra
using Printf
using BenchmarkTools
using NSEBase

# ──────────────────────────────────────────────────────────────────────────────
# Mock grid — minimal concrete subtype carrying only the type parameters and
# the grid size.  No FFTW plans, no differentiation matrices.
# ──────────────────────────────────────────────────────────────────────────────
#
# CHANNEL_AXES      = (2, 1, 3, 4): physical dim k lives in storage dim AXES[k]
# CHANNEL_FFT_ORDER = (2, 3, 4):    rfft on dim 2, signed FFTs on dims 3 and 4
#
const BENCH_AXES      = (2, 1, 3, 4)
const BENCH_FFT_ORDER = (2, 3, 4)

struct BenchGrid{S} <: NSEBase.AbstractGrid{Float64, 4, BENCH_AXES, BENCH_FFT_ORDER}
    ws :: Vector{Float64}   # wall-normal (inhomogeneous) quadrature weights
end

Base.size(g::BenchGrid{S}) where {S} = NSEBase.to_storage_order(S, g)
NSEBase.weights(g::BenchGrid) = g.ws
NSEBase.wavenumber_scale(::BenchGrid, ::Int) = 1.0
NSEBase.points(g::BenchGrid; dealias=false) = ntuple(d -> ones(size(g, d)), 4)

# Minimal ddx! stub (not exercised in dot benchmark)
NSEBase.ddx!(out, u, ::Val; kwargs...) = (out .= 0; out)
NSEBase.inhomogeneous_laplacian!(out, u; kwargs...) = (out .= 0; out)

# ──────────────────────────────────────────────────────────────────────────────
# Grid sizes  (representative channel-flow run: 63³ × 33)
# ──────────────────────────────────────────────────────────────────────────────
#   physical: Nx=63 Ny=33 Nz=63 Nt=63
#   stored:   S = (Nx, Ny, Nz, Nt) in physical-coord order
#   FTField parent size: (Ny, Nx_half, Nz, Nt)  where Nx_half = (Nx>>1)+1 = 32
const Nx = 63
const Ny = 33
const Nz = 63
const Nt = 63
const Nx_half = (Nx >> 1) + 1  # 32

ws = rand(Ny) .+ 0.1   # strictly positive weights (arbitrary, only timing matters)
g  = BenchGrid{(Nx, Ny, Nz, Nt)}(ws)

u = FTField(g)
v = FTField(g)
parent(u) .= randn(ComplexF64, size(parent(u)))
parent(v) .= randn(ComplexF64, size(parent(v)))

@assert size(parent(u)) == (Ny, Nx_half, Nz, Nt)

# ──────────────────────────────────────────────────────────────────────────────
# Approach A: CartesianIndices  (current better-spectral-loops)
# ──────────────────────────────────────────────────────────────────────────────
function dot_cartesian(u, v, ws, g)
    s = 0.0
    @inbounds for I in CartesianIndices(u)
        s += NSEBase.one_or_two(I, g) *
             ws[NSEBase.inhomogeneous_indices(I, g)...] *
             real(conj(u[I]) * v[I])
    end
    return s / 2
end

# ──────────────────────────────────────────────────────────────────────────────
# Approach B: hand-written loop nest matching the channel storage layout.
# Storage: (Ny=dim1, Nx_half=dim2, Nz=dim3, Nt=dim4).
# dim1 (j) is stride-1, so it goes innermost.
# The rfft branch is lifted outside the j-loop to avoid per-element branching.
# Loop order (outermost → innermost): Nt, Nz  →  rfft split  →  j (stride-1)
# ──────────────────────────────────────────────────────────────────────────────
function dot_handwritten(u, v, ws)
    s = 0.0
    # storage: (Ny=dim1, Nx_half=dim2, Nz=dim3, Nt=dim4)
    for it in 1:Nt, iz in 1:Nz               # slowest-varying dims outermost
        for j in 1:Ny                         # ix=1 (rfft zero, weight 1): stride-1
            @inbounds s += ws[j] * real(conj(u[j, 1, iz, it]) * v[j, 1, iz, it])
        end
        for ix in 2:Nx_half                   # rfft positive wavenumbers, weight 2
            for j in 1:Ny                     # innermost: stride-1
                @inbounds s += 2*ws[j] * real(conj(u[j, ix, iz, it]) * v[j, ix, iz, it])
            end
        end
    end
    return s / 2
end

# ──────────────────────────────────────────────────────────────────────────────
# Approach C: NSEBase.dot  (whatever is on the current branch)
# ──────────────────────────────────────────────────────────────────────────────
dot_nsebase(u, v) = dot(u, v)

# ──────────────────────────────────────────────────────────────────────────────
# Correctness
# ──────────────────────────────────────────────────────────────────────────────
ref = dot_cartesian(parent(u), parent(v), ws, g)
let d = abs(dot_handwritten(parent(u), parent(v), ws) - ref)
    @assert d < 1e-6 "hand-written disagrees: |Δ| = $d"
end
let d = abs(dot_nsebase(u, v) - ref)
    @assert d < 1e-6 "NSEBase.dot disagrees: |Δ| = $d"
end
println("Correctness: ok")

# ──────────────────────────────────────────────────────────────────────────────
# ProjectedField benchmark
# ──────────────────────────────────────────────────────────────────────────────
const M = 10   # number of projection modes

# modes tensor: (Ny, M, Nx_half, Nz, Nt) — shape required by ProjectedField
modes = ntuple(_ -> randn(ComplexF64, Ny, M, Nx_half, Nz, Nt), 2)
# ProjectedField data: (M, Nx_half, Nz, Nt)
a_data = randn(ComplexF64, M, Nx_half, Nz, Nt)
b_data = randn(ComplexF64, M, Nx_half, Nz, Nt)
pa = ProjectedField(g, a_data, modes)
pb = ProjectedField(g, b_data, modes)

dot_proj_nsebase(a, b) = dot(a, b)   # NSEBase.dot for ProjectedField

function dot_proj_handwritten(a, b)
    s = 0.0
    ad, bd = parent(a), parent(b)
    # ProjectedField parent shape: (M, Nx_half, Nz, Nt) — rfft dim is index 2
    for it in 1:Nt, iz in 1:Nz
        for m in 1:M
            @inbounds s += real(conj(ad[m, 1, iz, it]) * bd[m, 1, iz, it])
        end
        for ix in 2:Nx_half, m in 1:M
            @inbounds s += 2 * real(conj(ad[m, ix, iz, it]) * bd[m, ix, iz, it])
        end
    end
    return s / 2
end

let d = abs(dot_proj_nsebase(pa, pb) - dot_proj_handwritten(pa, pb))
    @assert d < 1e-6 "ProjectedField dot disagrees: |Δ| = $d"
end
println("ProjectedField correctness: ok")

# ──────────────────────────────────────────────────────────────────────────────
# Benchmarks
# ──────────────────────────────────────────────────────────────────────────────
BenchmarkTools.DEFAULT_PARAMETERS.samples = 50

pu = parent(u); pv = parent(v)   # plain arrays for the raw-loop variants

println("\n── FTField dot  (Nx=$Nx, Ny=$Ny, Nz=$Nz, Nt=$Nt) ─────────────────────")
t_cart = @belapsed dot_cartesian($pu, $pv, $ws, $g)
t_hand = @belapsed dot_handwritten($pu, $pv, $ws)
t_nse  = @belapsed dot_nsebase($u, $v)
@printf "  A  CartesianIndices (raw loop)  : %7.3f ms\n"  t_cart * 1e3
@printf "  B  hand-written split loop      : %7.3f ms\n"  t_hand * 1e3
@printf "  C  NSEBase.dot (FTField)        : %7.3f ms\n"  t_nse  * 1e3
@printf "  ratio C/A : %.3f\n"  (t_nse / t_cart)
@printf "  ratio C/B : %.3f\n"  (t_nse / t_hand)

println("\n── ProjectedField dot  (M=$M, Nx=$Nx, Nz=$Nz, Nt=$Nt) ─────────────────")
t_proj_hand = @belapsed dot_proj_handwritten($pa, $pb)
t_proj_nse  = @belapsed dot_proj_nsebase($pa, $pb)
@printf "  A  hand-written split loop          : %7.3f ms\n"  t_proj_hand * 1e3
@printf "  B  NSEBase.dot (ProjectedField)     : %7.3f ms\n"  t_proj_nse  * 1e3
@printf "  ratio B/A : %.3f\n"  (t_proj_nse / t_proj_hand)
