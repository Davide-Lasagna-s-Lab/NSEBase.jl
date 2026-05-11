# Correctness and timing comparison: for_each_mode-based ddx! / add_homogeneous_laplacian!
# vs the existing @generated split-block implementations.
#
# Array layout: (Nt, Nx, Nz, Ny) — matches AbstractChannelGrid
#   H = (2, 3, 1): nx rfft (dim 2), nz signed FFT (dim 3), nt signed FFT (dim 1)
#
# Run from NSEBase.jl root:
#   julia test/test_ddx_feachmode.jl

using NSEBase, LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "derivatives_feachmode.jl"))

# ─────────────────────────────────────────────────────────────────────────────
# Minimal test grid — same type params as AbstractChannelGrid
# ─────────────────────────────────────────────────────────────────────────────
struct TestChannelGrid{Nt, Nx, Nz, Ny} <:
    NSEBase.AbstractGrid{Float64, 4, (2, 4, 3, 1), (2, 3), 1}
end

Base.size(::TestChannelGrid{Nt, Nx, Nz, Ny}) where {Nt, Nx, Nz, Ny} = (Nt, Nx, Nz, Ny)

NSEBase.wavenumber_scale(::TestChannelGrid, dim::Int) =
    dim == 2 ? 2π : dim == 3 ? 1.5π : 1.0   # α (nx), β (nz), 1 (nt)

# ─────────────────────────────────────────────────────────────────────────────
# Correctness
# ─────────────────────────────────────────────────────────────────────────────
println("=== Correctness ===")

# Use small odd Nt/Nz so the negative-half blocks are non-trivial.
g   = TestChannelGrid{7, 8, 7, 4}()
u   = FTField(g)
parent(u) .= randn(ComplexF64, size(parent(u)))

out_ref = similar(u)
out_new = similar(u)

for (label, dim) in [("nt  (dim 1)", 1), ("nx  (dim 2)", 2), ("nz  (dim 3)", 3)]
    v = Val(dim)
    for adj in (false, true)
        ddx!(out_ref, u, v; adjoint=adj)
        ddx_fem!(out_new, u, v; adjoint=adj)
        err = maximum(abs, parent(out_ref) .- parent(out_new))
        tag = adj ? " adjoint" : "        "
        @assert err < 1e-14 "ddx! $label$tag: err=$err"
        println("  ddx! $label$tag: ok (err=$err)")
    end
end

# add_homogeneous_laplacian! adds to out, so initialise both copies identically.
parent(out_ref) .= parent(u)
parent(out_new) .= parent(u)
add_homogeneous_laplacian!(out_ref, u)
add_homogeneous_laplacian_fem!(out_new, u)
err = maximum(abs, parent(out_ref) .- parent(out_new))
@assert err < 1e-14 "add_homogeneous_laplacian!: err=$err"
println("  add_homogeneous_laplacian!      : ok (err=$err)")

println("All correctness checks passed.\n")

# ─────────────────────────────────────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────────────────────────────────────
println("=== Timing (channel-flow resolution) ===")

g_big = TestChannelGrid{63, 32, 63, 5}()
u_big = FTField(g_big)
parent(u_big) .= randn(ComplexF64, size(parent(u_big)))
o1 = similar(u_big)
o2 = similar(u_big)

function bench(f!, args...; warmup=5, reps=200)
    for _ in 1:warmup; f!(args...); end
    t = @elapsed for _ in 1:reps; f!(args...); end
    return t / reps
end

for (label, dim) in [("nt  (dim 1)", 1), ("nx  (dim 2)", 2), ("nz  (dim 3)", 3)]
    t_ref = bench(ddx!, o1, u_big, Val(dim))
    t_new = bench(ddx_fem!, o2, u_big, Val(dim))
    println("  ddx! $label : ref=$(round(t_ref*1e6, digits=1)) μs  " *
            "fem=$(round(t_new*1e6, digits=1)) μs  " *
            "ratio=$(round(t_new/t_ref, digits=3))")
end

t_ref = bench(add_homogeneous_laplacian!, o1, u_big)
t_new = bench(add_homogeneous_laplacian_fem!, o2, u_big)
println("  add_homogeneous_laplacian! : ref=$(round(t_ref*1e6, digits=1)) μs  " *
        "fem=$(round(t_new*1e6, digits=1)) μs  " *
        "ratio=$(round(t_new/t_ref, digits=3))")
