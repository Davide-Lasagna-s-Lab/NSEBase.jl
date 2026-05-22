# Benchmark: for_each_index vs for_each_homogeneous_index for FTField dot product.
#
# Channel-flow FTField layout: (Nt, Nx_half, Nz, Ny)
#   axis 1 = Nt   — temporal,    signed FFT  (stride 1, innermost in memory)
#   axis 2 = Nx   — streamwise,  rfft        (only non-negative wavenumbers stored)
#   axis 3 = Nz   — spanwise,    signed FFT
#   axis 4 = Ny   — wall-normal, inhomogeneous (quadrature weights ws[j])
#
# FFT_DIMS_ORDER = (2, 3, 1): rfft dim first, then nz, then nt.
#
# The two approaches generate different loop orders:
#
#   A  for_each_index style:
#      Loops in ascending array-dimension order (dim 1 = Nt innermost).
#      Column-major → stride-1 access on u/v.  Single contiguous block for each
#      spectral dim; rfft weight (_i2==1 ? 1 : 2) introduces one branch per Nt
#      iterations.
#
#   B  for_each_homogeneous_index style:
#      Loops in FFT_DIMS_ORDER order (rfft = Nx_half innermost among homogeneous dims).
#      Inhomogeneous loop (Ny) added manually inside.  Two-block split on Nz and
#      Nt eliminates sign-branch in wavenumber conversion; rfft weight still needs
#      a branch on _i1.  Inner-most memory access stride = Nt (not 1) → less
#      cache-friendly for large Nt.
#
# Run: julia --project=benchmark benchmark/spectralloops_dot.jl

const Nt      = 63             # temporal, signed FFT (odd to avoid Nyquist ambiguity)
const Nx      = 32             # streamwise, rfft
const Nz      = 63             # spanwise, signed FFT (odd)
const Ny      = 5              # wall-normal, inhomogeneous
const Nx_half = (Nx >> 1) + 1 # rfft storage width: 17

u  = randn(ComplexF64, Nt, Nx_half, Nz, Ny)
v  = randn(ComplexF64, Nt, Nx_half, Nz, Ny)
ws = rand(Ny)                  # quadrature weights (positive, arbitrary for timing)


# ──────────────────────────────────────────────────────────────────────────────
# A: for_each_index style
#    Loops ascend by array dimension: dim 1 (Nt) innermost → stride-1 access.
#    All spectral dims use a single contiguous block (no positive/negative split).
#    rfft Hermitian weight (_i2 == 1 ? 1 : 2) introduces one branch per Nt steps.
# ──────────────────────────────────────────────────────────────────────────────
function dot_foreach_index(u, v, ws)
    s = 0.0
    for _i4 in 1:Ny             # wall-normal, outermost
        for _i3 in 1:Nz         # spanwise, full block
            for _i2 in 1:Nx_half # streamwise rfft
                w = _i2 == 1 ? 1 : 2
                for _i1 in 1:Nt  # temporal, innermost (stride 1)
                    @inbounds s += w * ws[_i4] * real(conj(u[_i1,_i2,_i3,_i4]) * v[_i1,_i2,_i3,_i4])
                end
            end
        end
    end
    return s / 2
end


# ──────────────────────────────────────────────────────────────────────────────
# B: for_each_homogeneous_index style
#    Loops follow FFT_DIMS_ORDER = (2, 3, 1): Nx_half (rfft) innermost among homogeneous
#    dims, then Nz (two-block), then Nt (two-block).  Manual Ny loop added inside.
#    Access u[_i_nt, _i_nx, _i_nz, j]: innermost _i_nx → stride Nt (not 1).
# ──────────────────────────────────────────────────────────────────────────────
function dot_foreach_homogeneous_index(u, v, ws)
    s = 0.0
    for _i_nt in 1:(Nt >> 1) + 1        # nt positive block (FFT_DIMS_ORDER[3])
        for _i_nz in 1:(Nz >> 1) + 1    # nz positive block (FFT_DIMS_ORDER[2])
            for _i_nx in 1:Nx_half       # rfft, innermost homogeneous (FFT_DIMS_ORDER[1])
                w = _i_nx == 1 ? 1 : 2
                for j in 1:Ny            # inhomogeneous, added manually
                    @inbounds s += w * ws[j] * real(conj(u[_i_nt,_i_nx,_i_nz,j]) * v[_i_nt,_i_nx,_i_nz,j])
                end
            end
        end
        for _i_nz in (Nz >> 1) + 2:Nz   # nz negative block
            for _i_nx in 1:Nx_half
                w = _i_nx == 1 ? 1 : 2
                for j in 1:Ny
                    @inbounds s += w * ws[j] * real(conj(u[_i_nt,_i_nx,_i_nz,j]) * v[_i_nt,_i_nx,_i_nz,j])
                end
            end
        end
    end
    for _i_nt in (Nt >> 1) + 2:Nt       # nt negative block
        for _i_nz in 1:(Nz >> 1) + 1
            for _i_nx in 1:Nx_half
                w = _i_nx == 1 ? 1 : 2
                for j in 1:Ny
                    @inbounds s += w * ws[j] * real(conj(u[_i_nt,_i_nx,_i_nz,j]) * v[_i_nt,_i_nx,_i_nz,j])
                end
            end
        end
        for _i_nz in (Nz >> 1) + 2:Nz
            for _i_nx in 1:Nx_half
                w = _i_nx == 1 ? 1 : 2
                for j in 1:Ny
                    @inbounds s += w * ws[j] * real(conj(u[_i_nt,_i_nx,_i_nz,j]) * v[_i_nt,_i_nx,_i_nz,j])
                end
            end
        end
    end
    return s / 2
end


# ──────────────────────────────────────────────────────────────────────────────
# B': for_each_homogeneous_index style, Ny loop outermost
#    Same two-block split as B but the inhomogeneous loop is hoisted outside the
#    homogeneous loops.  Access u[_i_nt, _i_nx, _i_nz, j]: innermost _i_nx still
#    has stride Nt, but Ny (outermost) improves reuse across wavenumber blocks.
# ──────────────────────────────────────────────────────────────────────────────
function dot_foreach_homogeneous_index_ny_outer(u, v, ws)
    s = 0.0
    for j in 1:Ny                            # wall-normal, hoisted outermost
        wj = ws[j]
        for _i_nt in 1:(Nt >> 1) + 1
            for _i_nz in 1:(Nz >> 1) + 1
                for _i_nx in 1:Nx_half
                    w = _i_nx == 1 ? 1 : 2
                    @inbounds s += w * wj * real(conj(u[_i_nt,_i_nx,_i_nz,j]) * v[_i_nt,_i_nx,_i_nz,j])
                end
            end
            for _i_nz in (Nz >> 1) + 2:Nz
                for _i_nx in 1:Nx_half
                    w = _i_nx == 1 ? 1 : 2
                    @inbounds s += w * wj * real(conj(u[_i_nt,_i_nx,_i_nz,j]) * v[_i_nt,_i_nx,_i_nz,j])
                end
            end
        end
        for _i_nt in (Nt >> 1) + 2:Nt
            for _i_nz in 1:(Nz >> 1) + 1
                for _i_nx in 1:Nx_half
                    w = _i_nx == 1 ? 1 : 2
                    @inbounds s += w * wj * real(conj(u[_i_nt,_i_nx,_i_nz,j]) * v[_i_nt,_i_nx,_i_nz,j])
                end
            end
            for _i_nz in (Nz >> 1) + 2:Nz
                for _i_nx in 1:Nx_half
                    w = _i_nx == 1 ? 1 : 2
                    @inbounds s += w * wj * real(conj(u[_i_nt,_i_nx,_i_nz,j]) * v[_i_nt,_i_nx,_i_nz,j])
                end
            end
        end
    end
    return s / 2
end


# ──────────────────────────────────────────────────────────────────────────────
# Correctness
# ──────────────────────────────────────────────────────────────────────────────
ref = dot_foreach_index(u, v, ws)
for (name, f) in [("B  homogeneous_index",          dot_foreach_homogeneous_index),
                  ("B' homogeneous_index, Ny outer", dot_foreach_homogeneous_index_ny_outer)]
    val  = f(u, v, ws)
    err  = abs(val - ref)
    @assert err < 1e-8 "$name disagrees: err=$err"
end
println("Correctness: ok")


# ──────────────────────────────────────────────────────────────────────────────
# Timing
# ──────────────────────────────────────────────────────────────────────────────
function bench(f, args...; warmup=10, reps=500)
    for _ in 1:warmup; f(args...); end
    t = @elapsed for _ in 1:reps; f(args...); end
    return t / reps
end

t_a  = bench(dot_foreach_index,                    u, v, ws)
t_b  = bench(dot_foreach_homogeneous_index,         u, v, ws)
t_b2 = bench(dot_foreach_homogeneous_index_ny_outer, u, v, ws)

println("\nResults (Nt=$Nt, Nx=$Nx, Nz=$Nz, Ny=$Ny):")
println("  A   for_each_index           (col-major, stride-1)  : $(round(t_a *1e6, digits=2)) μs")
println("  B   for_each_homogeneous_index (Nx inner, stride-Nt) : $(round(t_b *1e6, digits=2)) μs")
println("  B'  for_each_homogeneous_index, Ny hoisted outer     : $(round(t_b2*1e6, digits=2)) μs")
println("  ratio B /A : $(round(t_b /t_a, digits=3))")
println("  ratio B'/A : $(round(t_b2/t_a, digits=3))")
