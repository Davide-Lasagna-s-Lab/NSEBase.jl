# Barebone benchmark: @generated inline body (ddx! style) vs for_each_wavenumber closure.
#
# FTField channel-flow layout: (Nt, Nx_half, Nz, Ny)
#   fft_dims = (nx=2, nz=3, nt=1)  →  nt is a signed FFT dim in array axis 1.
#
# Operation: in-place temporal derivative,  out = im * nt * scale * u
#
# The key difference between the two approaches:
#   A  ddx! style: split-block @generated loop — nt is computed branch-free
#      because each block covers either the positive or negative half of nt.
#   B  for_each_wavenumber style: single pass — the closure receives the raw FFTW
#      index and must recover the signed wavenumber with a conditional.
#   C  for_each_wavenumber style: with the split-block structure retained and signed
#      wavenumber pre-computed per block (i.e. for_each_wavenumber extended to pass
#      both FFTW index AND signed wavenumber to f).
#
# Run: julia perf/benchmark_for_each_wavenumber_vs_ddx.jl

const Nt      = 63
const Nx      = 32
const Nz      = 63
const Ny      = 5
const Nx_half = (Nx >> 1) + 1
const scale   = 2π

u   = randn(ComplexF64, Nt, Nx_half, Nz, Ny)
out = similar(u)

# ──────────────────────────────────────────────────────────────────────────────
# A: split-block, inline body  (exact ddx! style)
#    nt dimension (axis 1) is split; the wavenumber formula is specialised
#    per block, so no conditional appears in the hot path.
# ──────────────────────────────────────────────────────────────────────────────
function ddx_splitblock!(out, u)
    for _i4 in 1:Ny, _i3 in 1:Nz, _i2 in 1:Nx_half
        for _i1 in 1:(Nt >> 1) + 1          # nt ≥ 0
            nt = _i1 - 1
            @inbounds out[_i1, _i2, _i3, _i4] = im * nt * scale * u[_i1, _i2, _i3, _i4]
        end
        for _i1 in (Nt >> 1) + 2:Nt         # nt < 0
            nt = _i1 - Nt - 1
            @inbounds out[_i1, _i2, _i3, _i4] = im * nt * scale * u[_i1, _i2, _i3, _i4]
        end
    end
    return out
end

# ──────────────────────────────────────────────────────────────────────────────
# B: single-pass, conditional wavenumber  (current for_each_wavenumber + closure)
#    for_each_wavenumber passes only the FFTW index; the closure must recover the
#    signed wavenumber with a branch.
# ──────────────────────────────────────────────────────────────────────────────
function ddx_conditional!(out, u)
    for _i4 in 1:Ny, _i3 in 1:Nz, _i2 in 1:Nx_half, _i1 in 1:Nt
        nt = _i1 <= (Nt >> 1) + 1 ? _i1 - 1 : _i1 - Nt - 1
        @inbounds out[_i1, _i2, _i3, _i4] = im * nt * scale * u[_i1, _i2, _i3, _i4]
    end
    return out
end

# ──────────────────────────────────────────────────────────────────────────────
# C: split-block, closure  (for_each_wavenumber extended to also supply signed nt)
#    Retains the split-block structure of A but invokes the body via a function
#    call — isolates closure overhead from conditional overhead.
# ──────────────────────────────────────────────────────────────────────────────
@inline function _body_c!(out, u, _i1, _i2, _i3, _i4, nt)
    @inbounds out[_i1, _i2, _i3, _i4] = im * nt * scale * u[_i1, _i2, _i3, _i4]
end

function ddx_closure_splitblock!(out, u)
    for _i4 in 1:Ny, _i3 in 1:Nz, _i2 in 1:Nx_half
        for _i1 in 1:(Nt >> 1) + 1
            _body_c!(out, u, _i1, _i2, _i3, _i4, _i1 - 1)
        end
        for _i1 in (Nt >> 1) + 2:Nt
            _body_c!(out, u, _i1, _i2, _i3, _i4, _i1 - Nt - 1)
        end
    end
    return out
end

# ──────────────────────────────────────────────────────────────────────────────
# Correctness
# ──────────────────────────────────────────────────────────────────────────────
ddx_splitblock!(out, u)
ref = copy(out)
for (name, f!) in [("B conditional", ddx_conditional!), ("C closure+split", ddx_closure_splitblock!)]
    f!(out, u)
    err = maximum(abs, out .- ref)
    @assert err < 1e-14 "$name disagrees: err=$err"
end
println("Correctness: ok")

# ──────────────────────────────────────────────────────────────────────────────
# Timing
# ──────────────────────────────────────────────────────────────────────────────
function bench(f!, out, u; warmup=10, reps=500)
    for _ in 1:warmup; f!(out, u); end
    t = @elapsed for _ in 1:reps; f!(out, u); end
    return t / reps
end

t_a = bench(ddx_splitblock!,        out, u)
t_b = bench(ddx_conditional!,       out, u)
t_c = bench(ddx_closure_splitblock!, out, u)

println("\nResults (Nt=$Nt, Nx=$Nx, Nz=$Nz, Ny=$Ny):")
println("  A  split-block inline      (ddx! style)          : $(round(t_a*1e6, digits=2)) μs")
println("  B  single-pass conditional (for_each_wavenumber)       : $(round(t_b*1e6, digits=2)) μs")
println("  C  split-block closure     (for_each_wavenumber+wnum)  : $(round(t_c*1e6, digits=2)) μs")
println("  ratio B/A : $(round(t_b/t_a, digits=3))   (conditional cost vs inline)")
println("  ratio C/A : $(round(t_c/t_a, digits=3))   (closure overhead vs inline)")
