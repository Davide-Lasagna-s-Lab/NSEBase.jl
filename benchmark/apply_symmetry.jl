using BenchmarkTools
using PyPlot
using Printf
using NSEBase

# 4D array, all dims RFFT-transformed; H[1] is the rfft dim (half-sized)
const H = (1, 2, 3, 4)

# side lengths 16, 32, 64, 128
sizes = [16, 32, 64, 128]

times_ns  = Float64[]
allocs_b  = Float64[]

for n in sizes
    # spectral shape after rfft along dim 1: (n÷2+1, n, n, n)
    shape = (n÷2 + 1, n, n, n)
    data  = rand(ComplexF64, shape)

    result = @benchmark NSEBase.apply_symmetry!(d, Val(H)) setup=(d = copy(data)) evals=1

    t = median(result).time
    a = median(result).memory
    push!(times_ns, t)
    push!(allocs_b, a)

    @printf("n=%3d  shape=%-20s  time=%8.3f ms  allocs=%6.1f KB\n",
            n, string(shape), t/1e6, a/1024)
end

# ------------------------------------------------------------------ #
# figure                                                               #
# ------------------------------------------------------------------ #
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(sizes, times_ns ./ 1e6, "o-", color="steelblue", linewidth=1.5, markersize=6)
ax1.set_xlabel("n  (side length)")
ax1.set_ylabel("median time  (ms)")
ax1.set_title("apply_symmetry! — runtime")
ax1.set_xscale("log", base=2)
ax1.set_yscale("log")
ax1.set_xticks(sizes)
ax1.xaxis.set_major_formatter(PyPlot.matplotlib.ticker.ScalarFormatter())
ax1.grid(true, which="both", linestyle="--", alpha=0.4)

ax2.plot(sizes, allocs_b ./ 1024, "o-", color="darkorange", linewidth=1.5, markersize=6)
ax2.set_xlabel("n  (side length)")
ax2.set_ylabel("allocations  (KB)")
ax2.set_title("apply_symmetry! — allocations")
ax2.set_xscale("log", base=2)
ax2.set_xticks(sizes)
ax2.xaxis.set_major_formatter(PyPlot.matplotlib.ticker.ScalarFormatter())
ax2.grid(true, which="both", linestyle="--", alpha=0.4)

fig.tight_layout()

outpath = joinpath(@__DIR__, "apply_symmetry.png")
fig.savefig(outpath, dpi=150)
println("\nFigure saved to $outpath")
