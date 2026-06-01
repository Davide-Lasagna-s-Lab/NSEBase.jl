# Plot benchmark comparison results
using JSON
using PyPlot
using Statistics
using Printf

function plot_comparison(dev_file::String, cart_file::String, output_file::String="benchmark_comparison.png")
    dev_data = JSON.parsefile(dev_file)
    cart_data = JSON.parsefile(cart_file)

    # Extract function names and layouts
    functions = dev_data["functions"]
    layouts = dev_data["layouts"]
    sizes = dev_data["sizes"]



    # Create figure with subplots (one per function)
    fig, axes = subplots(length(functions), 1, figsize=(12, 3*length(functions)))
    if length(functions) == 1
        axes = [axes]
    end

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for (fi, fname) in enumerate(functions)
        ax = axes[fi]

        # Collect data for each layout
        for (li, layout) in enumerate(layouts)
            timings_dev = Float64[]
            timings_cart = Float64[]
            nelems_vals = Int[]

            for size_str in sizes
                dev_key = "$(fname)_$(layout)_$(size_str)_elems"
                cart_key = "$(fname)_$(layout)_$(size_str)_elems"

                if haskey(dev_data["timings"], dev_key) && haskey(cart_data["timings"], cart_key)
                    dev_time = dev_data["timings"][dev_key]["time_ns"] / 1000  # Convert to μs
                    cart_time = cart_data["timings"][cart_key]["time_ns"] / 1000

                    # Skip if either is 0 (indicates skipped benchmark)
                    if dev_time > 0 && cart_time > 0
                        push!(timings_dev, dev_time)
                        push!(timings_cart, cart_time)
                        push!(nelems_vals, parse(Int, size_str))
                    end
                end
            end

            if length(timings_dev) > 0
                # Calculate speedup ratio (dev / cart, so > 1 means cart is faster)
                speedups = timings_dev ./ timings_cart
                ax.plot(nelems_vals, speedups, marker="o", label=layout, color=colors[li], linewidth=2)
            end
        end

        ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.3, linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("Total Elements (log scale)")
        ax.set_ylabel("t_dev / t_cart")
        ax.set_title("$fname", fontweight="bold")
        ax.text(0.02, 0.98, "(values > 1 = CartesianIndices faster)",
               transform=ax.transAxes, ha="left", va="top", fontsize=8, style="italic")
        ax.grid(true, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        # For expand!/project!, lower the y-minimum to show data below 1.0
        ymin = (contains(fname, "expand") || contains(fname, "project")) ? 0.75 : 0.9
        ax.set_ylim([ymin, maximum(ax.get_ylim()) * 1.1])

        # Annotate with average speedup
        all_speedups = Float64[]
        for (li, layout) in enumerate(layouts)
            for size_str in sizes
                dev_key = "$(fname)_$(layout)_$(size_str)_elems"
                cart_key = "$(fname)_$(layout)_$(size_str)_elems"
                if haskey(dev_data["timings"], dev_key) && haskey(cart_data["timings"], cart_key)
                    dev_time = dev_data["timings"][dev_key]["time_ns"] / 1000
                    cart_time = cart_data["timings"][cart_key]["time_ns"] / 1000
                    if dev_time > 0 && cart_time > 0
                        push!(all_speedups, dev_time / cart_time)
                    end
                end
            end
        end

        if length(all_speedups) > 0
            avg_speedup = mean(all_speedups)
            ax.text(0.98, 0.95, @sprintf("Avg: %.2f×", avg_speedup),
                   transform=ax.transAxes, ha="right", va="top",
                   bbox=Dict("boxstyle" => "round", "facecolor" => "wheat", "alpha" => 0.8),
                   fontsize=10, fontweight="bold")
        end
    end

    tight_layout()
    savefig(output_file, dpi=150, bbox_inches="tight")
    println("Figure saved to $output_file")
    close(fig)
end

# Main
if length(ARGS) >= 2
    dev_file = ARGS[1]
    cart_file = ARGS[2]
    output_file = length(ARGS) >= 3 ? ARGS[3] : "benchmark_comparison.png"
    plot_comparison(dev_file, cart_file, output_file)
else
    println("Usage: julia plot_comparison.jl <dev.json> <cart.json> [output.png]")
    println()
    println("Example:")
    println("  julia plot_comparison.jl benchmark/output/dev_results.json benchmark/output/cart_results.json")
end
