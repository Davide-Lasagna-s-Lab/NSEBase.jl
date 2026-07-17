# Benchmark and profile the production three-dimensional Cartesian primitive
# Navier--Stokes operators on a channel grid.
#
# Run the timed benchmark from the package root with
#
#   julia --project=benchmark benchmark/cartesian_primitive.jl
#
# Use `--quick` for a short smoke run. To collect a sampling profile instead,
# pass one of `--profile=nse`, `--profile=forward`, or `--profile=adjoint`.

import BenchmarkTools
import Profile
using NSEBase

const QUICK = "--quick" in ARGS
const PROFILE_ARG = findfirst(arg -> startswith(arg, "--profile="), ARGS)
const PROFILE_OPERATOR = isnothing(PROFILE_ARG) ? nothing : Symbol(split(ARGS[PROFILE_ARG], "="; limit=2)[2])
const PROFILE_OPERATORS = (:nse, :forward, :adjoint)

PROFILE_OPERATOR in (nothing, PROFILE_OPERATORS...) ||
    throw(ArgumentError("profile operator must be one of $(join(PROFILE_OPERATORS, ", "))"))

const N = QUICK ? 9 : 31
const Ny = QUICK ? 11 : 33
const g = ChannelGrid(N, Ny, N; Nt=1, α=1, β=1, width=5)

u1(x, y, z, t) = (1 - y^2) * sin(x) * cos(z)
u2(x, y, z, t) = (1 - y^2)^2 * cos(x) * sin(z)
u3(x, y, z, t) = (1 - y^2) * sin(x + z)
v1(x, y, z, t) = (1 - y^2)^2 * cos(2x) * cos(z)
v2(x, y, z, t) = (1 - y^2) * sin(x) * sin(2z)
v3(x, y, z, t) = (1 - y^2)^2 * cos(x - z)

const u = FFT(VectorField(g, u1, u2, u3))
const v = FFT(VectorField(g, v1, v2, v3))
const out_nse = similar(u)
const out_forward = similar(u)
const out_adjoint = similar(u)

const Re = 400.0
const nse = CartesianPrimitive3DNSE(g, Re; flags=FFTW.ESTIMATE)
const forward = CartesianPrimitive3DLNSE(g, Re; mode=Forward(), flags=FFTW.ESTIMATE)
const adjoint = CartesianPrimitive3DLNSE(g, Re; mode=AdjointDiscrete(), flags=FFTW.ESTIMATE)

# Initialise the base-flow caches once so the timed two-argument LNSE calls
# measure only the repeated action used by optimisation and time steppers.
forward(0.0, u, v, out_forward)
adjoint(0.0, u, v, out_adjoint)

const operators = (
    nse = () -> nse(0.0, u, out_nse),
    forward = () -> forward(0.0, v, out_forward),
    adjoint = () -> adjoint(0.0, v, out_adjoint),
)

if isnothing(PROFILE_OPERATOR)
    samples = QUICK ? 3 : 20
    seconds = QUICK ? 0.1 : 5.0
    suite = BenchmarkTools.BenchmarkGroup()
    suite["NSE"] = BenchmarkTools.@benchmarkable $nse(0.0, $u, $out_nse)
    suite["LNSE forward (cached base)"] = BenchmarkTools.@benchmarkable $forward(0.0, $v, $out_forward)
    suite["LNSE adjoint (cached base)"] = BenchmarkTools.@benchmarkable $adjoint(0.0, $v, $out_adjoint)
    display(BenchmarkTools.run(suite; samples, seconds, evals=1, verbose=true))
else
    operator = getproperty(operators, PROFILE_OPERATOR)
    iterations = QUICK ? 3 : 100
    operator()
    Profile.clear()
    Profile.@profile for _ in 1:iterations
        operator()
    end
    Profile.print(; format=:tree, sortedby=:overhead)
end
