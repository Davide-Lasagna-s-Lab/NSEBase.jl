# Benchmark: advective vs divergence vs rotational advection form, for the
# nonlinear, forward-linearised, and continuous-adjoint 3D operators.
#
# The forms are mathematically equivalent (up to a projection-removed gradient)
# but cost a different number of FFTs per evaluation — the dominant cost in a
# pseudo-spectral / FD-hybrid solver. This times each (role × form) on the CPU on
# a realistic channel grid (x, z Fourier; y wall-normal finite differences) and
# reports the speedup over the advective baseline.
#
# For the linearised operators the base flow is set up once (3-arg call) and the
# 2-arg hot path is timed — this is what an iterative resolvent solver repeats.
#
# Setup (once):
#   julia --project=benchmarks -e 'using Pkg; Pkg.instantiate()'
# Run:
#   julia --project=benchmarks benchmarks/advection_forms.jl

using NSEBase, FDGrids, FFTW, LinearAlgebra, BenchmarkTools, Printf
import NSEBase: AbstractGrid, FTField, Field, VectorField, grid, dd!,
               inhomogeneous_laplacian!, ddx!, ddy!, ddz!, FFT

# ---- self-contained channel grid (x, z Fourier; y wall-normal FD) ---------- #
const BAXES = (2, 1, 3, 4)
const BFFT  = (2, 3, 4)

struct BenchChannelGrid{S, T, M1, M2, W} <: AbstractGrid{T, 4, BAXES, BFFT}
    y::Vector{T}; D₁::M1; D₂::M2; ws::W; α::T; β::T
end
function BenchChannelGrid(Nx, Ny, Nz, Nt; width=5, α=1.0, β=1.0)
    y, ws = FDGrids.grid(Ny, -1.0, 1.0, FDGrids.GaussLobattoGrid())
    D₁, D₂ = FDGrids.DiffMatrix(y, width, 1), FDGrids.DiffMatrix(y, width, 2)
    BenchChannelGrid{(Nx,Ny,Nz,Nt),Float64,typeof(D₁),typeof(D₂),typeof(ws)}(collect(y), D₁, D₂, ws, α, β)
end
Base.size(::BenchChannelGrid{S}) where {S} = (S[2], S[1], S[3], S[4])
NSEBase.weights(g::BenchChannelGrid) = g.ws
@inline NSEBase.wavenumber_scale(g::BenchChannelGrid{S,T}, d::Int) where {S,T} =
    d == BAXES[1] ? g.α : d == BAXES[3] ? g.β : d == BAXES[4] ? T(2π) : one(T)
_eqp(N, L) = collect((0:N-1) ./ N .* L)
function NSEBase.points(g::BenchChannelGrid{S}; dealias=false) where {S}
    Nx, Ny, Nz, Nt = S
    if dealias
        ps = NSEBase.get_padded_size(size(g), NSEBase.fft_storage_dims(g))
        Nx, Nz, Nt = ps[BAXES[1]], ps[BAXES[3]], ps[BAXES[4]]
    end
    r(d, v) = reshape(v, ntuple(k -> k == d ? length(v) : 1, 4))
    (r(BAXES[2], g.y), r(BAXES[1], _eqp(Nx, 2π/g.α)), r(BAXES[3], _eqp(Nz, 2π/g.β)), r(BAXES[4], _eqp(Nt, 1.0)))
end
# The benchmarked roles (NSE / forward / continuous-adjoint) never request the
# adjoint operator, so a forward-only wall-normal derivative suffices.
NSEBase.dd!(out::FTField{<:BenchChannelGrid}, u::FTField{<:BenchChannelGrid}, ::Val{1}; adjoint::Bool=false) =
    (mul!(parent(out), grid(u).D₁, parent(u), Val(1)); out)
NSEBase.inhomogeneous_laplacian!(out::FTField{<:BenchChannelGrid}, u::FTField{<:BenchChannelGrid}; adjoint::Bool=false) =
    (mul!(parent(out), grid(u).D₂, parent(u), Val(1)); out)

# ---- benchmark ------------------------------------------------------------- #
function field(g)
    A = FFT(VectorField(g, (y,x,z,t)->(1-y^2)*exp(cos(x))*sin(z)*cos(2π*t),
                            (y,x,z,t)->(1-y^2)*exp(sin(2x))*cos(z)*sin(2π*t),
                            (y,x,z,t)->(1-y^2)^2*cos(x)*exp(sin(2z))*cos(2π*t)))
    u = VectorField(g); tmp = FTField(g)
    ddy!(u[1],A[3]); ddz!(tmp,A[2]); u[1].-=tmp
    ddz!(u[2],A[1]); ddx!(tmp,A[3]); u[2].-=tmp
    ddx!(u[3],A[2]); ddy!(tmp,A[1]); u[3].-=tmp
    return u
end

const FORMS = [("advective", Advective()), ("divergence", Divergence()), ("rotational", Rotational())]
const CASES = [(33,48,33,17), (49,64,49,17)]

@printf("%-26s %12s %12s %12s   %8s %8s\n", "case (Nx,Ny,Nz,Nt) / role",
        "adv [ms]", "div [ms]", "rot [ms]", "div ×", "rot ×")
println(repeat("-", 92))

for (Nx,Ny,Nz,Nt) in CASES
    g = BenchChannelGrid(Nx, Ny, Nz, Nt)
    u = field(g); base = field(g)

    # nonlinear NSE
    t = map(FORMS) do (_, f)
        op = CartesianPrimitive3DNSE(g, 1000.0; form=f, flags=FFTW.MEASURE)
        @belapsed $op(0.0, $u, $(similar(u))) samples=15 evals=1
    end
    @printf("(%2d,%3d,%2d,%2d) NSE%9s %12.3f %12.3f %12.3f   %7.2f× %7.2f×\n",
            Nx,Ny,Nz,Nt,"", 1e3.*t..., t[1]/t[2], t[1]/t[3])

    # forward + continuous adjoint: set base once (3-arg), time the 2-arg hot path
    for (role, mode) in (("forward", Forward()), ("adjoint-cont", AdjointContinuous()))
        t = map(FORMS) do (_, f)
            op = CartesianPrimitive3DLNSE(g, 1000.0; mode=mode, form=f, flags=FFTW.MEASURE)
            op(0.0, base, u, similar(u))                      # cache base flow
            @belapsed $op(0.0, $u, $(similar(u))) samples=15 evals=1
        end
        @printf("%-26s %12.3f %12.3f %12.3f   %7.2f× %7.2f×\n",
                "             " * role, 1e3.*t..., t[1]/t[2], t[1]/t[3])
    end
end

println()
println("div ×, rot ×: speedup over the advective baseline (higher is better).")
println("Forms agree up to wall-normal FD truncation (and a projection-removed gradient).")
