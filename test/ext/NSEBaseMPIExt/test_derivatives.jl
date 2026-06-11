# Tests for the Val{STORAGE_DIM}-dispatched derivative API in
# `NSEBaseMPIExt/src/derivatives.jl`.
#
# Covers:
#   - `interior_dd!` / `boundary_dd!` cover the local dimension exactly
#   - staged `init_ddy!` / `complete_ddy!` matches the global derivative
#   - spectral directions (`:x`, `:z`, `:t`) bypass the FD path
#   - vector-field derivatives consume per-component halo request tuples
#   - `interior_laplacian!` / `boundary_laplacian!` reproduce the
#     analytic Laplacian after halo exchange
#   - the 2-arg `NSEBase.laplacian!` auto-exchange form agrees

import FDGrids
import LinearAlgebra
import MPI
import NSEBase
import Test

MPI.Initialized() || MPI.Init()

include("grid.jl")

nranks = MPI.Comm_size(MPI.COMM_WORLD)
rank   = MPI.Comm_rank(MPI.COMM_WORLD)

# Single-mode analytic test field. Each FFT direction has exactly one
# non-trivial wavenumber so the spectral derivative is exact for the
# chosen resolution. The wall-normal factor `(1 - y^2)` is a degree-2
# polynomial, exactly differentiable by the 5-point FD stencil.
const α_test = 2π
const β_test = 5.8
const γ_test = 1.0   # temporal wavenumber

u_fun(y, x, z, t)      =  (1 - y^2) * cos(α_test*x) * cos(β_test*z) * cos(γ_test*t)
dudx_fun(y, x, z, t)   = -(1 - y^2) * α_test * sin(α_test*x) * cos(β_test*z) * cos(γ_test*t)
d2udx2_fun(y, x, z, t) = -(1 - y^2) * α_test^2 * cos(α_test*x) * cos(β_test*z) * cos(γ_test*t)
dudy_fun(y, x, z, t)   = -2y * cos(α_test*x) * cos(β_test*z) * cos(γ_test*t)
d2udy2_fun(y, x, z, t) = -2 * cos(α_test*x) * cos(β_test*z) * cos(γ_test*t)
dudz_fun(y, x, z, t)   = -(1 - y^2) * β_test * cos(α_test*x) * sin(β_test*z) * cos(γ_test*t)
d2udz2_fun(y, x, z, t) = -(1 - y^2) * β_test^2 * cos(α_test*x) * cos(β_test*z) * cos(γ_test*t)
dudt_fun(y, x, z, t)   = -(1 - y^2) * γ_test * cos(α_test*x) * cos(β_test*z) * sin(γ_test*t)
lapl_fun(y, x, z, t)   = d2udx2_fun(y, x, z, t) +
                        d2udy2_fun(y, x, z, t) +
                        d2udz2_fun(y, x, z, t)

const Ny = 32; const Nx = 7; const Nz = 9; const Nt = 5
const NHALO = 2  # 5-point stencil -> half-width 2

base_comm = MPI.Comm_dup(MPI.COMM_WORLD)
# Wavenumber scales chosen so that x has period 1/2 (k_x = 2π gives spectral
# coefficients at integer wavenumbers) and z has period 2π/5.8 to match
# `cos(4π*x)` and `cos(5.8*z)` factors in the analytic test field.
g_parent = MockChannelGrid(Ny, Nx, Nz, Nt; stencil_width=5, α=2π, β=5.8)
g = NSEBaseMPIExt.distributed(g_parent, base_comm;
                              decomposed_physical_dims=(:y,), nprocesses=(nranks,), nhalo=(NHALO,))

# Pre-build FFT plans and reusable physical/spectral fields.
plans = NSEBase.FFTPlans(g; dealias=false, flags=NSEBase.FFTW.ESTIMATE)
u_phys = NSEBase.Field(g, u_fun)
u      = NSEBase.FTField(g); plans(u, u_phys)

Test.@testset "staged ddy! matches analytic du/dy" begin
    out = NSEBase.FTField(g)
    requests = NSEBase.init_requests!(u)
    NSEBase.init_ddy!(out, u)
    NSEBase.wait_requests!(requests)
    NSEBase.complete_ddy!(out, u)

    expected = NSEBase.FTField(g)
    plans(expected, NSEBase.Field(g, dudy_fun))

    Test.@test parent(out) ≈ parent(expected) rtol=1e-6
end

Test.@testset "staged VectorField derivative uses per-component halo requests" begin
    q        = NSEBase.VectorField(g, NSEBase.FTField; N=2)
    expected = NSEBase.VectorField(g, NSEBase.FTField; N=2)
    out      = NSEBase.VectorField(g, NSEBase.FTField; N=2)

    plans(q[1], NSEBase.Field(g, u_fun))
    plans(q[2], NSEBase.Field(g, (y, x, z, t) -> 2u_fun(y, x, z, t)))
    plans(expected[1], NSEBase.Field(g, dudy_fun))
    plans(expected[2], NSEBase.Field(g, (y, x, z, t) -> 2dudy_fun(y, x, z, t)))

    requests = NSEBase.init_requests!(q)
    NSEBase.init_ddy!(out, q)
    NSEBase.wait_requests!(requests)
    NSEBase.complete_ddy!(out, q)

    Test.@test parent(out[1]) ≈ parent(expected[1]) rtol=1e-6
    Test.@test parent(out[2]) ≈ parent(expected[2]) rtol=1e-6
end

sd(sym) = NSEBase.physical_to_storage_dim(NSEBase.grid(u), Val(sym))

Test.@testset "interior_dd! + wait + boundary_dd! covers the FD direction" begin
    out = NSEBase.FTField(g)
    requests = NSEBase.init_requests!(u)
    NSEBaseMPIExt.interior_dd!(out, u, sd(:y))
    NSEBase.wait_requests!(requests)
    NSEBaseMPIExt.boundary_dd!(out, u, sd(:y))

    expected = NSEBase.FTField(g)
    plans(expected, NSEBase.Field(g, dudy_fun))
    Test.@test parent(out) ≈ parent(expected) rtol=1e-6
end

Test.@testset "dd! along an FFT direction is spectral (no halo needed)" begin
    out = NSEBase.FTField(g)
    NSEBaseMPIExt.interior_dd!(out, u, sd(:x))
    # Boundary work is a no-op for FFT directions; the result must already match.
    expected = NSEBase.FTField(g)
    plans(expected, NSEBase.Field(g, dudx_fun))
    Test.@test parent(out) ≈ parent(expected) rtol=1e-6

    out_z = NSEBase.FTField(g)
    NSEBaseMPIExt.interior_dd!(out_z, u, sd(:z))
    expected_z = NSEBase.FTField(g)
    plans(expected_z, NSEBase.Field(g, dudz_fun))
    Test.@test parent(out_z) ≈ parent(expected_z) rtol=1e-6

    out_t = NSEBase.FTField(g)
    NSEBaseMPIExt.interior_dd!(out_t, u, sd(:t))
    expected_t = NSEBase.FTField(g)
    plans(expected_t, NSEBase.Field(g, dudt_fun))
    Test.@test parent(out_t) ≈ parent(expected_t) rtol=1e-6
end

Test.@testset "interior_laplacian! + boundary_laplacian! matches analytic Laplacian" begin
    out = NSEBase.FTField(g)
    requests = NSEBase.init_requests!(u)
    NSEBaseMPIExt.interior_laplacian!(out, u)
    NSEBase.wait_requests!(requests)
    NSEBaseMPIExt.boundary_laplacian!(out, u)

    expected = NSEBase.FTField(g)
    plans(expected, NSEBase.Field(g, lapl_fun))
    Test.@test parent(out) ≈ parent(expected) rtol=1e-6
end

Test.@testset "NSEBase.laplacian!(out, u) agrees with the explicit form" begin
    out_a = NSEBase.FTField(g)
    out_b = NSEBase.FTField(g)

    requests = NSEBase.init_requests!(u)
    NSEBaseMPIExt.interior_laplacian!(out_a, u)
    NSEBase.wait_requests!(requests)
    NSEBaseMPIExt.boundary_laplacian!(out_a, u)

    # 2-arg overload — auto-exchange.
    NSEBase.laplacian!(out_b, u)

    Test.@test parent(out_a) ≈ parent(out_b)
end

MPI.free(base_comm)
