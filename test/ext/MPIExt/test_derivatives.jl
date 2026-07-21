# Tests for the Val{STORAGE_DIM}-dispatched derivative API in
# `MPIExt/src/derivatives.jl`.
#
# Covers:
#   - `interior_dd!` / `boundary_dd!` cover the local dimension exactly
#   - staged `init_ddy!` / `complete_ddy!` matches the global derivative
#   - spectral directions (`:x`, `:z`, `:t`) bypass the FD path
#   - vector-field derivatives consume per-component halo request tuples
#   - `interior_laplacian!` / `boundary_laplacian!` reproduce the
#     analytic Laplacian after halo exchange
#   - the 2-arg `NSEBase.laplacian!` auto-exchange form agrees

using Test

import MPI

using NSEBase

const MPIExt = Base.get_extension(NSEBase, :MPIExt)

MPI.Initialized() || MPI.Init()

include("helpers.jl")

nranks = MPI.Comm_size(MPI.COMM_WORLD)
rank   = MPI.Comm_rank(MPI.COMM_WORLD)

# Single-mode analytic test field. Each FFT direction has exactly one
# non-trivial wavenumber so the spectral derivative is exact for the
# chosen resolution. The wall-normal factor `(1 - y^2)` is a degree-2
# polynomial, exactly differentiable by the 5-point FD stencil.
const α_test = 2π
const β_test = 5.8
const γ_test = 2π   # unit-period temporal wavenumber

u_fun(x, y, z, t)      =  (1 - y^2) * cos(α_test*x) * cos(β_test*z) * cos(γ_test*t)
dudx_fun(x, y, z, t)   = -(1 - y^2) * α_test * sin(α_test*x) * cos(β_test*z) * cos(γ_test*t)
d2udx2_fun(x, y, z, t) = -(1 - y^2) * α_test^2 * cos(α_test*x) * cos(β_test*z) * cos(γ_test*t)
dudy_fun(x, y, z, t)   = -2y * cos(α_test*x) * cos(β_test*z) * cos(γ_test*t)
d2udy2_fun(x, y, z, t) = -2 * cos(α_test*x) * cos(β_test*z) * cos(γ_test*t)
dudz_fun(x, y, z, t)   = -(1 - y^2) * β_test * cos(α_test*x) * sin(β_test*z) * cos(γ_test*t)
d2udz2_fun(x, y, z, t) = -(1 - y^2) * β_test^2 * cos(α_test*x) * cos(β_test*z) * cos(γ_test*t)
dudt_fun(x, y, z, t)   = -(1 - y^2) * γ_test * cos(α_test*x) * cos(β_test*z) * sin(γ_test*t)
lapl_fun(x, y, z, t)   = d2udx2_fun(x, y, z, t) +
                        d2udy2_fun(x, y, z, t) +
                        d2udz2_fun(x, y, z, t)

const Ny = 32; const Nx = 7; const Nz = 9; const Nt = 5
const NHALO = 2  # 5-point stencil -> half-width 2

base_comm = MPI.Comm_dup(MPI.COMM_WORLD)
# Wavenumber scales give x unit period, z period 2π/5.8, and production
# channel time unit period. Each trigonometric factor is therefore mode one.
g_parent = test_channel_grid(Ny, Nx, Nz, Nt; stencil_width=5, α=2π, β=5.8)
g = distributed(g_parent, base_comm;
                decomposed_physical_dims=(:y,), nprocesses=(nranks,), nhalo=(NHALO,))

# Pre-build FFT plans and reusable physical/spectral fields.
plans = NSEBase.FFTPlans(g; dealias=false, flags=NSEBase.FFTW.ESTIMATE)
u_phys = NSEBase.Field(g, u_fun)
u      = NSEBase.FTField(g); plans(u, u_phys)
u_serial = FFT(Field(g_parent, u_fun))
y_storage_dim = storage_dim(g, :y)
y_first = MPIExt.global_first_index(g, y_storage_dim)
local_y_indices = y_first:(y_first + size(g, y_storage_dim) - 1)

@testset verbose=true "staged ddy! matches analytic du/dy                          " begin
    out = NSEBase.FTField(g)
    NSEBase.ddy!(out, u)

    expected = NSEBase.FTField(g)
    plans(expected, NSEBase.Field(g, dudy_fun))

    @test parent(out) ≈ parent(expected) rtol=1e-6

    adjoint_out = NSEBase.ddy!(NSEBase.FTField(g), u; adjoint=true)
    serial_adjoint = NSEBase.ddy!(NSEBase.FTField(g_parent), u_serial; adjoint=true)
    @test parent(adjoint_out) ≈ selectdim(parent(serial_adjoint), y_storage_dim, local_y_indices) rtol=1e-6
end

@testset verbose=true "staged VectorField derivative uses component halo requests  " begin
    q        = NSEBase.VectorField(g, NSEBase.FTField; N=2)
    expected = NSEBase.VectorField(g, NSEBase.FTField; N=2)
    out      = NSEBase.VectorField(g, NSEBase.FTField; N=2)

    plans(q[1], NSEBase.Field(g, u_fun))
    plans(q[2], NSEBase.Field(g, (x, y, z, t) -> 2u_fun(x, y, z, t)))
    plans(expected[1], NSEBase.Field(g, dudy_fun))
    plans(expected[2], NSEBase.Field(g, (x, y, z, t) -> 2dudy_fun(x, y, z, t)))

    NSEBase.ddy!(out, q)

    @test parent(out[1]) ≈ parent(expected[1]) rtol=1e-6
    @test parent(out[2]) ≈ parent(expected[2]) rtol=1e-6
end

sd(sym) = NSEBase.physical_to_storage_dim(NSEBase.grid(u), Val(sym))

@testset verbose=true "interior_dd! + wait + boundary_dd! covers the FD direction  " begin
    out = NSEBase.FTField(g)
    requests = MPIExt.init_requests!(u)
    MPIExt.interior_dd!(out, u, sd(:y))
    MPIExt.wait_requests!(requests)
    MPIExt.boundary_dd!(out, u, sd(:y))

    expected = NSEBase.FTField(g)
    plans(expected, NSEBase.Field(g, dudy_fun))
    @test parent(out) ≈ parent(expected) rtol=1e-6
end

@testset verbose=true "dd! along an FFT direction is spectral (no halo needed)     " begin
    out = NSEBase.FTField(g)
    NSEBase.dd!(out, u, sd(:x))
    expected = NSEBase.FTField(g)
    plans(expected, NSEBase.Field(g, dudx_fun))
    @test parent(out) ≈ parent(expected) rtol=1e-6

    out_z = NSEBase.FTField(g)
    NSEBase.dd!(out_z, u, sd(:z))
    expected_z = NSEBase.FTField(g)
    plans(expected_z, NSEBase.Field(g, dudz_fun))
    @test parent(out_z) ≈ parent(expected_z) rtol=1e-6

    out_t = NSEBase.FTField(g)
    NSEBase.dd!(out_t, u, sd(:t))
    expected_t = NSEBase.FTField(g)
    plans(expected_t, NSEBase.Field(g, dudt_fun))
    @test parent(out_t) ≈ parent(expected_t) rtol=1e-6
end

@testset verbose=true "staged Laplacian matches the analytic Laplacian             " begin
    out = NSEBase.FTField(g)
    requests = MPIExt.init_requests!(u)
    MPIExt.interior_laplacian!(out, u)
    MPIExt.wait_requests!(requests)
    MPIExt.boundary_laplacian!(out, u)

    expected = NSEBase.FTField(g)
    plans(expected, NSEBase.Field(g, lapl_fun))
    @test parent(out) ≈ parent(expected) rtol=1e-6

    adjoint_out = NSEBase.laplacian!(NSEBase.FTField(g), u; adjoint=true)
    serial_adjoint = NSEBase.laplacian!(NSEBase.FTField(g_parent), u_serial; adjoint=true)
    @test parent(adjoint_out) ≈ selectdim(parent(serial_adjoint), y_storage_dim, local_y_indices) rtol=1e-6
end

@testset verbose=true "NSEBase.laplacian!(out, u) agrees with the explicit form    " begin
    out_a = NSEBase.FTField(g)
    out_b = NSEBase.FTField(g)

    requests = MPIExt.init_requests!(u)
    MPIExt.interior_laplacian!(out_a, u)
    MPIExt.wait_requests!(requests)
    MPIExt.boundary_laplacian!(out_a, u)

    # 2-arg overload — auto-exchange.
    NSEBase.laplacian!(out_b, u)

    @test parent(out_a) ≈ parent(out_b)
end

@testset verbose=true "dd! on a non-decomposed FD direction (multiple FD dims)      " begin
    # A grid with more than one FD direction where only one of them (:z) is
    # decomposed: :x and :y are plain, fully-local FD directions on every
    # rank, exercising the `STORAGE_DIM ∉ DDIMS, ∉ FFT_DIMS_ORDER` branch of
    # `dd!`. Low-degree polynomials are differentiated exactly by the 5-point
    # stencil.
    cavity_parent = LidDrivenCavityGrid(13, 13, 32; spanwise=:bounded, width=5)
    cavity = distributed(cavity_parent, MPI.Comm_dup(MPI.COMM_WORLD);
                         decomposed_physical_dims=(:z,), nprocesses=(nranks,), nhalo=(NHALO,))
    cavity_plans = NSEBase.FFTPlans(cavity; dealias=false, flags=NSEBase.FFTW.ESTIMATE)

    cav_fun(x, y, z, _)     = x^2 * y^2 * z^2
    dcav_dx_fun(x, y, z, _) = 2x * y^2 * z^2
    dcav_dy_fun(x, y, z, _) = x^2 * 2y * z^2

    v = NSEBase.FTField(cavity); cavity_plans(v, NSEBase.Field(cavity, cav_fun))

    outx = NSEBase.FTField(cavity)
    NSEBase.dd!(outx, v, Val(storage_dim(cavity, :x)))
    expected_x = NSEBase.FTField(cavity); cavity_plans(expected_x, NSEBase.Field(cavity, dcav_dx_fun))
    @test parent(outx) ≈ parent(expected_x) rtol=1e-10

    outy = NSEBase.FTField(cavity)
    NSEBase.dd!(outy, v, Val(storage_dim(cavity, :y)))
    expected_y = NSEBase.FTField(cavity); cavity_plans(expected_y, NSEBase.Field(cavity, dcav_dy_fun))
    @test parent(outy) ≈ parent(expected_y) rtol=1e-10
end

MPI.free(base_comm)
