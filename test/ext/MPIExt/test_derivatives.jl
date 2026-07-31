# Tests for the Val{STORAGE_DIM}-dispatched derivative API in
# `MPIExt/src/derivatives.jl`.
#
# Covers:
#   - `interior_dd!` / `boundary_dd!` cover the local dimension exactly
#   - staged halo exchange reproduces the undecomposed parent-grid derivative
#   - spectral directions (`:x`, `:z`, `:t`) bypass the FD path
#   - vector-field derivatives consume per-component halo request tuples
#   - `interior_laplacian!` / `boundary_laplacian!` reproduce the serial
#     parent-grid Laplacian after halo exchange
#   - the 2-arg `NSEBase.laplacian!` auto-exchange form agrees

using Test

import FDGrids
import HaloArrays
import LinearAlgebra
import MPI

using NSEBase

const MPIExt = Base.get_extension(NSEBase, :MPIExt)

MPI.Initialized() || MPI.Init()

include("../../support/rectangular_fixtures.jl")
using .TestRectangularFixtures: channel_grid

nranks = MPI.Comm_size(MPI.COMM_WORLD)
rank   = MPI.Comm_rank(MPI.COMM_WORLD)

# A smooth, multimode field exercises every Fourier direction and the
# production FDGrids wall-normal operator. Correctness is defined by the
# documented decomposition contract: each distributed result must equal the
# matching slab of the same operation on the undecomposed parent grid.
const α_test = 2π
const β_test = 5.8
u_fun(y, x, z, t) = (1 + y + y^3) *
                    exp(sin(α_test*x) + 0.2cos(β_test*z) + 0.1sin(2π*t))

const Ny = 32; const Nx = 7; const Nz = 9; const Nt = 5
const NHALO = 2  # 5-point stencil -> half-width 2

base_comm = MPI.Comm_dup(MPI.COMM_WORLD)
# The halo width is half the production grid's five-point FD stencil.
g_parent = channel_grid(; Nx, Ny, Nz, Nt, α=α_test, β=β_test, width=5)
g = distributed(g_parent, base_comm;
                decomposed_physical_dims=(:y,), nprocesses=(nranks,), nhalo=(NHALO,))

# Build identical serial and distributed states from the same parent grid.
plans = NSEBase.FFTPlans(g; dealias=false, flags=NSEBase.FFTW.ESTIMATE)
u_phys = NSEBase.Field(g, u_fun)
u      = NSEBase.FTField(g); plans(u, u_phys)

serial_plans = NSEBase.FFTPlans(g_parent; dealias=false, flags=NSEBase.FFTW.ESTIMATE)
u_serial = NSEBase.FTField(g_parent)
serial_plans(u_serial, NSEBase.Field(g_parent, u_fun))

y_first = MPIExt.global_first_index(g, :y)
y_range = y_first:(y_first + size(g, 1) - 1)
function serial_slab(field)
    selectors = ntuple(dim -> dim == 1 ? y_range : Colon(), ndims(parent(field)))
    return view(parent(field), selectors...)
end

# Serial references use the same FD matrices, Fourier scales, and parent-grid
# data. Any discrepancy therefore isolates decomposition or halo handling.
ddy_serial = NSEBase.FTField(g_parent); NSEBase.ddy!(ddy_serial, u_serial)
ddx_serial = NSEBase.FTField(g_parent); NSEBase.ddx!(ddx_serial, u_serial)
ddz_serial = NSEBase.FTField(g_parent); NSEBase.ddz!(ddz_serial, u_serial)
ddt_serial = NSEBase.FTField(g_parent); NSEBase.ddt!(ddt_serial, u_serial)
lapl_serial = NSEBase.FTField(g_parent); NSEBase.laplacian!(lapl_serial, u_serial)

@testset verbose=true "staged ddy! matches the serial parent-grid derivative       " begin
    out = NSEBase.FTField(g)
    NSEBase.ddy!(out, u)
    @test parent(out) ≈ serial_slab(ddy_serial) rtol=1e-6
end

@testset verbose=true "staged VectorField derivative matches the serial result     " begin
    q        = NSEBase.VectorField(g, NSEBase.FTField; N=2)
    out      = NSEBase.VectorField(g, NSEBase.FTField; N=2)

    plans(q[1], NSEBase.Field(g, u_fun))
    plans(q[2], NSEBase.Field(g, (y, x, z, t) -> 2u_fun(y, x, z, t)))

    q_serial = NSEBase.VectorField(g_parent, NSEBase.FTField; N=2)
    dq_serial = NSEBase.VectorField(g_parent, NSEBase.FTField; N=2)
    serial_plans(q_serial[1], NSEBase.Field(g_parent, u_fun))
    serial_plans(q_serial[2], NSEBase.Field(g_parent,
                                            (y, x, z, t) -> 2u_fun(y, x, z, t)))
    NSEBase.ddy!(dq_serial, q_serial)

    NSEBase.ddy!(out, q)

    @test parent(out[1]) ≈ serial_slab(dq_serial[1]) rtol=1e-6
    @test parent(out[2]) ≈ serial_slab(dq_serial[2]) rtol=1e-6
end

sd(sym) = NSEBase.physical_to_storage_dim(NSEBase.grid(u), Val(sym))

@testset verbose=true "interior_dd! + wait + boundary_dd! covers the FD direction  " begin
    out = NSEBase.FTField(g)
    requests = MPIExt.init_requests!(u)
    MPIExt.interior_dd!(out, u, sd(:y))
    MPIExt.wait_requests!(requests)
    MPIExt.boundary_dd!(out, u, sd(:y))

    @test parent(out) ≈ serial_slab(ddy_serial) rtol=1e-6
end

@testset verbose=true "dd! along an FFT direction is spectral (no halo needed)     " begin
    out = NSEBase.FTField(g)
    NSEBase.dd!(out, u, sd(:x))
    @test parent(out) ≈ serial_slab(ddx_serial) rtol=1e-6

    out_z = NSEBase.FTField(g)
    NSEBase.dd!(out_z, u, sd(:z))
    @test parent(out_z) ≈ serial_slab(ddz_serial) rtol=1e-6

    out_t = NSEBase.FTField(g)
    NSEBase.dd!(out_t, u, sd(:t))
    @test parent(out_t) ≈ serial_slab(ddt_serial) rtol=1e-6
end

@testset verbose=true "staged Laplacian matches the serial parent-grid Laplacian   " begin
    out = NSEBase.FTField(g)
    requests = MPIExt.init_requests!(u)
    MPIExt.interior_laplacian!(out, u)
    MPIExt.wait_requests!(requests)
    MPIExt.boundary_laplacian!(out, u)

    @test parent(out) ≈ serial_slab(lapl_serial) rtol=1e-6
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
    @test parent(out_b) ≈ serial_slab(lapl_serial) rtol=1e-6
end

MPI.free(base_comm)
