# Tests for the FFT-plan overrides in `MPIExt/src/fft.jl`.
#
# Covers:
#   - `FFTPlans` constructor on a decomposed grid plans against the full
#     halo-inclusive parent allocation
#   - in-place forward/inverse round-trip preserves the field exactly
#   - the allocating `NSEBase.FFT` / `NSEBase.IFFT` helpers use the same
#     halo-inclusive convention
#   - dealiased plans transform the padded parent allocation correctly

using Test

import HaloArrays
import MPI

using NSEBase,
      FDGrids

MPI.Initialized() || MPI.Init()

include("../../mock_channel_grid.jl")

nranks = MPI.Comm_size(MPI.COMM_WORLD)
rank   = MPI.Comm_rank(MPI.COMM_WORLD)

const Ny = 16; const Nx = 7; const Nz = 9; const Nt = 5
const NHALO = 1

base_comm = MPI.Comm_dup(MPI.COMM_WORLD)
g = distributed(MockChannelGrid(Ny, Nx, Nz, Nt), base_comm;
                decomposed_physical_dims=(:y,), nprocesses=(nranks,), nhalo=(NHALO,))

# Build a non-trivial physical-space field whose interior values are seeded
# deterministically per rank, then transform forward and back.
function build_seeded_field(g, dealias)
    u = NSEBase.Field(g; dealias=dealias)
    y, x, z, t = NSEBase.points(g; dealias=dealias)
    parent(u) .= @. sin(y) * cos(x) + 0.3 * sin(2*z) * cos(t)
    return u
end

@testset "FFTPlans round-trip on halo-backed FTField                          " begin
    plans = NSEBase.FFTPlans(g; dealias=false, flags=NSEBase.FFTW.ESTIMATE)

    u    = build_seeded_field(g, false)
    uhat = NSEBase.FTField(g)
    v    = NSEBase.Field(g)

    plans(uhat, u)
    plans(v, uhat)

    @test parent(v) ≈ parent(u)
end

@testset "Allocating FFT / IFFT helpers match FFTPlans                        " begin
    u = build_seeded_field(g, false)

    plans = NSEBase.FFTPlans(g; dealias=false, flags=NSEBase.FFTW.ESTIMATE)
    uhat_planned = NSEBase.FTField(g)
    plans(uhat_planned, u)

    uhat_alloc = NSEBase.FFT(u)
    @test parent(uhat_alloc) ≈ parent(uhat_planned)

    v_alloc = NSEBase.IFFT(uhat_alloc)
    @test parent(v_alloc) ≈ parent(u)
end

@testset "Dealiased FFTPlans round-trip preserves the spectral field          " begin
    plans_lo = NSEBase.FFTPlans(g; dealias=false, flags=NSEBase.FFTW.ESTIMATE)
    plans_hi = NSEBase.FFTPlans(g; dealias=true,  flags=NSEBase.FFTW.ESTIMATE)

    # Build a low-resolution FTField, IFFT into a dealiased Field, then
    # transform back; the original spectrum should be recovered.
    u = build_seeded_field(g, false)
    uhat = NSEBase.FTField(g)
    plans_lo(uhat, u)

    ud = NSEBase.Field(g; dealias=true)
    plans_hi(ud, uhat)

    uhat_back = NSEBase.FTField(g)
    plans_hi(uhat_back, ud)

    @test parent(uhat_back) ≈ parent(uhat)
end

MPI.free(base_comm)
