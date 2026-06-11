# Tests for the `DecomposedGrid` wrapper defined in
# `MPIExt/src/distributed.jl`.
#
# Covers:
#   - construction validation (required kwargs, rejected dims, communicator shape)
#   - `parent(g)` accessor
#   - cached `weights(g)` / `points(g)` per-rank values
#   - delegation: `wavenumber_scale`, `derivative_matrix`, `nhalo`,
#     `global_size`
#   - `growto` and `convert` round-trip preserving comm/dims/halo widths

import MPI
import NSEBase
import Test

MPI.Initialized() || MPI.Init()

include("grid.jl")

nranks = MPI.Comm_size(MPI.COMM_WORLD)
rank   = MPI.Comm_rank(MPI.COMM_WORLD)

const Ny = 16; const Nx = 7; const Nz = 9; const Nt = 5
const NHALO = 1

base_comm = MPI.Comm_dup(MPI.COMM_WORLD)
g_parent = MockChannelGrid(Ny, Nx, Nz, Nt)
g        = MPIExt.distributed(g_parent, base_comm;
                                     decomposed_physical_dims=(:y,), nprocesses=(nranks,), nhalo=(NHALO,))

Ny_local = Ny ÷ nranks
y_offset = rank * Ny_local

Test.@testset "distributed: required kwargs and validation" begin
    # Missing required kwargs.
    Test.@test_throws UndefKeywordError MPIExt.distributed(g_parent, base_comm)
    Test.@test_throws UndefKeywordError MPIExt.distributed(
        g_parent, base_comm; decomposed_physical_dims=(:y,))

    # `prod(nprocesses)` must equal `Comm_size(comm)`.
    Test.@test_throws ArgumentError MPIExt.distributed(
        g_parent, base_comm;
        decomposed_physical_dims=(:y,),
        nprocesses=(nranks * 2,), nhalo=(NHALO,))

    # Decomposed direction must be spatial inhomogeneous; `:x` is
    # FFT-transformed in the channel layout.
    Test.@test_throws ArgumentError MPIExt.distributed(
        g_parent, base_comm;
        decomposed_physical_dims=(:x,),
        nprocesses=(nranks,), nhalo=(NHALO,))

    # Duplicate decomposed_physical_dims entries are rejected.
    Test.@test_throws ArgumentError MPIExt.distributed(
        g_parent, base_comm;
        decomposed_physical_dims=(:y, :y),
        nprocesses=(1, nranks), nhalo=(NHALO, NHALO))

    # Decomposed dimensions always allocate HaloArray-backed fields, so their
    # halo widths must be positive.
    Test.@test_throws ArgumentError MPIExt.distributed(
        g_parent, base_comm;
        decomposed_physical_dims=(:y,),
        nprocesses=(nranks,), nhalo=(0,))

    # `distributed` builds the full Cartesian communicator itself, so the
    # input communicator must not already carry a topology.
    topologized_comm = MPI.Cart_create(base_comm, (nranks,);
                                       periodic=(false,), reorder=false)
    try
        Test.@test_throws ArgumentError MPIExt.distributed(
            g_parent, topologized_comm;
            decomposed_physical_dims=(:y,),
            nprocesses=(nranks,), nhalo=(NHALO,))
    finally
        MPI.free(topologized_comm)
    end
end

if nranks > 1
    Test.@testset "distributed: rejects non-divisible parent size" begin
        # Ny + 1 across nranks (>1) does not divide evenly.
        bad_parent = MockChannelGrid(Ny + 1, Nx, Nz, Nt)
        Test.@test_throws ArgumentError MPIExt.distributed(bad_parent, base_comm;
                                                                   decomposed_physical_dims=(:y,), nprocesses=(nranks,), nhalo=(1,))
    end
end

Test.@testset "parent(g) returns the original Undecomposed grid" begin
    Test.@test parent(g) === g_parent
end

Test.@testset "points(g) returns per-rank wall-normal coords + global FFT coords" begin
    y, x, z, t = NSEBase.points(g)
    # Wall-normal: this rank only owns its slab.
    Test.@test vec(y) == g_parent.y[y_offset+1 : y_offset+Ny_local]
    # FFT directions: every rank holds the global coordinate array.
    parent_y, parent_x, parent_z, parent_t = NSEBase.points(g_parent)
    Test.@test x == parent_x
    Test.@test z == parent_z
    Test.@test t == parent_t
end

Test.@testset "weights(g) returns per-rank slice and is cached" begin
    w = NSEBase.weights(g)
    Test.@test w == g_parent.ws[y_offset+1 : y_offset+Ny_local]
    # Cached: identity on repeated calls.
    Test.@test NSEBase.weights(g) === w
end

Test.@testset "wavenumber_scale delegates to parent (symbol arg)" begin
    Test.@test NSEBase.wavenumber_scale(g, :x) == g_parent.α
    Test.@test NSEBase.wavenumber_scale(g, :z) == g_parent.β
end

Test.@testset "nhalo / global_size are baked into the type" begin
    Test.@test MPIExt.nhalo(g) == (NHALO, 0, 0, 0)
    Test.@test MPIExt.nhalo(g, :y) == NHALO
    Test.@test MPIExt.global_size(g) == (Ny, Nx, Nz, Nt)
    Test.@test MPIExt.global_size(g, :y) == Ny
    Test.@test MPIExt.local_size(g, :y) == Ny_local
end

Test.@testset "growto enlarges only FFT dimensions and re-wraps" begin
    target = (Nx + 2, Nz + 2, Nt + 2)  # grow x, z, t by 2 each
    g2 = NSEBase.growto(g, target)
    Test.@test g2 isa MPIExt.DecomposedGrid
    Test.@test MPIExt.global_size(g2) == (Ny, target[1], target[2], target[3])
    Test.@test MPIExt.nhalo(g2) == MPIExt.nhalo(g)
    Test.@test MPIExt.decomposition_physical_dims(g2) == (:y,)
end

Test.@testset "convert preserves wrapper metadata" begin
    g32 = convert(Float32, g)
    Test.@test eltype(parent(g32).y) === Float32
    Test.@test MPIExt.decomposition_physical_dims(g32) == (:y,)
    Test.@test MPIExt.nhalo(g32) == MPIExt.nhalo(g)
    # Already-Float64 conversion is the identity.
    Test.@test convert(Float64, g) === g
end

MPI.free(base_comm)
