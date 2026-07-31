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

using Test

import MPI
import HaloArrays

using NSEBase,
      FDGrids

const MPIExt = Base.get_extension(NSEBase, :MPIExt)

MPI.Initialized() || MPI.Init()

include("../../support/rectangular_fixtures.jl")
using .TestRectangularFixtures: channel_grid

nranks = MPI.Comm_size(MPI.COMM_WORLD)
rank   = MPI.Comm_rank(MPI.COMM_WORLD)

const Ny = 16; const Nx = 7; const Nz = 9; const Nt = 5
const NHALO = 1

base_comm = MPI.Comm_dup(MPI.COMM_WORLD)
g_parent = channel_grid(; Nx, Ny, Nz, Nt)
g        = distributed(g_parent, base_comm;
                        decomposed_physical_dims=(:y,), nprocesses=(nranks,), nhalo=(NHALO,))

Ny_local = Ny ÷ nranks
y_offset = rank * Ny_local

@testset verbose=true "distributed: required kwargs and validation                 " begin
    # Missing required kwargs.
    @test_throws UndefKeywordError distributed(g_parent, base_comm)
    @test_throws UndefKeywordError distributed(
        g_parent, base_comm; decomposed_physical_dims=(:y,)
    )

    # `prod(nprocesses)` must equal `Comm_size(comm)`.
    @test_throws ArgumentError distributed(
        g_parent, base_comm;
        decomposed_physical_dims=(:y,),
        nprocesses=(nranks * 2,), nhalo=(NHALO,))

    # Decomposed direction must be spatial inhomogeneous; `:x` is
    # FFT-transformed in the channel layout.
    @test_throws ArgumentError distributed(
        g_parent, base_comm;
        decomposed_physical_dims=(:x,),
        nprocesses=(nranks,), nhalo=(NHALO,))

    # Duplicate decomposed_physical_dims entries are rejected.
    @test_throws ArgumentError distributed(
        g_parent, base_comm;
        decomposed_physical_dims=(:y, :y),
        nprocesses=(1, nranks), nhalo=(NHALO, NHALO))

    # Decomposed dimensions always allocate HaloArray-backed fields, so their
    # halo widths must be positive.
    @test_throws ArgumentError distributed(
        g_parent, base_comm;
        decomposed_physical_dims=(:y,),
        nprocesses=(nranks,), nhalo=(0,))

    # `distributed` builds the full Cartesian communicator itself, so the
    # input communicator must not already carry a topology.
    topologized_comm = MPI.Cart_create(base_comm, (nranks,);
                                       periodic=(false,), reorder=false)
    try
        @test_throws ArgumentError distributed(
            g_parent, topologized_comm;
            decomposed_physical_dims=(:y,),
            nprocesses=(nranks,), nhalo=(NHALO,))
    finally
        MPI.free(topologized_comm)
    end
end

if nranks > 1
    @testset verbose=true "distributed: rejects non-divisible parent size              " begin
        # Ny + 1 across nranks (>1) does not divide evenly.
        bad_parent = channel_grid(; Nx, Ny=Ny + 1, Nz, Nt)
        @test_throws ArgumentError distributed(bad_parent, base_comm;
                                                decomposed_physical_dims=(:y,), nprocesses=(nranks,), nhalo=(1,))
    end
end

@testset verbose=true "parent(g) returns the original Undecomposed grid            " begin
    @test parent(g) === g_parent
end

@testset verbose=true "points returns local wall-normal and global FFT coordinates " begin
    y, x, z, t = NSEBase.points(g)
    # Wall-normal: this rank only owns its slab.
    parent_y, parent_x, parent_z, parent_t = NSEBase.points(g_parent)
    @test vec(y) == vec(parent_y)[y_offset+1:y_offset+Ny_local]
    # FFT directions: every rank holds the global coordinate array.
    @test x == parent_x
    @test z == parent_z
    @test t == parent_t
end

@testset verbose=true "weights(g) returns per-rank slice and is cached             " begin
    w = NSEBase.weights(g)
    parent_weights = NSEBase.weights(g_parent)
    @test w == parent_weights[y_offset+1:y_offset+Ny_local]
    # Cached: identity on repeated calls.
    @test NSEBase.weights(g) === w
end

@testset verbose=true "wavenumber_scale delegates to parent (symbol arg)           " begin
    x_sd = NSEBase.storage_dim(g_parent, :x)
    z_sd = NSEBase.storage_dim(g_parent, :z)
    @test NSEBase.wavenumber_scale(g, :x) == NSEBase.wavenumber_scale(g_parent, x_sd)
    @test NSEBase.wavenumber_scale(g, :z) == NSEBase.wavenumber_scale(g_parent, z_sd)
end

@testset verbose=true "nhalo / global_size are baked into the type                 " begin
    @test MPIExt.nhalo(g) == (NHALO, 0, 0, 0)
    @test MPIExt.nhalo(g, :y) == NHALO
    @test MPIExt.global_size(g) == (Ny, Nx, Nz, Nt)
    @test MPIExt.global_size(g, :y) == Ny
    @test MPIExt.local_size(g, :y) == Ny_local
end

@testset verbose=true "growto enlarges only FFT dimensions and re-wraps            " begin
    target = (Nx + 2, Nz + 2, Nt + 2)  # grow x, z, t by 2 each
    g2 = NSEBase.growto(g, target)
    @test g2 isa MPIExt.DecomposedGrid
    @test MPIExt.global_size(g2) == (Ny, target[1], target[2], target[3])
    @test MPIExt.nhalo(g2) == MPIExt.nhalo(g)
    @test MPIExt.decomposition_physical_dims(g2) == (:y,)
end

@testset verbose=true "convert preserves wrapper metadata                          " begin
    g32 = convert(Float32, g)
    @test eltype(parent(g32)) === Float32
    @test MPIExt.decomposition_physical_dims(g32) == (:y,)
    @test MPIExt.nhalo(g32) == MPIExt.nhalo(g)
    # Already-Float64 conversion is the identity.
    @test convert(Float64, g) === g
end

MPI.free(base_comm)
