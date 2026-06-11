# Tests for the decomposed-grid interface defined in
# `MPIExt/src/abstractgrid.jl`.
#
# Runs under any rank count; uses a `MockChannelGrid` parent wrapped via
# `MPIExt.distributed` along the wall-normal `:y` direction. Every
# user-facing accessor exported by abstractgrid.jl gets at least one
# assertion.

import LinearAlgebra
import MPI
import NSEBase
import Test

MPI.Initialized() || MPI.Init()

include("grid.jl")

# -- common setup -----------------------------------------------------------
nranks = MPI.Comm_size(MPI.COMM_WORLD)
rank   = MPI.Comm_rank(MPI.COMM_WORLD)

# Pick a wall-normal size that divides cleanly across any number of ranks
# used in CI (we test with 1 and 4 in `runtests.jl`).
const Ny = 16; const Nx = 7; const Nz = 9; const Nt = 5
const NHALO = 1

base_comm = MPI.Comm_dup(MPI.COMM_WORLD)

g_parent = MockChannelGrid(Ny, Nx, Nz, Nt)
g        = MPIExt.distributed(g_parent, base_comm;
                                     decomposed_physical_dims=(:y,), nprocesses=(nranks,), nhalo=(NHALO,))

Ny_local = Ny ÷ nranks
y_offset = rank * Ny_local

Test.@testset "DecomposedGrid alias" begin
    Test.@test g isa MPIExt.DecomposedGrid
end

Test.@testset "comm / comm_size / comm_rank" begin
    # `distributed(...)` builds a full-Cart comm (one direction per
    # storage dim) for HaloArrays. `comm(g)` returns that comm.
    g_comm = MPIExt.comm(g)
    Test.@test MPI.Cartdim_get(g_comm) == length(size(g))
    Test.@test MPIExt.comm_size(g) == nranks
    Test.@test MPIExt.comm_rank(g) == rank
end

Test.@testset "decomposition_storage_dims / _physical_dims" begin
    Test.@test MPIExt.decomposition_storage_dims(g) == (1,)
    Test.@test MPIExt.decomposition_physical_dims(g) == (:y,)
    Test.@test MPIExt.ndecomposed_dims(g) == 1
end

Test.@testset "size / global_size" begin
    Test.@test size(g) == (Ny_local, Nx, Nz, Nt)
    Test.@test MPIExt.global_size(g) == (Ny, Nx, Nz, Nt)
    Test.@test size(g, 1) == Ny_local
end

Test.@testset "nhalo" begin
    # Halos live on the decomposed (wall-normal) storage axis only.
    Test.@test MPIExt.nhalo(g) == (NHALO, 0, 0, 0)
    Test.@test MPIExt.nhalo(g, :y) == NHALO
end

Test.@testset "global_first_index" begin
    Test.@test MPIExt.global_first_index(g, :y) == y_offset + 1
    # Non-decomposed coordinates always start at global index 1.
    Test.@test MPIExt.global_first_index(g, :x) == 1
    Test.@test MPIExt.global_first_index(g, :z) == 1
    Test.@test MPIExt.global_first_index(g, :t) == 1
end

Test.@testset "local_interior_range / local_boundary_ranges" begin
    interior = MPIExt.local_interior_range(g, :y)
    lower, upper = MPIExt.local_boundary_ranges(g, :y)

    # On every rank, the three ranges (interior, lower, upper) form a
    # disjoint cover of 1:Ny_local. Empty bands are 1:0.
    all_rows = Set(1:Ny_local)
    covered  = Set(interior) ∪ Set(lower) ∪ Set(upper)
    Test.@test covered == all_rows

    # Interior excludes halo-adjacent rows only where there is a neighbour.
    has_lower_neighbor = rank > 0
    has_upper_neighbor = rank < nranks - 1
    Test.@test first(interior) == (has_lower_neighbor ? NHALO + 1 : 1)
    Test.@test last(interior)  == (has_upper_neighbor ? Ny_local - NHALO : Ny_local)
    Test.@test lower == (has_lower_neighbor ? (1:NHALO) : (1:0))
    Test.@test upper == (has_upper_neighbor ? ((Ny_local - NHALO + 1):Ny_local) : (1:0))

    # Non-decomposed direction: interior is the whole axis, boundaries are empty.
    Test.@test MPIExt.local_interior_range(g, :x) == 1:Nx
    Test.@test MPIExt.local_boundary_ranges(g, :x) == (1:0, 1:0)
end

Test.@testset "derivative_matrix forwards to parent" begin
    y_sd = NSEBase.storage_dim(g, :y)
    x_sd = NSEBase.storage_dim(g, :x)

    D1 = MPIExt.derivative_matrix(g, y_sd, Val(1))
    D2 = MPIExt.derivative_matrix(g, y_sd, Val(2))
    Test.@test D1 === g_parent.D₁
    Test.@test D2 === g_parent.D₂

    # Adjoint flag is forwarded.
    D1_adj = MPIExt.derivative_matrix(g, y_sd, Val(1), Val(true))
    Test.@test D1_adj == LinearAlgebra.adjoint(g_parent.D₁)

    # FFT directions have no parent FD operator; the parent throws.
    Test.@test_throws ArgumentError MPIExt.derivative_matrix(g, x_sd, Val(1))
end

Test.@testset "local_size names the per-rank interior size" begin
    Test.@test MPIExt.local_size(g) == size(g)
end

MPI.free(base_comm)
