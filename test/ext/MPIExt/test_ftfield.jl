# Tests for the spectral-field constructor in
# `MPIExt/src/ftfield.jl`.
#
# Verifies that `FTField(g)` on a decomposed grid:
#   - allocates `HaloArrays.HaloArray` storage
#   - sizes the interior to `NSEBase.transform_size(g)`

import HaloArrays
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
g_halo   = MPIExt.distributed(g_parent, base_comm;
                                     decomposed_physical_dims=(:y,), nprocesses=(nranks,), nhalo=(NHALO,))

Test.@testset "FTField on decomposed grid is HaloArray-backed" begin
    uhat = NSEBase.FTField(g_halo)
    Test.@test parent(uhat) isa HaloArrays.HaloArray
    Test.@test size(uhat) == NSEBase.transform_size(g_halo)
    Test.@test HaloArrays.nhalo(parent(uhat)) == MPIExt.nhalo(g_halo)
    Test.@test all(parent(uhat) .== 0)
end

Test.@testset "FTField via convert preserves storage policy" begin
    uhat = NSEBase.FTField(g_halo)
    uhat32 = similar(uhat, ComplexF32)
    Test.@test parent(uhat32) isa HaloArrays.HaloArray
end

MPI.free(base_comm)
