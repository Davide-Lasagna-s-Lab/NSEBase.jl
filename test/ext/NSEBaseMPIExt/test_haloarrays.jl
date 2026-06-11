# Tests for HaloArray-backed field storage and halo request setup.
#
# Covers:
#   - HaloArray-backed Field / FTField allocation
#   - init_requests! on Field, FTField, VectorField
#   - wait_requests! on MPI MultiRequest batches and nested tuples

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
g        = NSEBaseMPIExt.distributed(g_parent, base_comm;
                                     decomposed_physical_dims=(:y,), nprocesses=(nranks,), nhalo=(NHALO,))

Test.@testset "Field allocation picks HaloArray storage when nhalo > 0" begin
    u = NSEBase.Field(g)
    Test.@test parent(u) isa HaloArrays.HaloArray
    Test.@test HaloArrays.nhalo(parent(u)) == NSEBaseMPIExt.nhalo(g)
    Test.@test all(parent(u) .== 0)
end

Test.@testset "FTField allocation picks HaloArray storage when nhalo > 0" begin
    uhat = NSEBase.FTField(g)
    Test.@test parent(uhat) isa HaloArrays.HaloArray
    Test.@test HaloArrays.nhalo(parent(uhat)) == NSEBaseMPIExt.nhalo(g)
    Test.@test all(parent(uhat) .== 0)
end

Test.@testset "VectorField allocation propagates halo policy" begin
    q = NSEBase.VectorField(g, NSEBase.FTField; N=3)
    Test.@test length(q) == 3
    Test.@test all(n -> parent(q[n]) isa HaloArrays.HaloArray, 1:3)
end

Test.@testset "init_requests! on Field / FTField" begin
    u    = NSEBase.Field(g)
    uhat = NSEBase.FTField(g)

    reqs_u    = NSEBase.init_requests!(u)
    reqs_uhat = NSEBase.init_requests!(uhat)

    Test.@test NSEBase.wait_requests!(reqs_u) === nothing
    Test.@test NSEBase.wait_requests!(reqs_uhat) === nothing
end

Test.@testset "init_requests! on VectorField returns nested request tuple" begin
    q = NSEBase.VectorField(g, NSEBase.FTField; N=2)
    reqs = NSEBase.init_requests!(q)
    Test.@test reqs isa Tuple && length(reqs) == 2
    Test.@test NSEBase.wait_requests!(reqs) === nothing
end

Test.@testset "halo requests propagates owned values to neighbouring ranks" begin
    # Tag each rank's owned interior with its rank number; the lower and
    # upper halo cells should contain the neighbours' values after exchange.
    u = NSEBase.Field(g)
    a = parent(u)
    fill!(parent(a), 0)  # zero the dense storage, including halo cells
    a .= Float64(rank + 1)

    reqs = NSEBase.init_requests!(u)
    NSEBase.wait_requests!(reqs)

    # `HaloArray` scalar indexing accepts halo indices such as 0 and n+1, but
    # `view` follows Julia's ordinary array bounds.  Inspect whole halo planes
    # through the dense parent storage, shifting by the halo width.
    P = parent(a)
    h = HaloArrays.nhalo(a)[1]
    if rank > 0
        # Lower halo received from rank - 1.
        Test.@test all(view(P, h, :, :, :) .== Float64(rank))
    end
    if rank < nranks - 1
        # Upper halo received from rank + 1.
        Test.@test all(view(P, h + size(g, 1) + 1, :, :, :) .== Float64(rank + 2))
    end
end

MPI.free(base_comm)
