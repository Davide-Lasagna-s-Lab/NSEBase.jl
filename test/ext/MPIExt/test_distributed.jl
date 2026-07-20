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
using LinearAlgebra

import MPI

using NSEBase

const MPIExt = Base.get_extension(NSEBase, :MPIExt)

MPI.Initialized() || MPI.Init()

include("helpers.jl")

nranks = MPI.Comm_size(MPI.COMM_WORLD)
rank   = MPI.Comm_rank(MPI.COMM_WORLD)

const Ny = 16; const Nx = 7; const Nz = 9; const Nt = 5
const NHALO = 1

base_comm = MPI.Comm_dup(MPI.COMM_WORLD)
g_parent = test_channel_grid(Ny, Nx, Nz, Nt)
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
        bad_parent = test_channel_grid(Ny + 1, Nx, Nz, Nt)
        @test_throws ArgumentError distributed(bad_parent, base_comm;
                                                decomposed_physical_dims=(:y,), nprocesses=(nranks,), nhalo=(1,))
    end
end

@testset verbose=true "parent(g) returns the original Undecomposed grid            " begin
    @test parent(g) === g_parent
    @test eltype(g) === Float64
end

@testset verbose=true "points(g) returns local FD and global Fourier coordinates   " begin
    y, x, z, t = NSEBase.points(g)
    # Wall-normal: this rank only owns its slab.
    @test vec(y) == g_parent.xs[1][y_offset+1:y_offset+Ny_local]
    # FFT directions: every rank holds the global coordinate array.
    parent_y, parent_x, parent_z, parent_t = NSEBase.points(g_parent)
    @test x == parent_x
    @test z == parent_z
    @test t == parent_t && vec(t) ≈ (0:Nt-1) ./ Nt
end

@testset verbose=true "weights(g) returns per-rank slice and is cached             " begin
    w = NSEBase.weights(g)
    @test w == g_parent.ws[1][y_offset+1:y_offset+Ny_local]
    # Cached: identity on repeated calls.
    @test NSEBase.weights(g) === w
end

@testset verbose=true "wavenumber_scale delegates to parent (symbol arg)           " begin
    @test NSEBase.wavenumber_scale(g, :x) == NSEBase.wavenumber_scale(g_parent, NSEBase.storage_dim(g_parent, :x))
    @test NSEBase.wavenumber_scale(g, :z) == NSEBase.wavenumber_scale(g_parent, NSEBase.storage_dim(g_parent, :z)) &&
          NSEBase.wavenumber_scale(g, :t) == 2π
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
    @test eltype(g32) === Float32
    @test eltype(parent(g32).xs[1]) === Float32
    @test MPIExt.decomposition_physical_dims(g32) == (:y,)
    @test MPIExt.nhalo(g32) == MPIExt.nhalo(g)
    # Already-Float64 conversion is the identity.
    @test convert(Float64, g) === g
end

@testset verbose=true "case constructors accept a decomposed rectangular grid      " begin
    case_parent = ChannelGrid(Nx, Ny, Nz; Nt, α=1, β=1, lim=(2, 6), width=3)
    case_grid = distributed(case_parent, base_comm; decomposed_physical_dims=(:y,),
                            nprocesses=(nranks,), nhalo=(NHALO,))
    equations = PlaneCouetteFlow(case_grid, 100; fftw_flags=FFTW.ESTIMATE, dealias=false)
    local_y = vec(points(case_grid)[only(inhomogeneous_storage_dims(case_grid))])
    y₋, y₊ = extrema(case_parent.xs[1])
    expected_base = @. 2 * (local_y - y₋) / (y₊ - y₋) - 1
    @test equations.base[1] ≈ expected_base

    streamwise_parent = StreamwiseInvariantChannelGrid(Ny, Nz; Nt, β=1, lim=(2, 6), width=3)
    streamwise_grid = distributed(streamwise_parent, base_comm; decomposed_physical_dims=(:y,),
                                  nprocesses=(nranks,), nhalo=(NHALO,))
    streamwise = PlaneCouetteFlow(streamwise_grid, 100; fftw_flags=FFTW.ESTIMATE, dealias=false)
    streamwise_y = vec(points(streamwise_grid)[only(inhomogeneous_storage_dims(streamwise_grid))])
    streamwise_expected = @. 2 * (streamwise_y - y₋) / (y₊ - y₋) - 1

    two_dimensional_parent = TwoDimensionalChannelGrid(Nx, Ny; Nt, α=1, lim=(2, 6), width=3)
    two_dimensional_grid = distributed(two_dimensional_parent, base_comm; decomposed_physical_dims=(:y,),
                                       nprocesses=(nranks,), nhalo=(NHALO,))
    two_dimensional = PlanePoiseuilleFlow(two_dimensional_grid, 100;
                                          fftw_flags=FFTW.ESTIMATE, dealias=false)
    two_dimensional_convection = RayleighBenardFlow(two_dimensional_grid, 1, 0.71, 1000;
                                                    fftw_flags=FFTW.ESTIMATE, dealias=false)
    convection_state = VectorField(two_dimensional_grid; N=3)
    add_base_flow!(convection_state, two_dimensional_convection.base)
    convection_residual = two_dimensional_convection.nl(0.0, convection_state, similar(convection_state))
    two_dimensional_y = vec(points(two_dimensional_grid)[only(inhomogeneous_storage_dims(two_dimensional_grid))])
    two_dimensional_expected = @. 1 - (2 * (two_dimensional_y - y₋) / (y₊ - y₋) - 1)^2
    two_dimensional_temperature = @. (1 - (2 * (two_dimensional_y - y₋) / (y₊ - y₋) - 1)) / 2

    @test case_grid isa AbstractChannel3DGrid{Float64}
    @test streamwise_grid isa AbstractStreamwiseInvariantChannelGrid{Float64}
    @test MPIExt.decomposition_physical_dims(streamwise_grid) == (:y,)
    @test streamwise.base[3] ≈ streamwise_expected
    @test two_dimensional_grid isa AbstractTwoDimensionalChannelGrid{Float64}
    @test two_dimensional.base[1] ≈ two_dimensional_expected
    @test two_dimensional_convection.nl isa CartesianPrimitive2DBoussinesqNSE
    @test two_dimensional_convection.base[3] ≈ two_dimensional_temperature
    @test norm(convection_residual[1]) < 1e-12
    @test convection_residual[2] ≈ two_dimensional_convection.nl.Ri .* convection_state[3] atol=1e-12
    @test norm(convection_residual[3]) < 1e-10
end

MPI.free(base_comm)
