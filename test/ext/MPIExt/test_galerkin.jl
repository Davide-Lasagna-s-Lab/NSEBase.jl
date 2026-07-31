# Tests for MPI-reduced inner products and the `project!` Galerkin override.
#
# The documented MPI contracts say that `dot` and `project!` first compute
# weighted contributions over the locally owned slab and then Allreduce them.
# Deterministic global spectral data let us compare those reduced results with
# the ordinary serial operations on the exact same parent `ChannelGrid`.

using Test

import LinearAlgebra
import HaloArrays
import MPI
import Random

using NSEBase,
      FDGrids

const MPIExt = Base.get_extension(NSEBase, :MPIExt)

MPI.Initialized() || MPI.Init()

include("../../support/rectangular_fixtures.jl")
using .TestRectangularFixtures: channel_grid

nranks = MPI.Comm_size(MPI.COMM_WORLD)
rank   = MPI.Comm_rank(MPI.COMM_WORLD)

const Ny = 16; const Nx = 5; const Nz = 5; const Nt = 5
const NHALO = 1
const NMODES = 2
const NCOMP = 2

base_comm = MPI.Comm_dup(MPI.COMM_WORLD)
g_parent = channel_grid(; Nx, Ny, Nz, Nt)
g = distributed(g_parent, base_comm;
                decomposed_physical_dims=(:y,), nprocesses=(nranks,), nhalo=(NHALO,))

Ny_local = Ny ÷ nranks
y_first = MPIExt.global_first_index(g, :y)
y_range = y_first:(y_first + Ny_local - 1)
Nx_hat = (Nx >> 1) + 1

# `ProjectedField(grid, modes)` expects `modes` to be a tuple of arrays
# of shape `(Nm, inh_axes..., fft_axes...)`. Construct global modes once and
# give each distributed basis the slab owned by its rank.
rng = Random.MersenneTwister(0xc0ffee)
global_mode_shape = (NMODES, Ny, Nx_hat, Nz, Nt)
global_modes = ntuple(_ -> randn(rng, ComplexF64, global_mode_shape...), NCOMP)
local_modes = ntuple(n -> copy(view(global_modes[n], :, y_range, :, :, :)), NCOMP)
basis = NSEBase.ProjectedField(g, local_modes)
serial_basis = NSEBase.ProjectedField(g_parent, global_modes)

# Populate distributed fields from slabs of deterministic global arrays. This
# makes the serial fields exact global counterparts, including their Hermitian
# multiplicities and production quadrature weights.
field_shape = (Ny, Nx_hat, Nz, Nt)
global_u = ntuple(_ -> randn(rng, ComplexF64, field_shape...), NCOMP)
global_v = ntuple(_ -> randn(rng, ComplexF64, field_shape...), NCOMP)
u = NSEBase.VectorField(g, NSEBase.FTField; N=NCOMP)
v = NSEBase.VectorField(g, NSEBase.FTField; N=NCOMP)
u_serial = NSEBase.VectorField(g_parent, NSEBase.FTField; N=NCOMP)
v_serial = NSEBase.VectorField(g_parent, NSEBase.FTField; N=NCOMP)
for n in 1:NCOMP
    parent(u[n]) .= view(global_u[n], y_range, :, :, :)
    parent(v[n]) .= view(global_v[n], y_range, :, :, :)
    parent(u_serial[n]) .= global_u[n]
    parent(v_serial[n]) .= global_v[n]
end

@testset verbose=true "dot and norm equal the serial global reductions             " begin
    # The scalar overload performs one Allreduce; VectorField aggregation is a
    # sum of those already-global component results, as documented by MPIExt.
    @test LinearAlgebra.dot(u[1], v[1]) ≈ LinearAlgebra.dot(u_serial[1], v_serial[1])
    @test LinearAlgebra.dot(u, v) ≈ LinearAlgebra.dot(u_serial, v_serial)
    @test LinearAlgebra.norm(u) ≈ LinearAlgebra.norm(u_serial)
end

@testset verbose=true "project! coefficients equal the serial global projection    " begin
    a_local = NSEBase.project!(similar(basis), u)
    a_serial = NSEBase.project!(similar(serial_basis), u_serial)

    # Allreduce must recover the undecomposed weighted projection, not merely
    # produce copies that happen to agree across ranks.
    @test parent(a_local) ≈ parent(a_serial)

    # Retain the explicit rank-consistency assertion from the original test.
    a_root = Vector{ComplexF64}(vec(parent(a_local)))
    MPI.Bcast!(a_root, 0, MPIExt.comm(g))
    @test vec(parent(a_local)) ≈ a_root
end

MPI.free(base_comm)
