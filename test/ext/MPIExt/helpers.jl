# Shared production-grid fixture for the MPI extension tests.

import FDGrids
import NSEBase

"""
    test_channel_grid(Ny, Nx, Nz, Nt; T=Float64, stencil_width=3, α=2π, β=2π)

Construct the production channel grid used by MPI tests. A mapped uniform
wall-normal grid retains the previous MPI fixture's discretisation while all
grid operations exercise `NSEBase.ChannelGrid` itself.
"""
function test_channel_grid(Ny::Int, Nx::Int, Nz::Int, Nt::Int;
                           T::Type{<:Real}=Float64, stencil_width::Int=3,
                           α::Real=2π, β::Real=2π)
    dist = FDGrids.MappedGrid(T(1))
    return NSEBase.ChannelGrid(Nx, Ny, Nz; Nt, α, β, dist, width=stencil_width, T)
end
