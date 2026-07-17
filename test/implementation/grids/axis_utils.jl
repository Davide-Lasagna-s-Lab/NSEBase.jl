# Tests for homogeneous_axes and inhomogeneous_axes (src/ftfield.jl).
#
# Both functions are @generated and dispatch on FTField.  The tests verify:
#
#   1. Standard layout (inh dims first, FFT dims last).
#      FFT_DIMS_ORDER = (2, 3), so homogeneous_axes returns the rfft and z
#      axes of the FTField, and inhomogeneous_axes returns the y axis.
#
#   2. Flipped layout (FFT dims first, inh dim last).
#      FFT_DIMS_ORDER = (1, 2), so homogeneous_axes must return the rfft
#      and z axes of the FTField (axes 1 and 2), and inhomogeneous_axes
#      must return the y axis (axis 3).
#
# Together these two layouts verify that the functions read the correct
# storage positions regardless of which dimensions come first.

@testset verbose=true "homogeneous_axes / inhomogeneous_axes (FTField)             " begin

    # ---------------------------------------------------------------- #
    # Inhomogeneous dimension first, FFT dimensions last               #
    # FFT_DIMS_ORDER = (2, 3), inh dim = 1                             #
    # FTField shape: (Ny, (Nx>>1)+1, Nz)                               #
    # ---------------------------------------------------------------- #
    @testset verbose=true "RectangularGrid: inhomogeneous first                        " begin
        Ny, Nx, Nz = 7, 9, 7
        g = channel_test_grid(Ny, Nx, Nz)
        u = FTField(g)
        # FTField shape: (Ny, (Nx>>1)+1, Nz) = (7, 5, 7)

        hom = NSEBase.homogeneous_axes(u)
        inh = NSEBase.inhomogeneous_axes(u)

        # Two homogeneous axes: rfft (axis 2) and z (axis 3)
        @test length(hom) == 2
        @test hom[1] == axes(u, 2)   # rfft axis: 1:(Nx>>1)+1 = 1:5
        @test hom[2] == axes(u, 3)   # z axis: 1:Nz = 1:7

        # One inhomogeneous axis: y (axis 1)
        @test length(inh) == 1
        @test inh[1] == axes(u, 1)   # y axis: 1:Ny = 1:7

        # CartesianIndices over the homogeneous axes covers all wavenumbers
        @test length(CartesianIndices(hom)) == ((Nx >> 1) + 1) * Nz
        # CartesianIndices over inhomogeneous covers all wall-normal points
        @test length(CartesianIndices(inh)) == Ny
    end

    # ---------------------------------------------------------------- #
    # FFT dimensions first, inhomogeneous dimension last               #
    # FFT_DIMS_ORDER = (1, 2), inh dim = 3                             #
    # FTField shape: ((Nx>>1)+1, Nz, Ny)                               #
    # ---------------------------------------------------------------- #
    @testset verbose=true "RectangularGrid: homogeneous first                          " begin
        Nx, Nz, Ny = 9, 7, 7
        g = flipped_channel_test_grid(Nx, Nz, Ny)
        u = FTField(g)
        # FTField shape: ((Nx>>1)+1, Nz, Ny) = (5, 7, 7)

        hom = NSEBase.homogeneous_axes(u)
        inh = NSEBase.inhomogeneous_axes(u)

        # Two homogeneous axes: rfft (axis 1) and z (axis 2)
        @test length(hom) == 2
        @test hom[1] == axes(u, 1)   # rfft axis: 1:(Nx>>1)+1 = 1:5
        @test hom[2] == axes(u, 2)   # z axis: 1:Nz = 1:7

        # One inhomogeneous axis: y (axis 3)
        @test length(inh) == 1
        @test inh[1] == axes(u, 3)   # y axis: 1:Ny = 1:7

        # Coverage counts are the same for both layouts.
        @test length(CartesianIndices(hom)) == ((Nx >> 1) + 1) * Nz
        @test length(CartesianIndices(inh)) == Ny
    end

    # ---------------------------------------------------------------- #
    # shift! correctness on the flipped layout                         #
    # ---------------------------------------------------------------- #
    # The shift! function for FTField relies on homogeneous_axes and
    # inhomogeneous_axes.  Verify that it gives the same physical result
    # on both storage layouts when they represent the same field.
    @testset verbose=true "shift! gives correct phase on a flipped layout              " begin
        Nx, Nz, Ny = 9, 7, 7
        g_tri  = channel_test_grid(Ny, Nx, Nz)
        g_flip = flipped_channel_test_grid(Nx, Nz, Ny)

        # Build the same spectral content on both grids.
        # Leading-inhomogeneous shape: (Ny, (Nx>>1)+1, Nz)
        # Leading-homogeneous shape: ((Nx>>1)+1, Nz, Ny)
        u_tri  = FTField(g_tri)
        u_flip = FTField(g_flip)

        # Set a single non-zero coefficient at (kx=1, kz=0) on each grid.
        # Leading-inhomogeneous: kx is axis 2 and kz is axis 3.
        # Leading-homogeneous: kx is axis 1 and kz is axis 2.
        parent(u_tri)[1, 2, 1]  = 1.0 + 0im   # y=1, kx=1, kz=0 (rfft idx 2 → kx=1)
        parent(u_flip)[2, 1, 1] = 1.0 + 0im   # kx=1 (rfft idx 2), kz=0, y=1

        shifts = (1.0, 0.0)   # one shift per FFT dim

        shift!(u_tri,  shifts)
        shift!(u_flip, shifts)

        # Phase factor for kx=1 with shift=1.0 and wavenumber_scale=1.0:
        #   exp(i * 1 * 1.0 * 1.0) — both grids use scale 1
        expected_phase = exp(im * 1.0 * 1.0)

        @test parent(u_tri)[1, 2, 1]  ≈ expected_phase
        @test parent(u_flip)[2, 1, 1] ≈ expected_phase
    end
end
