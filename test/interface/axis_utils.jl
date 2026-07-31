# Axis helpers must follow grid metadata rather than assume that bounded axes
# precede Fourier axes. Two production rectangular layouts exercise both
# storage orders.

@testset verbose=true "FTField homogeneous and inhomogeneous axes                  " begin
    @testset verbose=true "Bounded storage first                                       " begin
        g = steady_channel_grid(Ny=9, Nx=9, Nz=7)
        u = FTField(g)

        @test size(u) == (9, 5, 7)
        @test NSEBase.homogeneous_axes(u) == (axes(u, 2), axes(u, 3))
        @test NSEBase.inhomogeneous_axes(u) == (axes(u, 1),)
        @test length(CartesianIndices(NSEBase.homogeneous_axes(u))) == 5 * 7
        @test length(CartesianIndices(NSEBase.inhomogeneous_axes(u))) == 9
    end

    @testset verbose=true "Fourier storage first                                       " begin
        g = flipped_grid(Ny=9, Nx=9, Nz=7)
        u = FTField(g)

        @test size(u) == (5, 7, 9)
        @test NSEBase.homogeneous_axes(u) == (axes(u, 1), axes(u, 2))
        @test NSEBase.inhomogeneous_axes(u) == (axes(u, 3),)
        @test length(CartesianIndices(NSEBase.homogeneous_axes(u))) == 5 * 7
        @test length(CartesianIndices(NSEBase.inhomogeneous_axes(u))) == 9
    end

    @testset verbose=true "Storage permutation preserves phase action                  " begin
        standard = FTField(steady_channel_grid(Ny=9, Nx=9, Nz=7))
        flipped = FTField(flipped_grid(Ny=9, Nx=9, Nz=7))

        # Both locations represent the same `(kx=1, kz=0, y-index=1)` mode.
        parent(standard)[1, 2, 1] = 1
        parent(flipped)[2, 1, 1] = 1

        @test shift!(standard, (1.0, 0.0)) === standard
        @test shift!(flipped, (1.0, 0.0)) === flipped
        @test parent(standard)[1, 2, 1] ≈ cis(1.0)
        @test parent(flipped)[2, 1, 1] ≈ cis(1.0)
    end
end
