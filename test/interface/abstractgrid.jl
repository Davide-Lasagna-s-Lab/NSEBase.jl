# Contract tests for the grid metadata and required interface consumed by
# NSEBase. Numerical accuracy of the FDGrids geometry belongs to
# ReSolverRectangularGrids; these tests only exercise NSEBase's interpretation
# of a production grid.

@testset verbose=true "AbstractGrid interface                                      " begin
    @testset verbose=true "Cartesian and storage dimensions                            " begin
        g = channel_grid()

        @test g isa AbstractGrid{Float64, 4}
        @test size(g) == (15, 7, 9, 5)
        @test ntuple(dim -> size(g, dim), 4) == size(g)

        # Physical-to-storage accessors expose the AXES type parameter through
        # both runtime Symbols and compile-time Val arguments.
        @test storage_dim(g, :x) == 2
        @test storage_dim(g, :y) == 1
        @test storage_dim(g, :z) == 3
        @test storage_dim(g, :t) == 4
        @test storage_dim(g, Val(:x)) == 2
        @test physical_to_storage_dim(g, :y) == Val(1)
        @test physical_to_storage_dim(g, Val(:t)) == Val(4)

        @test physical_dim(g, 1) == :y
        @test physical_dim(g, 2) == :x
        @test physical_dim(g, Val(3)) == :z
        @test physical_dim(g, nothing) === nothing
        @test physical_dim(g, Val(nothing)) === nothing

        @test to_storage_order((:X, :Y, :Z, :T), g) == (:Y, :X, :Z, :T)
        @test_throws ArgumentError storage_dim(g, :q)
        @test_throws ArgumentError physical_dim(g, 0)
        @test_throws ArgumentError physical_dim(g, 5)
        @test_throws MethodError to_storage_order((:X, :Y), g)
    end

    @testset verbose=true "Homogeneous and inhomogeneous metadata                      " begin
        g = channel_grid()

        @test fft_storage_dims(g) === (2, 3, 4)
        @test spatial_fft_storage_dims(g) === (2, 3)
        @test inhomogeneous_storage_dims(g) === (1,)
        @test spatial_inhomogeneous_storage_dims(g) === (1,)
        @test rfft_storage_dim(g) === 2

        @test fft_physical_dims(g) == (:x, :z, :t)
        @test spatial_fft_physical_dims(g) == (:x, :z)
        @test inhomogeneous_physical_dims(g) == (:y,)
        @test spatial_inhomogeneous_physical_dims(g) == (:y,)
        @test rfft_physical_dim(g) == :x

        # Missing Cartesian coordinates are represented by `nothing`, while
        # present time remains excluded from the spatial FFT subset.
        gst = spacetime_grid()
        @test storage_dim(gst, :z) === nothing
        @test physical_to_storage_dim(gst, :z) == Val(nothing)
        @test fft_physical_dims(gst) == (:x, :t)
        @test spatial_fft_physical_dims(gst) == (:x,)
        @test to_storage_order((:X, :Y, :Z, :T), gst) == (:Y, :X, :T)
    end

    @testset verbose=true "Derived sizes, coordinates, and conversion                  " begin
        g = channel_grid()

        @test transform_size(g) == (15, 4, 9, 5)
        @test fft_norm(g) == (7, 9, 5)
        @test map(size, points(g)) == ((15, 1, 1, 1), (1, 7, 1, 1),
                                       (1, 1, 9, 1), (1, 1, 1, 5))
        @test size(weights(g)) == (15,)
        @test wavenumber_scale(g, storage_dim(g, :x)) == 1
        @test wavenumber_scale(g, storage_dim(g, :z)) == 1
        @test wavenumber_scale(g, storage_dim(g, :t)) == 2π

        grown = growto(g, (11, 7, 3))
        @test size(grown) == (15, 11, 7, 3)
        @test fft_storage_dims(grown) == fft_storage_dims(g)
        @test points(grown)[1] == points(g)[1]

        g32 = convert(Float32, g)
        @test eltype(g32) === Float32
        @test size(g32) == size(g)
        @test fft_storage_dims(g32) == fft_storage_dims(g)
    end

    @testset verbose=true "Storage-order permutations                                  " begin
        g = flipped_grid()

        @test size(g) == (9, 7, 9)
        @test fft_storage_dims(g) == (1, 2)
        @test inhomogeneous_storage_dims(g) == (3,)
        @test fft_physical_dims(g) == (:x, :z)
        @test inhomogeneous_physical_dims(g) == (:y,)
        @test transform_size(g) == (5, 7, 9)
        @test to_storage_order((:X, :Y, :Z, :T), g) == (:X, :Z, :Y)
    end

    @testset verbose=true "Index partitioning                                          " begin
        g = steady_channel_grid()
        I = CartesianIndex(2, 3, 4)

        @test NSEBase.one_or_two(CartesianIndex(2, 1, 4), g) == 1
        @test NSEBase.one_or_two(I, g) == 2
        @test NSEBase.inhomogeneous_indices(I, g) == (2,)
        @test NSEBase.homogeneous_indices(I, g) == (3, 4)
        @test NSEBase.combine_indices(g, CartesianIndex(2), CartesianIndex(3, 4)) == (2, 3, 4)
        @test NSEBase.combine_indices(g, :, CartesianIndex(3, 4)) == (Colon(), 3, 4)
    end
end
