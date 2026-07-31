# Wavenumber conversion is an NSEBase storage contract. These tests use real
# rectangular layouts but compare only the two public index representations;
# they do not retest any grid's numerical discretisation.

@testset verbose=true "WaveNumberVector conversion                                 " begin
    @testset verbose=true "Tuple-like interface                                        " begin
        k = WaveNumberVector(2, -1, 0)
        @test length(k) == 3
        @test k[1] == 2
        @test k[2] == -1
        @test k[1:3] == (2, -1, 0)
    end

    @testset verbose=true "One, two, and three Fourier directions                      " begin
        grids = (line_grid(Nh=23), steady_channel_grid(Nx=11, Nz=9),
                 channel_grid(Nx=9, Nz=7, Nt=5))

        for g in grids
            fft_axes = map(dim -> Base.OneTo(transform_size(g)[dim]), fft_storage_dims(g))
            for I in CartesianIndices(fft_axes)
                k = to_wavenumber_vector(g, I)

                # Stored rfft wavenumbers are non-negative, so converting a
                # storage index to a wavenumber and back needs no conjugation.
                @test Base.front(to_homogeneous_indices(g, k)) == Tuple(I)
                @test last(to_homogeneous_indices(g, k)) === false
                @test to_wavenumber_vector(g, Tuple(I)) == k

                # The opposite full wavenumber represents the same stored
                # coefficient through Hermitian conjugation when k₁ > 0.
                if k[1] > 0
                    k_conjugate = WaveNumberVector(ntuple(j -> -k[j], length(k)))
                    @test Base.front(to_homogeneous_indices(g, k_conjugate)) == Tuple(I)
                    @test last(to_homogeneous_indices(g, k_conjugate)) === true
                end
            end
        end
    end

    @testset verbose=true "Dimension mismatch                                          " begin
        g = steady_channel_grid()
        @test_throws ArgumentError to_wavenumber_vector(g, (1,))
        @test_throws ArgumentError to_wavenumber_vector(g, CartesianIndex(1, 2, 3))
    end

    @testset verbose=true "Combining homogeneous and inhomogeneous indices             " begin
        CI = CartesianIndex

        # The Val overload expresses the documented interleaving contract
        # directly and needs no synthetic grid type.
        @test NSEBase.combine_indices(Val(()), CI(10, 20, 30), CI()) == (10, 20, 30)
        @test NSEBase.combine_indices(Val((1,)), CI(20, 30), CI(10)) == (10, 20, 30)
        @test NSEBase.combine_indices(Val((2,)), CI(10, 30), CI(20)) == (10, 20, 30)
        @test NSEBase.combine_indices(Val((3,)), CI(10, 20), CI(30)) == (10, 20, 30)
        @test NSEBase.combine_indices(Val((1, 2)), CI(30), CI(10, 20)) == (10, 20, 30)
        @test NSEBase.combine_indices(Val((1, 3)), CI(20), CI(10, 30)) == (10, 20, 30)
        @test NSEBase.combine_indices(Val((2, 3)), CI(10), CI(20, 30)) == (10, 20, 30)
        @test NSEBase.combine_indices(Val((1, 2, 3)), CI(), CI(10, 20, 30)) == (10, 20, 30)
        @test NSEBase.combine_indices(Val((1, 3, 5)), CI(20, 40), CI(10, 30, 50)) ==
              (10, 20, 30, 40, 50)
        @test NSEBase.combine_indices(Val((2, 4, 6)), CI(10, 30, 50), CI(20, 40, 60)) ==
              (10, 20, 30, 40, 50, 60)
    end
end
