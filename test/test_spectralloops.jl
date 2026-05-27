# Helper: collect all index tuples visited by for_each_homogeneous_index.
function collect_wavenumbers(g)
    result = NTuple{length(NSEBase.fft_dims(g)), Int}[]
    NSEBase.for_each_homogeneous_index(g) do _, idx
        push!(result, idx)
    end
    return result
end

function collect_wavenumber_vectors(g)
    result = NTuple{length(NSEBase.fft_dims(g)), Int}[]
    NSEBase.for_each_homogeneous_index(g) do _, idx
        k = NSEBase.to_wavenumber_vector(g, idx)
        push!(result, ntuple(i -> k[i], length(k)))
    end
    return result
end

@testset verbose=true "Spectral looping                                                    " begin
    @testset "for_each_homogeneous_index" begin
        # Grid with one rfft dimension of size 7, FFT_DIMS_ORDER = (1,)
        g = SpectralTestGrid{(7,), 1, (1, nothing, nothing, nothing), (1,)}()
        modes = collect_wavenumbers(g)
        @test modes == [(1,), (2,), (3,), (4,)]   # 1:(7>>1)+1 = 1:4
        @test length(modes) == (7 >> 1) + 1

        # Grid with rfft along dim 1 (size 7) and signed fft along dim 2 (size 5), FFT_DIMS_ORDER = (1, 2)
        g = SpectralTestGrid{(7, 5), 2, (1, 2, nothing, nothing), (1, 2)}()
        modes = collect_wavenumbers(g)

        # rfft indices: 1:4, signed-fft indices: 1:3 (pos) then 4:5 (neg)
        rfft_indices  = 1:(7 >> 1) + 1          # 1:4
        pos_indices   = 1:(5 >> 1) + 1          # 1:3
        neg_indices   = (5 >> 1) + 2:5          # 4:5
        expected = [(i1, i2) for i2 in [pos_indices; neg_indices]
                              for i1 in rfft_indices]
        @test modes == expected
        @test length(modes) == ((7 >> 1) + 1) * 5

        # no duplicate indices
        g = SpectralTestGrid{(7, 5), 2, (1, 2, nothing, nothing), (1, 2)}()
        modes = collect_wavenumbers(g)
        @test length(modes) == length(unique(modes))

        # all storage indices covered
        g = SpectralTestGrid{(7, 5), 2, (1, 2, nothing, nothing), (1, 2)}()
        modes = collect_wavenumbers(g)
        all_i1 = sort(unique(first.(modes)))
        all_i2 = sort(unique(last.(modes)))
        @test all_i1 == collect(1:(7 >> 1) + 1)
        @test all_i2 == collect(1:5)

        # zero allocations
        g = SpectralTestGrid{(7, 5), 2, (1, 2, nothing, nothing), (1, 2)}()
        f(grid) = @allocated NSEBase.for_each_homogeneous_index((_, idx)->nothing, grid)
        # Warm up
        f(g)
        allocs = f(g)
        @test allocs == 0
    end

    @testset "to_wavenumber_vector order" begin
        # Grid with one rfft dimension of size 7, FFT_DIMS_ORDER = (1,)
        g = SpectralTestGrid{(7,), 1, (1, nothing, nothing, nothing), (1,)}()
        ks = collect_wavenumber_vectors(g)
        @test ks == [(0,), (1,), (2,), (3,)]   # 0:(7>>1) = 0:3
        @test all(k -> k[1] >= 0, ks)

        # Grid with rfft along dim 1 (size 7) and signed fft along dim 2 (size 5), FFT_DIMS_ORDER = (1, 2)
        g = SpectralTestGrid{(7, 5), 2, (1, 2, nothing, nothing), (1, 2)}()
        ks = collect_wavenumber_vectors(g)

        # rfft wavenumbers: 0:3, signed-FFT wavenumbers: 0:2 (positive block)
        # then -2:-1 (negative block), matching FFTW storage order.
        expected = [(k1, k2) for k2 in [0:(5 >> 1); -(5 >> 1):-1]
                             for k1 in 0:(7 >> 1)]
        @test ks == expected

        # wavenumbers are symmetric
        g = SpectralTestGrid{(7, 5), 2, (1, 2, nothing, nothing), (1, 2)}()
        ks = collect_wavenumber_vectors(g)
        k2_vals = sort(unique(last.(ks)))
        @test k2_vals == collect(-(5 >> 1):(5 >> 1))

        # no duplicate wavenumbers
        g = SpectralTestGrid{(7, 5), 2, (1, 2, nothing, nothing), (1, 2)}()
        ks = collect_wavenumber_vectors(g)
        @test length(ks) == length(unique(ks))

        # mode count matches storage size
        g = SpectralTestGrid{(7, 5), 2, (1, 2, nothing, nothing), (1, 2)}()
        ks = collect_wavenumber_vectors(g)
        @test length(ks) == ((7 >> 1) + 1) * 5
    end
end
