# Helper: collect all index tuples visited by for_each_homogeneous_index.
function collect_wavenumbers(g)
    result = NTuple{length(NSEBase.fft_dims(g)), Int}[]
    NSEBase.for_each_homogeneous_index(g) do _, args...
        push!(result, args)
    end
    return result
end

function collect_wavenumber_vectors(g)
    result = NTuple{length(NSEBase.fft_dims(g)), Int}[]
    NSEBase.for_each_homogeneous_index(g) do _, args...
        k = NSEBase.to_wavenumber_vector(g, args)
        push!(result, ntuple(i -> k[i], length(k)))
    end
    return result
end

@testset verbose=true "Spectral looping                    " begin
    struct TestGrid{S, D, AXES, ORDER} <: AbstractGrid{Float64, D, AXES, ORDER} end
    Base.size(::TestGrid{S}) where {S} = S

    @testset "for_each_homogeneous_index" begin
        # Grid with one rfft dimension of size 7, ORDER = (1,)
        g = TestGrid{(7,), 1, nothing, (1,)}()
        modes = collect_wavenumbers(g)
        @test modes == [(1,), (2,), (3,), (4,)]   # 1:(7>>1)+1 = 1:4
        @test length(modes) == (7 >> 1) + 1

        # Grid with rfft along dim 1 (size 7) and signed fft along dim 2 (size 5), ORDER = (1, 2)
        g = TestGrid{(7, 5), 2, nothing, (1, 2)}()
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
        g = TestGrid{(7, 5), 2, nothing, (1, 2)}()
        modes = collect_wavenumbers(g)
        @test length(modes) == length(unique(modes))

        # all storage indices covered
        g = TestGrid{(7, 5), 2, nothing, (1, 2)}()
        modes = collect_wavenumbers(g)
        all_i1 = sort(unique(first.(modes)))
        all_i2 = sort(unique(last.(modes)))
        @test all_i1 == collect(1:(7 >> 1) + 1)
        @test all_i2 == collect(1:5)

        # zero allocations
        g = TestGrid{(7, 5), 2, nothing, (1, 2)}()
        f(grid) = @allocated NSEBase.for_each_homogeneous_index((args...)->nothing, grid)
        # Warm up
        f(g)
        allocs = f(g)
        @test allocs == 0
    end

    @testset "to_wavenumber_vector order" begin
        # Grid with one rfft dimension of size 7, ORDER = (1,)
        g = TestGrid{(7,), 1, nothing, (1,)}()
        ks = collect_wavenumber_vectors(g)
        @test ks == [(0,), (1,), (2,), (3,)]   # 0:(7>>1) = 0:3
        @test all(k -> k[1] >= 0, ks)

        # Grid with rfft along dim 1 (size 7) and signed fft along dim 2 (size 5), ORDER = (1, 2)
        g = TestGrid{(7, 5), 2, nothing, (1, 2)}()
        ks = collect_wavenumber_vectors(g)

        # rfft wavenumbers: 0:3, signed-FFT wavenumbers: 0:2 (positive block)
        # then -2:-1 (negative block), matching FFTW storage order.
        expected = [(k1, k2) for k2 in [0:(5 >> 1); -(5 >> 1):-1]
                             for k1 in 0:(7 >> 1)]
        @test ks == expected

        # wavenumbers are symmetric
        g = TestGrid{(7, 5), 2, nothing, (1, 2)}()
        ks = collect_wavenumber_vectors(g)
        k2_vals = sort(unique(last.(ks)))
        @test k2_vals == collect(-(5 >> 1):(5 >> 1))

        # no duplicate wavenumbers
        g = TestGrid{(7, 5), 2, nothing, (1, 2)}()
        ks = collect_wavenumber_vectors(g)
        @test length(ks) == length(unique(ks))

        # mode count matches storage size
        g = TestGrid{(7, 5), 2, nothing, (1, 2)}()
        ks = collect_wavenumber_vectors(g)
        @test length(ks) == ((7 >> 1) + 1) * 5
    end
end
