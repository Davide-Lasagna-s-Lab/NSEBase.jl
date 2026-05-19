# Helper: collect all index/frequency tuples visited by for_each_wavenumber / for_each_freq
function collect_wavenumbers(g)
    result = NTuple{length(NSEBase.fft_dims(g)), Int}[]
    NSEBase.for_each_wavenumber(g) do _, args...
        push!(result, args)
    end
    return result
end

function collect_freqs(g)
    result = NTuple{length(NSEBase.fft_dims(g)), Int}[]
    NSEBase.for_each_freq(g) do args...
        push!(result, args)
    end
    return result
end

@testset verbose=true "Spectral looping                    " begin
    struct TestGrid{S, D, AXES, ORDER} <: AbstractGrid{Float64, D, AXES, ORDER} end
    Base.size(::TestGrid{S}) where {S} = S

    @testset "for_each_wavenumber" begin
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
        f(grid) = @allocated NSEBase.for_each_wavenumber((args...)->nothing, grid)
        # Warm up
        f(g)
        allocs = f(g)
        @test allocs == 0
    end

    @testset "for_each_freq" begin
        # Grid with one rfft dimension of size 7, ORDER = (1,)
        g = TestGrid{(7,), 1, nothing, (1,)}()
        freqs = collect_freqs(g)
        @test freqs == [(0,), (1,), (2,), (3,)]   # 0:(7>>1) = 0:3
        @test all(f -> f[1] >= 0, freqs)

        # Grid with rfft along dim 1 (size 7) and signed fft along dim 2 (size 5), ORDER = (1, 2)
        g = TestGrid{(7, 5), 2, nothing, (1, 2)}()
        freqs = collect_freqs(g)

        # rfft frequencies: 0:3, signed-fft frequencies: 0:2 (pos) then -2:-1 (neg)
        rfft_freqs = 0:(7 >> 1)           # 0:3
        pos_freqs  = 0:(5 >> 1)           # 0:2
        neg_freqs  = -(5 >> 1):-1         # -2:-1
        expected = [(f1, f2) for f2 in -(5 >> 1):(5 >> 1)
                             for f1 in 0:(7 >> 1)]
        @test freqs == expected

        # frequencies are symmetric
        g = TestGrid{(7, 5), 2, nothing, (1, 2)}()
        freqs = collect_freqs(g)
        f2_vals = sort(unique(last.(freqs)))
        @test f2_vals == collect(-(5 >> 1):(5 >> 1))

        # no duplicate frequencies
        g = TestGrid{(7, 5), 2, nothing, (1, 2)}()
        freqs = collect_freqs(g)
        @test length(freqs) == length(unique(freqs))

        # mode count matches storage size
        g = TestGrid{(7, 5), 2, nothing, (1, 2)}()
        freqs = collect_freqs(g)
        @test length(freqs) == ((7 >> 1) + 1) * 5

        # zero allocations
        g = TestGrid{(7, 5), 2, nothing, (1, 2)}()
        f(grid) = @allocated NSEBase.for_each_freq((args...)->nothing, grid)
        # Warm up
        f(g)
        allocs = f(g)
        @test allocs == 0
    end

    @testset "f_e_m and f_e_f agree" begin
        for sz in [(7,), (5, 7), (5, 7, 9)]
            g = TestGrid{sz, length(sz), nothing, ntuple(identity, length(sz))}()
            n_modes = 0; NSEBase.for_each_wavenumber(g) do args...; n_modes += 1; end
            n_freqs = 0; NSEBase.for_each_freq(g) do args...; n_freqs += 1; end
            @test n_modes == n_freqs
        end
    end
end
