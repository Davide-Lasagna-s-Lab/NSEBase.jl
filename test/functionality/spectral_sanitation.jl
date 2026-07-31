# Contract tests for the raw spectral sanitation utilities shared by FTField
# and ProjectedField. Wrapper construction, indexing, and broadcasting live in
# their dedicated files; this file exercises dimension-order-independent logic.

function _raw_dc_is_hermitian(data, fft_dims)
    length(fft_dims) <= 1 && return true
    rfft_dim = first(fft_dims)
    signed_dims = Base.tail(fft_dims)
    ranges = ntuple(d -> d == rfft_dim ? (1:1) : axes(data, d), ndims(data))

    for I in CartesianIndices(ranges)
        partner = CartesianIndex(ntuple(ndims(data)) do d
            d in signed_dims && I[d] != 1 ? size(data, d) - I[d] + 2 : I[d]
        end)
        data[I] == conj(data[partner]) || return false
    end
    return true
end

function _raw_non_dc_is_unchanged(after, before, rfft_dim)
    for I in CartesianIndices(after)
        I[rfft_dim] == 1 && continue
        after[I] == before[I] || return false
    end
    return true
end

@testset verbose=true "Raw spectral sanitation utilities                           " begin
    @testset verbose=true "Zero or one transformed dimension is an exact no-op         " begin
        rng = MersenneTwister(91)
        for shape in ((11,), (5, 7), (3, 5, 7), (3, 5, 7, 9))
            data = randn(rng, ComplexF64, shape)
            no_fft = copy(data)
            one_fft = copy(data)
            singleton_dim = (ndims(data),)

            @test NSEBase.apply_symmetry!(no_fft, ()) === no_fft
            @test no_fft == data
            @test NSEBase.apply_symmetry!(one_fft, singleton_dim) === one_fft
            @test one_fft == data
        end
    end

    @testset verbose=true "Two transformed dimensions honor arbitrary storage order    " begin
        rng = MersenneTwister(92)
        for fft_dims in ((2, 3), (3, 1))
            data = randn(rng, ComplexF64, 7, 5, 9)
            before = copy(data)
            repaired = NSEBase.apply_symmetry!(data, fft_dims)

            @test repaired === data
            @test _raw_dc_is_hermitian(repaired, fft_dims)
            @test _raw_non_dc_is_unchanged(repaired, before, first(fft_dims))
        end

        via_val = randn(rng, ComplexF64, 7, 5, 9)
        @test NSEBase.apply_symmetry!(via_val, Val((2, 3))) === via_val
        @test _raw_dc_is_hermitian(via_val, (2, 3))
    end

    @testset verbose=true "Several signed dimensions are reflected simultaneously      " begin
        rng = MersenneTwister(93)
        cases = ((3, 4), (4, 2), (1, 2, 3), (3, 2, 4))
        for fft_dims in cases
            data = randn(rng, ComplexF64, 5, 7, 9, 11)
            before = copy(data)
            repaired = NSEBase.apply_symmetry!(data, fft_dims)

            @test repaired === data
            @test _raw_dc_is_hermitian(repaired, fft_dims)
            @test _raw_non_dc_is_unchanged(repaired, before, first(fft_dims))

            snapshot = copy(repaired)
            NSEBase.apply_symmetry!(repaired, fft_dims)
            @test repaired == snapshot
        end
    end

    @testset verbose=true "Conjugate pairs are replaced by their nearest average       " begin
        data = zeros(ComplexF64, 2, 1, 3)
        data[1, 1, 2] = 1 + 4im
        data[1, 1, 3] = 5 + 2im
        data[2, 1, 2] = -2 + 6im
        data[2, 1, 3] = 4 - 2im

        NSEBase.apply_symmetry!(data, (2, 3))
        @test data[1, 1, 2] == 3 + 1im
        @test data[1, 1, 3] == 3 - 1im
        @test data[2, 1, 2] == 1 + 4im
        @test data[2, 1, 3] == 1 - 4im
    end

    @testset verbose=true "Mean normalization changes only the all-zero Fourier slice  " begin
        rng = MersenneTwister(94)
        data = randn(rng, ComplexF64, 5, 4, 7)
        before = copy(data)

        @test NSEBase.normalise_mean!(data, (2, 3)) === data
        @test data[:, 1, 1] == complex.(real.(before[:, 1, 1]))
        @test data[:, 2:end, :] == before[:, 2:end, :]
        @test data[:, 1, 2:end] == before[:, 1, 2:end]

        via_val = copy(before)
        @test NSEBase.normalise_mean!(via_val, Val((2, 3))) === via_val
        @test via_val == data
    end
end
