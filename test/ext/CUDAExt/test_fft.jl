@testset verbose=true "CUDA FFT plans                                              " begin
    @testset verbose=true "Transform utilities                                         " begin
        @testset verbose=true "_apply_mask!                                                " begin
            for sz in ((5,), (5, 6), (3, 4, 5))
                cache = CUDA.randn(ComplexF64, sz...)
                NSEBase._apply_mask!(cache)
                @test all(iszero, Array(cache))
            end
        end

        @testset verbose=true "_copy_to/from_padded!, 1D                                   " begin
            # NTuple{1}: only the rfft dim is transformed
            M, M_pad = 8, NSEBase.get_padded_size((8,), (1,))[1]
            M_spec, M_spec_pad = M ÷ 2 + 1, M_pad ÷ 2 + 1

            u     = CUDA.randn(ComplexF32, M_spec)
            cache = CUDA.zeros(ComplexF32, M_spec_pad)

            NSEBase._apply_mask!(cache)
            NSEBase._copy_to_padded!(cache, u, (1,))

            # resolved range is embedded at the low end; padding zone stays zero
            @test Array(cache[1:M_spec]) ≈ Array(u)
            @test all(iszero, Array(cache)[M_spec+1:end])

            # round-trip
            u2 = CUDA.zeros(ComplexF32, M_spec)
            NSEBase._copy_from_padded!(u2, cache, (1,))
            @test Array(u2) ≈ Array(u)
        end

        @testset verbose=true "_copy_to/from_padded!, 2D                                   " begin
            # NTuple{2}: rfft dim + one full-spectrum dim
            M, N   = 8, 6
            M_pad, N_pad = NSEBase.get_padded_size((M, N), (1, 2))
            M_spec, M_spec_pad = M ÷ 2 + 1, M_pad ÷ 2 + 1

            u     = CUDA.randn(ComplexF64, M_spec, N)
            cache = CUDA.zeros(ComplexF64, M_spec_pad, N_pad)

            NSEBase._apply_mask!(cache)
            NSEBase._copy_to_padded!(cache, u, (1, 2))

            # round-trip
            u2 = CUDA.zeros(ComplexF64, M_spec, N)
            NSEBase._copy_from_padded!(u2, cache, (1, 2))
            @test Array(u2) ≈ Array(u)

            # positive-frequency block is at the low end of each dim
            N_pos = (N >> 1) + 1
            @test Array(cache)[1:M_spec, 1:N_pos] ≈ Array(u)[:, 1:N_pos]

            # negative-frequency block is at the high end of dim 2
            N_neg = N - N_pos
            if N_neg > 0
                @test Array(cache)[1:M_spec, N_pad-N_neg+1:N_pad] ≈ Array(u)[:, N_pos+1:end]
            end
        end

        @testset verbose=true "_copy_to/from_padded!, 3D                                   " begin
            # NTuple{3}: rfft dim + two full-spectrum dims
            L, M, N  = 4, 6, 8
            L_pad, M_pad, N_pad = NSEBase.get_padded_size((L, M, N), (1, 2, 3))
            L_spec, L_spec_pad  = L ÷ 2 + 1, L_pad ÷ 2 + 1

            u     = CUDA.randn(ComplexF64, L_spec, M, N)
            cache = CUDA.zeros(ComplexF64, L_spec_pad, M_pad, N_pad)

            NSEBase._apply_mask!(cache)
            NSEBase._copy_to_padded!(cache, u, (1, 2, 3))

            u2 = CUDA.zeros(ComplexF64, L_spec, M, N)
            NSEBase._copy_from_padded!(u2, cache, (1, 2, 3))
            @test Array(u2) ≈ Array(u)
        end

        @testset verbose=true "_add_from_padded!, 1D                                       " begin
            M, M_pad = 8, NSEBase.get_padded_size((8,), (1,))[1]
            M_spec, M_spec_pad = M ÷ 2 + 1, M_pad ÷ 2 + 1

            cache = CUDA.randn(ComplexF64, M_spec_pad)
            accum = CUDA.randn(ComplexF64, M_spec)
            accum0 = copy(accum)

            NSEBase._add_from_padded!(accum, cache, (1,))
            @test Array(accum) ≈ Array(accum0) .+ Array(cache)[1:M_spec]
        end

        @testset verbose=true "_add_from_padded!, 2D                                       " begin
            M, N   = 8, 6
            M_pad, N_pad = NSEBase.get_padded_size((M, N), (1, 2))
            M_spec = M ÷ 2 + 1

            cache = CUDA.randn(ComplexF64, (M_pad ÷ 2 + 1), N_pad)
            accum = CUDA.randn(ComplexF64, M_spec, N)
            accum0 = copy(accum)

            NSEBase._add_from_padded!(accum, cache, (1, 2))

            # positive-frequency block
            N_pos = (N >> 1) + 1
            @test Array(accum)[:, 1:N_pos] ≈ Array(accum0)[:, 1:N_pos] .+ Array(cache)[1:M_spec, 1:N_pos]

            # negative-frequency block
            N_neg = N - N_pos
            if N_neg > 0
                @test Array(accum)[:, N_pos+1:end] ≈ Array(accum0)[:, N_pos+1:end] .+ Array(cache)[1:M_spec, N_pad-N_neg+1:N_pad]
            end
        end
    end

    @testset verbose=true "Plan construction                                           " begin
        for T in [Float64, Float32]
            # 1D: cache is the standard spectral shape, norm = 1/N, fixed threads
            p = CUDAExt._make_cufft_plans((8,), (1,), T)
            @test p.norm ≈ T(1/13)
            @test size(p.cache) == (7,)

            # 2D, rfft dim only: dim-2 untransformed, norm = 1/M, fixed threads
            p = CUDAExt._make_cufft_plans((8, 6), (1,), T)
            @test p.norm == T(1/13)
            @test size(p.cache) == (7, 6)

            # 2D, both dims transformed: norm = 1/(M*N), fixed threads
            p = CUDAExt._make_cufft_plans((8, 6), (1, 2), T)
            @test p.norm == T(1/(13*9))
            @test size(p.cache) == (7, 9)

            # 3D, all dims transformed
            p = CUDAExt._make_cufft_plans((4, 6, 8), (1, 2, 3), T)
            @test p.norm == T(1/(7*9*13))
            @test size(p.cache) == (4, 9, 13)
        end

        # Production-grid constructor.
        g = CUDA_CHANNEL_GRID
        gd = CUDA.cu(g)
        p = FFTPlans(gd)
        padded_size = NSEBase.get_padded_size(size(g), fft_storage_dims(g))
        expected_norm = inv(prod(padded_size[dim] for dim in fft_storage_dims(g)))
        @test p.norm == Float32(expected_norm)
        @test size(p.cache) == NSEBase._get_transform_size(padded_size, rfft_storage_dim(g))
        @test p.backend == cuFFT
    end

    @testset verbose=true "Transform execution                                         " begin
        # construct plans
        sz = (4, 6, 8)
        odr = (1, 2, 3)
        p = CUDAExt._make_cufft_plans(sz, odr, Float32)

        # construct fields to be transformed
        pad_sz = NSEBase.get_padded_size(sz, odr)
        u = CUDA.randn(Float32, pad_sz)
        u2 = CUDA.zeros(Float32, pad_sz)
        û = CUDA.zeros(ComplexF32, NSEBase._get_transform_size(pad_sz, odr[1]))
        û_accum = CUDA.zeros(ComplexF32, NSEBase._get_transform_size(pad_sz, odr[1]))

        # test round-trip
        @test Array(p(similar(u), p(û, u, false, true), true, true))  ≈ Array(u)
        @test Array(p(similar(û), p(u2, û, true, true), false, true)) ≈ Array(û)

        # test accumulation
        p(û_accum, u, true, true)
        p(û_accum, u, true, true)
        @test Array(p(similar(u), û_accum, true, true)) ≈ 2*Array(u)
    end
end
