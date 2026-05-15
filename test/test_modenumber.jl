@testset verbose=true "Mode number conversion              " begin
    # utility grid for testing
    struct TempGrid{S, H} <: AbstractGrid{Float64, 4, (1, 2, 3, 4), H} end
    Base.size(g::TempGrid{S}) where {S} = S

    @testset "1D                            " begin
        N1 = 23 # N1 can be even or odd
        g = TempGrid{(16, N1, 16, 16), (2,)}()

        n1s = collect(0:(N1 >> 1) + 1)

        # Non-conjugate half: n1 ≥ 0, forward ordering
        for _n1 in 1:(N1 >> 1) + 1
            n = ModeNumber(n1s[_n1])
            @test NSEBase._modenumber_to_indices(g, n) == (_n1, false)
        end

        n1s = collect(0:-1:-(N1 >> 1))

        # Conjugate half: n1 < 0, reversed ordering
        for _n1 in 2:(N1 >> 1) + 1
            n = ModeNumber(n1s[_n1])
            @test NSEBase._modenumber_to_indices(g, n) == (_n1, true)
        end
    end

    @testset "2D                            " begin
        N1 = 23 # N1 can be even or odd
        N2 = 23 # N2 has to be odd
        g = TempGrid{(16, N1, N2, 16), (2, 3)}()

        n1s = collect(0:(N1 >> 1) + 1)
        n2s = [collect(0:(N2 >> 1)); collect(-(N2 >> 1):-1)]

        # Non-conjugate half: n1 ≥ 0, forward ordering
        for _n1 in 1:(N1 >> 1) + 1, _n2 in 1:N2
            n = ModeNumber(n1s[_n1], n2s[_n2])
            @test NSEBase._modenumber_to_indices(g, n) == (_n1, _n2, false)
        end

        n1s = collect(0:-1:-(N1 >> 1))
        n2s = [[0]; collect(-1:-1:-(N2 >> 1)); collect((N2 >> 1):-1:1)]

        # Conjugate half: n1 < 0, reversed ordering
        for _n1 in 2:(N1 >> 1) + 1, _n2 in 1:N2
            n = ModeNumber(n1s[_n1], n2s[_n2])
            @test NSEBase._modenumber_to_indices(g, n) == (_n1, _n2, true)
        end
    end

    @testset "3D                            " begin
        N1 = 23 # N1 can be even or odd
        N2 = 23 # Nz has to be odd
        N3 = 23 # N3 has to be odd
        g = TempGrid{(16, N1, N2, N3), (2, 3, 4)}()

        n1s = collect(0:(N1 >> 1) + 1)
        n2s = [collect(0:(N2 >> 1)); collect(-(N3 >> 1):-1)]
        n3s = [collect(0:(N3 >> 1)); collect(-(N3 >> 1):-1)]

        # Non-conjugate half: n1 ≥ 0, forward ordering
        for _n1 in 1:(N1 >> 1) + 1, _n2 in 1:N2, _n3 in 1:N3
            n = ModeNumber(n1s[_n1], n2s[_n2], n3s[_n3])
            @test NSEBase._modenumber_to_indices(g, n) == (_n1, _n2, _n3, false)
        end

        n1s = collect(0:-1:-(N1 >> 1))
        n2s = [[0]; collect(-1:-1:-(N2 >> 1)); collect((N2 >> 1:-1:1))]
        n3s = [[0]; collect(-1:-1:-(N3 >> 1)); collect((N3 >> 1:-1:1))]

        # Conjugate half: n1 < 0, reversed ordering
        for _n1 in 2:(N1 >> 1) + 1, _n2 in 1:N2, _n3 in 1:N3
            n = ModeNumber(n1s[_n1], n2s[_n2], n3s[_n3])
            @test NSEBase._modenumber_to_indices(g, n) == (_n1, _n2, _n3, true)
        end
    end
end
