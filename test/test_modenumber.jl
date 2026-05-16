@testset verbose=true "Mode number conversion              " begin
    # utility grid for testing
    struct TempGrid{S, H} <: AbstractGrid{Float64, 4, (1, 2, 3, 4), H} end
    Base.size(::TempGrid{S}) where {S} = S

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

    @testset "combined indexing         " begin
        struct MockGrid{D, AXES, ORDER} <: AbstractGrid{Float64, D, AXES, ORDER} end

        # ── ORDER = () : no constrained dims, all free ────────────────────────────
        @test NSEBase._combine_indices(MockGrid{3, nothing, ()}(), (10, 20, 30), ()) == (10, 20, 30)

        # ── ORDER length 1 ────────────────────────────────────────────────────────
        # ORDER = (1,): dim 1 → K, dims 2-3 → I
        @test NSEBase._combine_indices(MockGrid{3, nothing, (1,)}(), (20, 30), (10,)) == (10, 20, 30)

        # ORDER = (2,): dims 1,3 → I; dim 2 → K
        @test NSEBase._combine_indices(MockGrid{3, nothing, (2,)}(), (10, 30), (20,)) == (10, 20, 30)

        # ORDER = (3,): dims 1,2 → I; dim 3 → K
        @test NSEBase._combine_indices(MockGrid{3, nothing, (3,)}(), (10, 20), (30,)) == (10, 20, 30)

        # ORDER = (1,): dims () → I; dim 1 → K
        @test NSEBase._combine_indices(MockGrid{1, nothing, (1,)}(), (), (42,)) == (42,)

        # # ── ORDER length 2 ────────────────────────────────────────────────────────
        # ORDER = (1,2): dims 1,2 → K; dim 3 → I
        @test NSEBase._combine_indices(MockGrid{3, nothing, (1, 2)}(), (30,), (10, 20)) == (10, 20, 30)

        # ORDER = (2,3): dim 1 → I; dims 2,3 → K
        @test NSEBase._combine_indices(MockGrid{3, nothing, (2, 3)}(), (10,), (20, 30)) == (10, 20, 30)

        # ORDER = (1,3): dims 1,3 → K; dim 2 → I
        @test NSEBase._combine_indices(MockGrid{3, nothing, (1, 3)}(), (20,), (10, 30)) == (10, 20, 30)

        # ORDER = (1,2): both dims → K, no free dims
        @test NSEBase._combine_indices(MockGrid{2, nothing, (1, 2)}(), (), (10, 20)) == (10, 20)

        # ── ORDER length 3 ────────────────────────────────────────────────────────
        # ORDER = (1,2,3): every dim → K, no free dims
        @test NSEBase._combine_indices(MockGrid{3, nothing, (1, 2, 3)}(), (), (10, 20, 30)) == (10, 20, 30)

        # ORDER = (1,3,5): dims 1,3,5 → K; dims 2,4 → I
        @test NSEBase._combine_indices(MockGrid{5, nothing, (1, 3, 5)}(), (20, 40), (10, 30, 50)) == (10, 20, 30, 40, 50)

        # ORDER = (3,4,5): dims 1,2 → I; dims 3,4,5 → K
        @test NSEBase._combine_indices(MockGrid{5, nothing, (3, 4, 5)}(), (10, 20), (30, 40, 50)) == (10, 20, 30, 40, 50)

        # ORDER = (2,4,6): even dims → K; odd dims → I
        @test NSEBase._combine_indices(MockGrid{6, nothing, (2, 4, 6)}(), (10, 30, 50), (20, 40, 60)) == (10, 20, 30, 40, 50, 60)
    end
end
