# NSEBase owns the coefficient-space phase action and container semantics of a
# shift. ReSolverRectangularGrids separately checks shifted analytical fields.

@testset verbose=true "Spectral shifts                                             " begin
    @testset verbose=true "Coefficient phase                                           " begin
        g = steady_channel_grid(Ny=9, Nx=9, Nz=7, α=1.5, β=0.75)
        u = FTField(g)
        k = WaveNumberVector(2, -3)
        stored = Base.front(to_homogeneous_indices(g, k))
        parent(u)[1, stored...] = 1

        shifts = (0.2, -0.4)
        expected = cis(k[1] * shifts[1] * wavenumber_scale(g, 2) +
                       k[2] * shifts[2] * wavenumber_scale(g, 3))

        @test shift!(u, shifts) === u
        @test parent(u)[1, stored...] ≈ expected
    end

    @testset verbose=true "Identity and tuple validation                               " begin
        g = steady_channel_grid()
        u = test_ftfield(g)
        before = copy(parent(u))

        @test shift!(u, (0.0, 0.0)) === u
        @test parent(u) == before
    end

    @testset verbose=true "Allocating form and composition                             " begin
        g = steady_channel_grid()
        u = test_ftfield(g)
        before = copy(parent(u))
        a = (0.4, -0.2)
        b = (0.1, 0.7)

        shifted = shift(u, a)
        @test shifted !== u
        @test parent(u) == before
        @test parent(shift!(shift!(copy(u), a), b)) ≈ parent(shift!(copy(u), a .+ b))
    end

    @testset verbose=true "VectorField components                                      " begin
        g = steady_channel_grid()
        u = VectorField(ntuple(component -> test_ftfield(g; seed=component), 3)...)
        expected = ntuple(component -> shift!(copy(u[component]), (0.3, -0.5)), 3)

        @test shift!(u, (0.3, -0.5)) === u
        @test all(parent(u[n]) ≈ parent(expected[n]) for n in 1:3)
    end

    @testset verbose=true "ProjectedField mode axis                                    " begin
        g = steady_channel_grid()
        modes = test_modes(g; ncomponents=3, nmodes=5)
        a = ProjectedField(g, randn(MersenneTwister(4), ComplexF64, 5, 5, 7), modes)
        magnitudes = abs.(parent(a))

        @test shift!(a, (0.31, -0.17)) === a
        @test abs.(parent(a)) ≈ magnitudes
    end
end
