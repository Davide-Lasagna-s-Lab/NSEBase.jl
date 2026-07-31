# These tests cover NSEBase's derivative routing, spectral multiplier, mode
# forwarding, and composition. Accuracy against analytical functions remains
# with ReSolverRectangularGrids, which owns the FDGrids discretisation.

@testset verbose=true "Derivative operators                                        " begin
    @testset verbose=true "Physical wrappers and absent coordinates                    " begin
        g = line_grid()
        u = test_ftfield(g)

        for (wrapper!, physical_direction) in ((ddx!, :x), (ddy!, :y))
            wrapped = FTField(g)
            primitive = FTField(g)
            @test wrapper!(wrapped, u) === wrapped
            dd!(primitive, u, physical_to_storage_dim(g, physical_direction))
            @test parent(wrapped) ≈ parent(primitive)
        end

        untouched = test_ftfield(g; seed=2)
        before = copy(parent(untouched))
        @test ddz!(untouched, u) === untouched
        @test parent(untouched) == before
        @test ddt!(untouched, u) === untouched
        @test parent(untouched) == before
    end

    @testset verbose=true "Fourier multiplier and discrete adjoint                     " begin
        g = line_grid(scale=1.75)
        u = test_ftfield(g)
        forward = ddy!(FTField(g), u)
        adjoint_result = ddy!(FTField(g), u, AdjointDiscrete())
        scale = wavenumber_scale(g, storage_dim(g, :y))

        for index in axes(parent(u), 2)
            wavenumber = index - 1
            @test parent(forward)[:, index] ≈
                  (im * wavenumber * scale) .* parent(u)[:, index]
        end
        @test parent(adjoint_result) ≈ -parent(forward)
    end

    @testset verbose=true "Inhomogeneous operator lookup                               " begin
        g = line_grid()
        u = test_ftfield(g)
        storage_direction = storage_dim(g, :x)

        for mode in (Forward(), AdjointDiscrete())
            expected = FTField(g)
            operator = NSEBase.derivative_matrix(g, storage_direction, Val(1), mode)
            mul!(parent(expected), operator, parent(u), Val(storage_direction))

            actual = ddx!(FTField(g), u, mode)
            @test parent(actual) ≈ parent(expected)
        end
    end

    @testset verbose=true "Spatial Laplacian composition                               " begin
        g = channel_grid(Nx=5, Ny=9, Nz=5, Nt=3)
        u = test_ftfield(g)
        expected = FTField(g)
        first = FTField(g)
        second = FTField(g)

        # Start with the bounded wall-normal second derivative supplied by
        # the grid, then add the two spatial Fourier second derivatives.
        wall_normal = storage_dim(g, :y)
        D² = NSEBase.derivative_matrix(g, wall_normal, Val(2), Forward())
        mul!(parent(expected), D², parent(u), Val(wall_normal))
        for derivative! in (ddx!, ddz!)
            derivative!(first, u)
            derivative!(second, first)
            expected .+= second
        end

        actual = laplacian!(FTField(g), u)
        @test parent(actual) ≈ parent(expected) rtol=2e-13 atol=2e-13

        # Logical time is deliberately not part of the spatial Laplacian.
        ddt!(first, u)
        ddt!(second, first)
        @test parent(actual) ≉ parent(expected .+ second)
    end

    @testset verbose=true "VectorField component routing                               " begin
        g = line_grid()
        u = VectorField(ntuple(n -> test_ftfield(g; seed=n), 3)...)
        output = VectorField(g, FTField; N=3)

        @test ddy!(output, u) === output
        @test all(parent(output[n]) ≈ parent(ddy!(FTField(g), u[n])) for n in 1:3)
        @test laplacian!(output, u) === output
        @test all(parent(output[n]) ≈ parent(laplacian!(FTField(g), u[n])) for n in 1:3)
    end
end
