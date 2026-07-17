@testset verbose=true "Rectangular grids                                           " begin
    channel = ChannelGrid(7, 13, 5; Nt=1, α=1, β=1, width=3)
    @test channel isa RectangularGrid{1}
    @test channel isa AbstractChannel3DGrid
    @test channel isa AbstractChannelGrid
    @test !(channel isa AbstractChannel2D3CGrid)
    @test size(channel) == (13, 7, 5, 1)
    @test fft_physical_dims(channel) == (:x, :z, :t)
    @test inhomogeneous_physical_dims(channel) == (:y,)
    @test size(weights(channel)) == (13,)
    @test wavenumber_scale(channel, storage_dim(channel, :x)) ≈ 1
    @test wavenumber_scale(channel, storage_dim(channel, :y)) == 1
    @test channel.scales == (1.0, 1.0, 2π)
    @test typeof(channel).parameters[3] == size(channel)
    @test fieldnames(typeof(channel))[1] == :xs
    @test :sz ∉ fieldnames(typeof(channel))
    @test map(size, points(channel)) == ((13, 1, 1, 1), (1, 7, 1, 1), (1, 1, 5, 1), (1, 1, 1, 1))
    @test vec(points(channel)[2]) ≈ (0:6) .* (2π / 7)
    @test points(channel, (9, 7, 3)) == points(growto(channel, (9, 7, 3)))
    @test_throws ArgumentError ChannelGrid(8, 13, 5; Nt=1, α=1, β=1, width=3)

    y_dim = storage_dim(channel, :y)
    forward_matrix = @inferred NSEBase.derivative_matrix(channel, y_dim, Val(1), Val(false))
    adjoint_matrix = @inferred NSEBase.derivative_matrix(channel, y_dim, Val(1), Val(true))
    @test forward_matrix === channel.D₁[1] && adjoint_matrix === channel.D₁⁺[1]
    @test typeof(forward_matrix) !== typeof(adjoint_matrix)

    grown = growto(channel, (9, 7, 3))
    @test size(grown) == (13, 9, 7, 3)
    @test grown.xs == channel.xs
    @test convert(Float32, channel) isa RectangularGrid{1, Float32}

    channel32 = ChannelGrid(7, 13, 5; Nt=1, α=1, β=1, width=3, T=Float32)
    @test all(coordinate -> eltype(coordinate) === Float32, points(channel32))

    from_data = ChannelGrid(channel.xs[1], 7, 5, 1, 1, 1, channel.D₁[1], channel.D₂[1],
                            channel.D₁⁺[1], channel.D₂⁺[1], channel.ws[1])
    @test size(from_data) == size(channel)
    @test points(from_data) == points(channel)
    @test from_data.D₁[1] === channel.D₁[1]
    @test from_data.D₁⁺[1] === channel.D₁⁺[1]

    lowlevel(grid_size, axes, fft_dims, scales) = RectangularGrid(channel.xs, channel.D₁, channel.D₂,
        channel.D₁⁺, channel.D₂⁺, channel.ws, scales, grid_size, axes, fft_dims)
    @test_throws ArgumentError lowlevel((13, 7, 5, 1), (2, 2, 3, 4), (2, 3, 4), (1, 1, 2π))
    @test_throws ArgumentError lowlevel((13, 7, 5, 1), (2, 1, 3, nothing), (2, 3, 4), (1, 1, 2π))
    @test_throws ArgumentError lowlevel((13, 7, 5, 1), CHANNEL_3D_AXES, (), ())
    @test_throws ArgumentError lowlevel((13, 7, 5, 1), CHANNEL_3D_AXES, (2, 2, 4), (1, 1, 2π))
    @test_throws ArgumentError lowlevel((13, 7, 5, 1), CHANNEL_3D_AXES, (2, 3, 5), (1, 1, 2π))
    @test_throws ArgumentError lowlevel((13, 7, 5, 1), CHANNEL_3D_AXES, (2, 3), (1, 1))
    @test_throws ArgumentError lowlevel((13, 7, 5, 0), CHANNEL_3D_AXES, (2, 3, 4), (1, 1, 2π))

    converted = convert(Float32, from_data)
    @test converted.D₁⁺[1] ≈ Float32.(from_data.D₁⁺[1])
    @test converted.D₂⁺[1] ≈ Float32.(from_data.D₂⁺[1])

    u, du = FTField(channel), FTField(channel)
    parent(u) .= reshape(channel.xs[1] .^ 2, :, 1, 1, 1)
    ddy!(du, u)
    @test parent(du)[:, 1, 1, 1] ≈ 2 .* channel.xs[1] atol=1e-10

    duct = rectangular_test_grid((11, 9, 7, 1), SQUARE_DUCT_AXES, SQUARE_DUCT_FFT_ORDER;
                                 scales=(4, 2π), limits=((0, 2), (-1, 1)), width=3)
    @test duct isa RectangularGrid{2}
    @test duct isa SquareDuctGrid
    @test duct isa AbstractSquareDuctGrid
    @test duct.xs[1] !== duct.xs[2]
    @test size(duct) == (11, 9, 7, 1)
    @test spatial_inhomogeneous_physical_dims(duct) == (:x, :y)
    @test size(weights(duct)) == (11, 9)
    @test wavenumber_scale(duct, storage_dim(duct, :z)) ≈ 4
    @test duct.scales == (4.0, 2π)
    @test typeof(duct).parameters[3] == size(duct)
    @test fieldnames(typeof(duct))[1] == :xs
    @test :sz ∉ fieldnames(typeof(duct))
    @test map(size, points(duct)) == ((11, 1, 1, 1), (1, 9, 1, 1), (1, 1, 7, 1), (1, 1, 1, 1))
    @test points(duct, (9, 3)) == points(growto(duct, (9, 3)))
    @test size(growto(duct, (9, 3))) == (11, 9, 9, 3)
    @test_throws ArgumentError rectangular_test_grid((11, 9, 8, 1), SQUARE_DUCT_AXES, SQUARE_DUCT_FFT_ORDER;
                                                      scales=(4, 2π), limits=((0, 2), (-1, 1)), width=3)

    converted_duct = convert(Float32, duct)
    @test converted_duct isa RectangularGrid{2, Float32}
    @test converted_duct.D₁⁺[1] ≈ Float32.(duct.D₁⁺[1])
    @test converted_duct.D₂⁺[2] ≈ Float32.(duct.D₂⁺[2])

    field, dx, dy, lap = FTField(duct), FTField(duct), FTField(duct), FTField(duct)
    x, y = duct.xs
    parent(field) .= reshape(x .^ 2, :, 1, 1, 1) .+ reshape(y .^ 2, 1, :, 1, 1)
    ddx!(dx, field)
    ddy!(dy, field)
    inhomogeneous_laplacian!(lap, field)
    @test parent(dx)[:, 1, 1, 1] ≈ 2 .* x atol=1e-10
    @test parent(dy)[1, :, 1, 1] ≈ 2 .* y atol=1e-10
    @test parent(lap)[:, :, 1, 1] ≈ fill(4.0, length(x), length(y)) atol=1e-9

    mixed_width_duct = LidDrivenCavityGrid(11, 11; xwidth=3, ywidth=5)
    @test typeof(mixed_width_duct.D₁[1]) != typeof(mixed_width_duct.D₁[2])

    square_duct = SquareDuctGrid(11, 7, 1, 4; width=3)
    @test size(square_duct) == (11, 11, 7, 1)
    @test square_duct.xs[1] === square_duct.xs[2]
    @test square_duct.D₁[1] === square_duct.D₁[2]

    square_duct32 = convert(Float32, square_duct)
    for field in (:xs, :D₁, :D₂, :D₁⁺, :D₂⁺, :ws)
        values = getfield(square_duct32, field)
        @test values[1] == values[2]
    end

    cavity = LidDrivenCavityGrid(11, 9; xlim=(-2, 2), ylim=(0, 1), width=3)
    @test cavity isa RectangularGrid{2}
    @test cavity isa AbstractLidDrivenCavityGrid
    @test LID_DRIVEN_CAVITY_2D_AXES == (1, 2, nothing, 3)
    @test size(cavity) == (11, 9, 1)
    @test fft_physical_dims(cavity) == (:t,)
    @test extrema(cavity.xs[1]) == (-2.0, 2.0)
    @test extrema(cavity.xs[2]) == (0.0, 1.0)
    @test size(weights(cavity)) == (11, 9)
    @test points(cavity, (3,)) == points(growto(cavity, (3,)))

    grid₃ = LidDrivenCavityGrid(7, 7, 7; spanwise=:bounded, dist=FDGrids.GaussLobattoGrid(), width=3)
    x₃ = grid₃.xs[1]
    u₃, lap₃ = FTField(grid₃), FTField(grid₃)
    parent(u₃) .= reshape(x₃ .^ 2, :, 1, 1, 1) .+ reshape(x₃ .^ 2, 1, :, 1, 1) .+ reshape(x₃ .^ 2, 1, 1, :, 1)
    inhomogeneous_laplacian!(lap₃, u₃)
    @test parent(lap₃) ≈ fill(6.0, 7, 7, 7, 1) atol=1e-10

    @test ConstantBodyForce(2; component=3).component == 3
    @test CoriolisForce(0.2).Ro == 0.2
    @test applicable(PlaneCouetteFlow, channel, 100)
    @test applicable(PlanePoiseuilleFlow, channel, 100)
    @test square_duct isa AbstractSquareDuctGrid
    @test applicable(SquareDuctFlow, duct, 100)
    @test applicable(LidDrivenCavityFlow, cavity, 100)
    @test !applicable(PlaneCouetteFlow, cavity, 100)
    @test_throws UndefKeywordError LidDrivenCavityFlow(cavity, 100)
    @test !applicable(SquareDuctFlow, channel, 100)
    @test !applicable(LidDrivenCavityFlow, channel, 100)
    @test !applicable(PlanePoiseuilleFlow, cavity, 100)
    @test !applicable(RayleighBenardFlow, cavity, 1, 1, 1)
    @test_throws ArgumentError LidDrivenCavityGrid(9, 9, 9; spanwise=:unsupported, width=3)
end
