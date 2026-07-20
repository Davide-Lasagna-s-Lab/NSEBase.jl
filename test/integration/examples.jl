include_example(file) = redirect_stdout(devnull) do
    include(joinpath(@__DIR__, "..", "..", "examples", file))
end

@testset verbose=true "Case examples                                               " begin
    channel = include_example("channel.jl")
    @test channel.grid isa AbstractChannel3DGrid
    @test channel.couette isa ProjectedNSE
    @test channel.poiseuille isa ProjectedNSE

    duct = include_example("duct.jl")
    @test duct.grid isa SquareDuctGrid
    @test duct.equations isa ProjectedNSE
    @test duct.square_grid isa SquareDuctGrid
    @test duct.square_equations isa ProjectedNSE

    cavity = include_example("lid_driven_cavity.jl")
    @test cavity.grid isa AbstractLidDrivenCavityGrid
    @test cavity.equations isa ProjectedNSE

    streamwise_invariant = include_example("streamwise_invariant_channel.jl")
    @test streamwise_invariant.grid isa AbstractStreamwiseInvariantChannelGrid
    @test streamwise_invariant.equations isa ProjectedNSE

    two_dimensional = include_example("two_dimensional_channel.jl")
    @test two_dimensional.grid isa AbstractTwoDimensionalChannelGrid
    @test two_dimensional.equations isa ProjectedNSE

    convection = include_example("rayleigh_benard.jl")
    @test convection.two_dimensional.grid isa AbstractTwoDimensionalChannelGrid
    @test convection.two_dimensional.equations isa ProjectedNSE
    @test convection.two_dimensional.state isa VectorField{3}
    @test convection.three_dimensional.grid isa AbstractChannel3DGrid
    @test convection.three_dimensional.equations isa ProjectedNSE
    @test convection.three_dimensional.state isa VectorField{4}
end
