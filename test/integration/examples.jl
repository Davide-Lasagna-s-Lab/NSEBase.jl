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

    rpcf = include_example("rpcf.jl")
    @test rpcf.grid isa AbstractChannel2D3CGrid
    @test rpcf.equations isa ProjectedNSE

    convection = include_example("rayleigh_benard.jl")
    @test convection.grid isa AbstractChannel3DGrid
    @test convection.equations isa ProjectedNSE
end
