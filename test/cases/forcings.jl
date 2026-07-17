@testset verbose=true "Case forcings                                               " begin
    g = ChannelGrid(3, 9, 3; Nt=1, α=1, β=1, width=3)
    u, out = VectorField(g; N=3), VectorField(g; N=3)
    parent(u[1]) .= 1
    parent(u[2]) .= 2
    parent(u[3]) .= 3

    force = CoriolisForce(0.25)
    force(out, u, Forward())
    @test parent(out[1]) == fill(0.5, size(out[1]))
    @test parent(out[2]) == fill(-0.25, size(out[2]))
    @test iszero(parent(out[3]))

    out .= 0
    force(out, u, AdjointDiscrete())
    @test parent(out[1]) == fill(-0.5, size(out[1]))
    @test parent(out[2]) == fill(0.25, size(out[2]))

    out .= 0
    rpcf_force = CoriolisForce(0.25; components=(3, 1))
    rpcf_force(out, u, Forward())
    @test parent(out[1]) == fill(-0.75, size(out[1]))
    @test parent(out[3]) == fill(0.25, size(out[3]))
    @test_throws ArgumentError CoriolisForce(1; components=(1, 1))
    @test_throws ArgumentError CoriolisForce(1; components=(1, 4))

    out .= 0
    pressure = ConstantBodyForce(2.5; component=1)
    pressure(out, u, Forward())
    pressure(out, u, Forward())
    @test parent(out[1])[:, 1, 1, 1] == fill(5.0, 9)
    parent(out[1])[:, 1, 1, 1] .= 0
    @test iszero(parent(out[1]))
    @test iszero(parent(out[2]))
    @test iszero(parent(out[3]))

    @test_throws ArgumentError ConstantBodyForce(1; component=4)(out, u, Forward())
end
