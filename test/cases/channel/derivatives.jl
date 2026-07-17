@testset verbose=true "Derivatives                                                 " begin
    Ny, Nx, Nz, Nt = 17, 7, 7, 7
    α, β = 1.25, 0.75
    g = ChannelGrid(Nx, Ny, Nz; Nt, α, β, width=5)

    u(x, y, z, t) = (1 - y^2) * cos(2α*x) * sin(β*z) * cos(2π*t)
    dudx(x, y, z, t) = -2α * (1 - y^2) * sin(2α*x) * sin(β*z) * cos(2π*t)
    dudy(x, y, z, t) = -2y * cos(2α*x) * sin(β*z) * cos(2π*t)
    dudz(x, y, z, t) = β * (1 - y^2) * cos(2α*x) * cos(β*z) * cos(2π*t)
    dudt(x, y, z, t) = -2π * (1 - y^2) * cos(2α*x) * sin(β*z) * sin(2π*t)
    lapu(x, y, z, t) = (-2 - ((2α)^2 + β^2) * (1 - y^2)) * cos(2α*x) * sin(β*z) * cos(2π*t)

    û = FFT(Field(g, u))
    @test ddx!(FTField(g), û) ≈ FFT(Field(g, dudx)) atol=2e-11 rtol=2e-11
    @test ddy!(FTField(g), û) ≈ FFT(Field(g, dudy)) atol=2e-11 rtol=2e-11
    @test ddz!(FTField(g), û) ≈ FFT(Field(g, dudz)) atol=2e-11 rtol=2e-11
    @test ddt!(FTField(g), û) ≈ FFT(Field(g, dudt)) atol=2e-11 rtol=2e-11
    @test laplacian!(FTField(g), û) ≈ FFT(Field(g, lapu)) atol=2e-10 rtol=2e-10

    zero_fun(x, y, z, t) = zero(y + x + z + t)
    modes = channel_wall_normal_modes(g; Nm=3, components=3)
    a = project(FFT(VectorField(g, u, zero_fun, zero_fun)), modes)
    expected = project(FFT(VectorField(g, dudt, zero_fun, zero_fun)), modes)
    @test ddt!(similar(a), a) ≈ expected atol=2e-11 rtol=2e-11
end

@testset verbose=true "Weighted derivative adjoints                                " begin
    g = ChannelGrid(7, 17, 7; Nt=7, α=1.25, β=0.75, width=5)
    a(x, y, z, t) = (1 + y + 0.3y^2) * (1 + 0.2cos(1.25x) + 0.1sin(0.75z) + 0.05cos(2π*t))
    b(x, y, z, t) = (0.4 - 0.2y + y^2) * (1 - 0.1sin(1.25x) + 0.3cos(0.75z) - 0.07sin(2π*t))
    â, b̂ = FFT(Field(g, a)), FFT(Field(g, b))

    Da, D⁺b = FTField(g), FTField(g)
    ddy!(Da, â)
    ddy!(D⁺b, b̂; adjoint=true)
    @test dot(Da, b̂) ≈ dot(â, D⁺b) atol=2e-12 rtol=2e-12

    inhomogeneous_laplacian!(Da, â)
    inhomogeneous_laplacian!(D⁺b, b̂; adjoint=true)
    @test dot(Da, b̂) ≈ dot(â, D⁺b) atol=2e-12 rtol=2e-12
end
