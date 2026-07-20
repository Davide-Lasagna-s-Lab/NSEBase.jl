@testset verbose=true "Derivatives                                                 " begin
    Ny, Nx, Nz, Nt = 17, 9, 9, 9
    α, β = 1.25, 0.75
    g = ChannelGrid(Nx, Ny, Nz; Nt, α, β, width=5)

    fx(x) = case_periodic_profile(α*x)
    dfx(x) = α * case_periodic_profile_d1(α*x)
    d2fx(x) = α^2 * case_periodic_profile_d2(α*x)
    fz(z) = case_periodic_profile(β*z)
    dfz(z) = β * case_periodic_profile_d1(β*z)
    d2fz(z) = β^2 * case_periodic_profile_d2(β*z)
    ft(t) = case_periodic_profile(2π*t)
    dft(t) = 2π * case_periodic_profile_d1(2π*t)
    u(x, y, z, t) = (1 - y^2) * fx(x) * fz(z) * ft(t)
    dudx(x, y, z, t) = (1 - y^2) * dfx(x) * fz(z) * ft(t)
    dudy(x, y, z, t) = -2y * fx(x) * fz(z) * ft(t)
    dudz(x, y, z, t) = (1 - y^2) * fx(x) * dfz(z) * ft(t)
    dudt(x, y, z, t) = (1 - y^2) * fx(x) * fz(z) * dft(t)
    lapu(x, y, z, t) = (-2fx(x) * fz(z) + (1 - y^2) * (d2fx(x) * fz(z) + fx(x) * d2fz(z))) * ft(t)

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
    a(x, y, z, t) = (1 + y + 0.3y^2) * exp(sin(1.25x)) * exp(cos(0.75z)) * exp(sin(2π*t))
    b(x, y, z, t) = (0.4 - 0.2y + y^2) * exp(cos(1.25x)) * exp(sin(0.75z)) * exp(cos(2π*t))
    â, b̂ = FFT(Field(g, a)), FFT(Field(g, b))

    Da, D⁺b = FTField(g), FTField(g)
    ddy!(Da, â)
    ddy!(D⁺b, b̂; adjoint=true)
    @test dot(Da, b̂) ≈ dot(â, D⁺b) atol=2e-12 rtol=2e-12

    inhomogeneous_laplacian!(Da, â)
    inhomogeneous_laplacian!(D⁺b, b̂; adjoint=true)
    @test dot(Da, b̂) ≈ dot(â, D⁺b) atol=2e-12 rtol=2e-12
end
