@testset verbose=true "Analytic norms and phase alignment                          " begin
    Ny, Nx, Nz, Nt = 25, 5, 17, 17
    g = ChannelGrid(Nx, Ny, Nz; Nt, α=1.0, β=1.0, width=7)
    f₁(x, y, z, t) = (1 - y^2) * exp(cos(z)) * cos(sin(2π*t))
    f₂(x, y, z, t) = cos(π*y) * (1 - y^2) * exp(sin(z)) * cos(2π*t)^2

    û, v̂ = FFT(Field(g, f₁)), FFT(Field(g, f₂))
    q̂, r̂ = FFT(VectorField(g, f₁)), FFT(VectorField(g, f₂))
    @test dot(û, v̂) ≈ 0.339593 rtol=1e-5
    @test norm(û)^2 ≈ 1.487980 rtol=1e-5
    @test norm(FFT(VectorField(g, f₁, f₂)))^2 ≈ 1.930734 rtol=1e-5

    modes = channel_wall_normal_modes(g)
    a, b = project(q̂, modes), project(r̂, modes)
    @test norm(a) ≈ norm(q̂)
    @test normdiff(q̂, r̂)^2 ≈ 1.251547 rtol=1e-5
    @test normdiff(a, b)^2 ≈ 1.251547 rtol=1e-5

    shifted(x, y, z, t) = f₁(x + π, y, z - π, t - π/2)
    _, minimizing_shift = minnormdiff(û, FFT(Field(g, shifted)), (4, 4, 4))
    @test_broken all(minimizing_shift .≈ (0, π, 0.25))
end
