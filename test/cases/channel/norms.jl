@testset verbose=true "Analytical norms and phase alignment                        " begin
    Ny, Nx, Nz, Nt = 25, 9, 9, 9
    g = ChannelGrid(Nx, Ny, Nz; Nt, α=1.0, β=1.0, width=7)
    periodic(x, z, t) = case_periodic_profile(x) * case_periodic_profile(z) * case_periodic_profile(2π*t)
    f₁(x, y, z, t) = (1 - y^2) * periodic(x, z, t)
    f₂(x, y, z, t) = (1 + y) * (1 - y^2) * periodic(x, z, t)

    û, v̂ = FFT(Field(g, f₁)), FFT(Field(g, f₂))
    q̂, r̂ = FFT(VectorField(g, f₁)), FFT(VectorField(g, f₂))
    periodic_norm2 = CASE_PERIODIC_PROFILE_NORM2^3
    u_norm2, v_norm2 = CASE_WALL_POLY_NORM2 * periodic_norm2, (128 // 105) * periodic_norm2
    uv_dot, difference_norm2 = u_norm2, (16 // 105) * periodic_norm2
    @test dot(û, v̂) ≈ uv_dot rtol=1e-12
    @test norm(û)^2 ≈ u_norm2 rtol=1e-12
    @test norm(FFT(VectorField(g, f₁, f₂, f₁)))^2 ≈ 2u_norm2 + v_norm2 rtol=1e-12

    modes = channel_wall_normal_modes(g)
    a, b = project(q̂, modes), project(r̂, modes)
    @test norm(a) ≈ norm(q̂)
    @test normdiff(q̂, r̂)^2 ≈ difference_norm2 rtol=1e-12
    @test normdiff(a, b)^2 ≈ difference_norm2 rtol=1e-12

    # A band-limited field makes the alignment unique and exact to roundoff; channel time has unit period.
    phase_field(x, y, z, t) = (1 - y^2) * (1 + 0.2cos(x) + 0.1sin(z)) * (1 + 0.3cos(2π*t) + 0.2sin(4π*t))
    shifted(x, y, z, t) = phase_field(x, y, z - π, t - 1/4)
    min_diff, minimizing_shift = minnormdiff(FFT(Field(g, phase_field)), FFT(Field(g, shifted)), (4, 4, 4))
    @test min_diff ≈ 0 atol=1e-14
    @test all(minimizing_shift .≈ (0, π, 0.25))
end
