@testset verbose=true "Three-direction norm weighting                              " begin
    Ny, Nx, Nz, Nt = 17, 9, 9, 9
    g = ChannelGrid(Nx, Ny, Nz; Nt, α=2π, β=2π, width=5)
    f₁(x, y, z, t) = (1 - y^2) * exp(sin(2π*x)) * exp(cos(2π*z)) * exp(sin(2π*t))
    f₂(x, y, z, t) = cos(π*y) * (1 - y^2) * exp(cos(2π*x)) * exp(sin(2π*z)) * exp(cos(2π*t))
    modes = channel_wall_normal_modes(g; Nm=3)
    a = project(FFT(VectorField(g, f₁)), modes)
    b = project(FFT(VectorField(g, f₂)), modes)

    A = FarazmandWeight(2π, 2π, 2π)
    expected = copy(a)
    for nt in 1:Nt, nz in 1:Nz, nx in 1:(Nx >> 1) + 1, m in axes(expected, 1)
        kx = nx - 1
        kz = nz <= (Nz >> 1) + 1 ? nz - 1 : nz - Nz - 1
        kt = nt <= (Nt >> 1) + 1 ? nt - 1 : nt - Nt - 1
        expected[m, nx, nz, nt] /= 1 + 4π^2 * (kx^2 + kz^2 + kt^2)
    end
    @test lmul!(A, copy(a)) ≈ expected
    @test dot(a, A, b) == dot(a, lmul!(A, copy(b)))
end
