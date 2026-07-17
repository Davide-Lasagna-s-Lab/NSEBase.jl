@testset verbose=true "Three-direction vector-field shifts                         " begin
    g = ChannelGrid(9, 17, 9; Nt=9, α=1.0, β=1.0, width=5)
    sx, sz, st = 0.37, 1.11, 0.23
    u = ((x, y, z, t) -> y + (1 - y^2) * cos(x) * cos(z) * cos(2π*t),
         (x, y, z, t) -> -(π/2) * cos(π*y/2)^2 * cos(x) * sin(z) * sin(2π*t),
         (x, y, z, t) -> (π/2) * sin(π*y) * cos(x) * cos(z) * sin(2π*t))
    shifted = ((x, y, z, t) -> y + (1 - y^2) * cos(x + sx) * cos(z + sz) * cos(2π*(t + st)),
               (x, y, z, t) -> -(π/2) * cos(π*y/2)^2 * cos(x + sx) * sin(z + sz) * sin(2π*(t + st)),
               (x, y, z, t) -> (π/2) * sin(π*y) * cos(x + sx) * cos(z + sz) * sin(2π*(t + st)))

    û, shifted_û = FFT(VectorField(g, u...)), FFT(VectorField(g, shifted...))
    @test shift!(û, (0, 0, 0)) === û
    @test shift!(copy(û), (sx, sz, st)) ≈ shifted_û atol=1e-12
end
