@testset verbose=true "Flow constructors                                           " begin
    g = ChannelGrid(7, 17, 7; Nt=3, α=1.25, β=0.75, lim=(2, 6), width=5)
    y = g.xs[1]
    η = @. 2 * (y - 2) / 4 - 1
    couette_base = NSEBase.plane_couette_base(g)
    poiseuille_base = NSEBase.plane_poiseuille_base(g)
    @test couette_base ≈ η
    @test couette_base !== y
    @test poiseuille_base ≈ 1 .- η .^ 2
    @test poiseuille_base[[1, end]] ≈ [0, 0] atol=1e-14

    couette = PlaneCouetteFlow(g, 400; Ro=0.25, fftw_flags=FFTW.ESTIMATE, dealias=false)
    @test couette.base == (η, nothing, nothing)
    @test couette.base[1] !== y
    @test couette.nl.force isa CoriolisForce
    @test couette.ln.force isa CoriolisForce
    @test couette.nl.force.Ro == 0.25
    @test couette.ln.force.Ro == 0.25
    @test couette.nl.force.components == (1, 2)
    @test couette.nl.plans isa FFTPlans{false}
    @test size(couette.nl.pcache[1][1]) == size(g)

    poiseuille = PlanePoiseuilleFlow(g, 400; Ro=0.25, f=1.5, fftw_flags=FFTW.ESTIMATE, dealias=false)
    @test poiseuille.base == (1 .- η .^ 2, nothing, nothing)
    @test poiseuille.nl.force isa CompoundForcing
    @test poiseuille.ln.force isa CompoundForcing
    pressure, rotation = poiseuille.nl.force.forces
    @test pressure isa ConstantBodyForce
    @test pressure.value == 1.5
    @test pressure.component == 1
    @test rotation isa CoriolisForce
    @test rotation.Ro == 0.25
    @test poiseuille.nl.plans isa FFTPlans{false}
    @test size(poiseuille.nl.pcache[1][1]) == size(g)
end

@testset verbose=true "Cartesian primitive nonlinear equations                     " begin
    u(x, y, z, t) = y + (1 - y^2) * cos(2.9x) * exp(cos(5.8z)) * atan(sin(2π*t))
    v(x, y, z, t) = cos(π*y/2)^2 * sin(2.9x) * exp(sin(5.8z)) * cos(sin(2π*t))
    w(x, y, z, t) = cos(π*y) * (1 - y^2) * exp(sin(2.9x)) * exp(sin(5.8z)) * cos(2π*t)^2
    ux(x, y, z, t) = -2.9 * (1 - y^2) * sin(2.9x) * exp(cos(5.8z)) * atan(sin(2π*t))
    vx(x, y, z, t) = 2.9 * cos(π*y/2)^2 * cos(2.9x) * exp(sin(5.8z)) * cos(sin(2π*t))
    wx(x, y, z, t) = 2.9 * cos(π*y) * (1 - y^2) * cos(2.9x) * exp(sin(2.9x)) * exp(sin(5.8z)) * cos(2π*t)^2
    uxx(x, y, z, t) = -2.9^2 * (1 - y^2) * cos(2.9x) * exp(cos(5.8z)) * atan(sin(2π*t))
    vxx(x, y, z, t) = -2.9^2 * cos(π*y/2)^2 * sin(2.9x) * exp(sin(5.8z)) * cos(sin(2π*t))
    wxx(x, y, z, t) = 2.9^2 * cos(π*y) * (1 - y^2) * (cos(2.9x)^2 - sin(2.9x)) * exp(sin(2.9x)) * exp(sin(5.8z)) * cos(2π*t)^2
    uy(x, y, z, t) = 1 - 2y * cos(2.9x) * exp(cos(5.8z)) * atan(sin(2π*t))
    vy(x, y, z, t) = -(π/2) * sin(π*y) * sin(2.9x) * exp(sin(5.8z)) * cos(sin(2π*t))
    wy(x, y, z, t) = -(π * sin(π*y) * (1 - y^2) + 2y * cos(π*y)) * exp(sin(2.9x)) * exp(sin(5.8z)) * cos(2π*t)^2
    uyy(x, y, z, t) = -2 * exp(cos(5.8z)) * cos(2.9x) * atan(sin(2π*t))
    vyy(x, y, z, t) = -(π^2/2) * cos(π*y) * sin(2.9x) * exp(sin(5.8z)) * cos(sin(2π*t))
    wyy(x, y, z, t) = -(π^2 * cos(π*y) * (1 - y^2) - 4π*y * sin(π*y) + 2cos(π*y)) * exp(sin(2.9x)) * exp(sin(5.8z)) * cos(2π*t)^2
    uz(x, y, z, t) = -5.8 * (1 - y^2) * cos(2.9x) * sin(5.8z) * exp(cos(5.8z)) * atan(sin(2π*t))
    vz(x, y, z, t) = 5.8 * cos(π*y/2)^2 * sin(2.9x) * cos(5.8z) * exp(sin(5.8z)) * cos(sin(2π*t))
    wz(x, y, z, t) = 5.8 * cos(π*y) * (1 - y^2) * exp(sin(2.9x)) * cos(5.8z) * exp(sin(5.8z)) * cos(2π*t)^2
    uzz(x, y, z, t) = 5.8^2 * (1 - y^2) * cos(2.9x) * (sin(5.8z)^2 - cos(5.8z)) * exp(cos(5.8z)) * atan(sin(2π*t))
    vzz(x, y, z, t) = 5.8^2 * cos(π*y/2)^2 * sin(2.9x) * (cos(5.8z)^2 - sin(5.8z)) * exp(sin(5.8z)) * cos(sin(2π*t))
    wzz(x, y, z, t) = 5.8^2 * cos(π*y) * (1 - y^2) * exp(sin(2.9x)) * (cos(5.8z)^2 - sin(5.8z)) * exp(sin(5.8z)) * cos(2π*t)^2

    Re, Ro = 37.0, 0.23
    fu(x, y, z, t) = (uxx(x, y, z, t) + uyy(x, y, z, t) + uzz(x, y, z, t)) / Re - u(x, y, z, t) * ux(x, y, z, t) - v(x, y, z, t) * uy(x, y, z, t) - w(x, y, z, t) * uz(x, y, z, t) + Ro * v(x, y, z, t)
    fv(x, y, z, t) = (vxx(x, y, z, t) + vyy(x, y, z, t) + vzz(x, y, z, t)) / Re - u(x, y, z, t) * vx(x, y, z, t) - v(x, y, z, t) * vy(x, y, z, t) - w(x, y, z, t) * vz(x, y, z, t) - Ro * u(x, y, z, t)
    fw(x, y, z, t) = (wxx(x, y, z, t) + wyy(x, y, z, t) + wzz(x, y, z, t)) / Re - u(x, y, z, t) * wx(x, y, z, t) - v(x, y, z, t) * wy(x, y, z, t) - w(x, y, z, t) * wz(x, y, z, t)

    g = ChannelGrid(33, 41, 33; Nt=33, α=2.9, β=5.8, width=19)
    op = CartesianPrimitive3DNSE(g, Re; force=CoriolisForce(Ro), flags=FFTW.ESTIMATE)
    state = FFT(VectorField(g, u, v, w))
    exact = FFT(VectorField(g, fu, fv, fw))
    result = op(0.0, state, similar(state))
    @test result ≈ exact rtol=3e-8
end

@testset verbose=true "Cartesian primitive linearised equations                    " begin
    ux(x, y, z, t) = y + (1 - y^2) * cos(2.9x) * exp(cos(5.8z)) * atan(sin(2π*t))
    uy(x, y, z, t) = cos(π*y/2)^4 * exp(sin(2.9x)) * sin(5.8z) * cos(sin(2π*t))
    uz(x, y, z, t) = (1 - y^2) * sin(2.9x) * cos(5.8z) * cos(cos(2π*t))
    vx(x, y, z, t) = cos(π*y/2)^2 * sin(2.9x) * exp(sin(5.8z)) * cos(sin(2π*t))
    vy(x, y, z, t) = (1 - y^2) * cos(2.9x) * exp(cos(5.8z)) * atan(cos(2π*t))
    vz(x, y, z, t) = sin(π*y) * exp(cos(2.9x)) * sin(5.8z)^2 * cos(2π*t)
    wx(x, y, z, t) = cos(π*y) * (1 - y^2) * cos(2.9x)^2 * exp(sin(5.8z)) * cos(2π*t)^2
    wy(x, y, z, t) = cos(π*y/2) * sin(2.9x) * cos(5.8z) * sin(2π*t)^2
    wz(x, y, z, t) = cos(π*y/2)^4 * exp(sin(2.9x)) * sin(5.8z) * atan(cos(2π*t))

    g = ChannelGrid(9, 17, 9; Nt=9, α=2.9, β=5.8, width=5)
    u = FFT(VectorField(g, ux, uy, uz))
    v = FFT(VectorField(g, vx, vy, vz))
    w = FFT(VectorField(g, wx, wy, wz))
    Re, Ro = 31.0, 0.17
    nonlinear = CartesianPrimitive3DNSE(g, Re; force=CoriolisForce(Ro), flags=FFTW.ESTIMATE)
    forward = CartesianPrimitive3DLNSE(g, Re; force=CoriolisForce(Ro), flags=FFTW.ESTIMATE, mode=Forward())
    adjoint = CartesianPrimitive3DLNSE(g, Re; force=CoriolisForce(Ro), flags=FFTW.ESTIMATE, mode=AdjointDiscrete())

    perturbed = nonlinear(0.0, u .+ 1e-6 .* v, similar(u)) - nonlinear(0.0, u, similar(u))
    linearised = forward(0.0, u, 1e-6 .* v, similar(u))
    @test norm(perturbed - linearised) < 1e-11
    @test abs(dot(forward(0.0, u, v, similar(u)), w) - dot(v, adjoint(0.0, u, w, similar(u)))) < 1e-12
end
