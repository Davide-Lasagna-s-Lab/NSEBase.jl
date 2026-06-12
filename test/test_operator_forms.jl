# Advection-form variants: correctness on a real collocation channel grid.
#
# The divergence and rotational forms are mathematically equivalent to the
# advective form up to a gradient (which the projection removes). On the
# ChannelMockGrid the wall-normal derivative is a polynomial-exact collocation
# operator, so the discrete product rule holds exactly for the resolved fields
# and the forms agree to machine precision:
#
#   - divergence ≡ advective                       (full operator)
#   - rotational ≡ advective  modulo a gradient     (so curl of the difference → 0)
#
# We also check that each forward form is the correct linearisation of the
# nonlinear operator *of the same form* (finite-difference vs. linearised).

# ---- helpers ------------------------------------------------------- #
_relnorm(a, b) = norm(a - b) / norm(a)

# divergence-free field u = ∇×A from three scalar potentials (commuting spectral
# and collocation derivatives make ∇·u = 0 hold discretely).
function _divfree(g, a, b, c)
    A = FFT(VectorField(g, a, b, c)); u = VectorField(g); tmp = FTField(g)
    ddy!(u[1], A[3]); ddz!(tmp, A[2]); u[1] .-= tmp
    ddz!(u[2], A[1]); ddx!(tmp, A[3]); u[2] .-= tmp
    ddx!(u[3], A[2]); ddy!(tmp, A[1]); u[3] .-= tmp
    return u
end

function _vorticity(g, f)
    W = VectorField(g); t = FTField(g)
    ddy!(W[1], f[3]); ddz!(t, f[2]); W[1] .-= t
    ddz!(W[2], f[1]); ddx!(t, f[3]); W[2] .-= t
    ddx!(W[3], f[2]); ddy!(t, f[1]); W[3] .-= t
    return W
end
_curl_relnorm(g, a, b) = norm(_vorticity(g, a .- b)) / norm(_vorticity(g, a))

# finite-difference linearisation of the nonlinear operator `nl` about `U` in
# direction `v`: [N(U+εv) − N(U)], returned as a VectorField (≈ ε·L(v)).
function _fd_linearise(nl, U, v, ε)
    a  = nl(0.0, U .+ ε .* v, similar(U))
    a0 = nl(0.0, U,           similar(U))
    return a .- a0
end


@testset "Advection-form variants — 3D                                        " begin
    g = ChannelMockGrid(14, 13, 13)
    # Broadband exp(cos/sin) content in the Fourier directions (x, z) excites all
    # resolved wavenumbers; the wall-normal (y) dependence stays a low-degree
    # polynomial so the collocation product rule is exact and the forms agree to
    # machine precision. The FFT band-limits the fields, so dealiased products
    # are exact.
    U = _divfree(g, (y,x,z)->(1-y^2) *exp(cos(x))  *sin(z),
                    (y,x,z)->(1-y^2) *exp(sin(2x)) *cos(z),
                    (y,x,z)->(1-y^2)^2*cos(x)      *exp(sin(2z)))
    v = _divfree(g, (y,x,z)->(1-y^2) *sin(x)       *exp(cos(2z)),
                    (y,x,z)->(1-y^2)^2*exp(cos(2x))*sin(z),
                    (y,x,z)->(1-y^2) *exp(cos(x))  *exp(sin(z)))
    Re = 137.0

    nse(form)  = Cartesian3DNSE(g, Re; form=form, flags=FFTW.ESTIMATE)
    lnse(m, f) = Cartesian3DNSE(g, Re; mode=m, form=f, flags=FFTW.ESTIMATE)

    @testset "nonlinear NSE" begin
        oa = nse(Advective())( 0.0, U, similar(U))
        od = nse(Divergence())(0.0, U, similar(U))
        or = nse(Rotational())(0.0, U, similar(U))
        @test _relnorm(oa, od)        < 1e-10              # divergence ≡ advective
        @test _curl_relnorm(g, oa, or) < 1e-10             # rotational ≡ advective mod gradient
        # each form is the correct linearisation of itself
        for f in (Advective(), Divergence(), Rotational())
            ln  = lnse(Forward(), f)
            fd  = _fd_linearise(nse(f), U, v, 1e-6)
            lin = ln(0.0, U, v, similar(U)); lin .*= 1e-6
            @test _relnorm(fd, lin) < 1e-4
        end
    end

    @testset "forward LNSE" begin
        ra = lnse(Forward(), Advective())( 0.0, U, v, similar(U))
        rd = lnse(Forward(), Divergence())(0.0, U, v, similar(U))
        rr = lnse(Forward(), Rotational())(0.0, U, v, similar(U))
        @test _relnorm(ra, rd)        < 1e-10
        @test _curl_relnorm(g, ra, rr) < 1e-10
    end

    @testset "continuous adjoint LNSE" begin
        ca = lnse(AdjointContinuous(), Advective())( 0.0, U, v, similar(U))
        cd = lnse(AdjointContinuous(), Divergence())(0.0, U, v, similar(U))
        cr = lnse(AdjointContinuous(), Rotational())(0.0, U, v, similar(U))
        @test _relnorm(ca, cd) < 1e-10
        @test _relnorm(ca, cr) < 1e-10              # rotational routes to divergence
    end

    @testset "discrete adjoint identity (advective)" begin
        # ⟨L v, w⟩ == ⟨v, L* w⟩ to machine precision (unit weights ⇒ D* = Dᵀ).
        # Generic (non-solenoidal) v, w give a non-degenerate pairing; the
        # identity is an operator transpose and holds for any v, w.
        vg = FFT(VectorField(g, (y,x,z)->(1-y^2)*exp(sin(x)) *cos(z),
                                (y,x,z)->(1-y^2)*exp(cos(2x))*sin(z),
                                (y,x,z)->(1-y^2)*sin(x)      *exp(sin(z))))
        wg = FFT(VectorField(g, (y,x,z)->(1-y^2)*cos(x)      *exp(sin(2z)),
                                (y,x,z)->(1-y^2)*exp(sin(2x))*cos(z),
                                (y,x,z)->(1-y^2)*exp(cos(x)) *cos(z)))
        Lf = lnse(Forward(), Advective())
        La = lnse(AdjointDiscrete(), Advective())
        lhs = dot(Lf(0.0, U, vg, similar(U)), wg)
        rhs = dot(vg, La(0.0, U, wg, similar(U)))
        @test abs(lhs - rhs) / abs(lhs) < 1e-10
        # divergence/rotational discrete adjoints are Phase B
        @test_throws ArgumentError lnse(AdjointDiscrete(), Divergence())(0.0, U, vg, similar(U))
    end
end
