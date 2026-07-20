case_poly11(x) = @. x * (1 - x)
case_poly21(x) = @. x^2 * (1 - x)
case_poly12(x) = @. x * (1 - x)^2
case_poly22(x) = @. x^2 * (1 - x)^2
case_sin2pi(x) = @. sin(2π * x)
case_wall_poly(y) = @. 1 - y^2
case_wall_poly2(y) = @. (1 - y^2)^2
case_wall_sine(y) = @. sin(π * y)

# Exact bounded-domain integrals used by the case-level norm tests.
const CASE_UNIT_BUBBLE_NORM2 = 1 // 30
const CASE_WALL_POLY_NORM2 = 16 // 15

# The case-level Fourier tests use odd nine-point grids, whose distinct wavenumbers are
# `0:4`. This band-limited profile gives every one of those modes a nonzero coefficient,
# so it exercises the complete Fourier operator while retaining exact derivative and norm
# references. In particular, its mean square over one period is `2133 / 2048`.
case_periodic_profile(ξ) = @. 1 + sin(ξ) / 4 + cos(2ξ) / 8 - sin(3ξ) / 16 + cos(4ξ) / 32
case_periodic_profile_d1(ξ) = @. cos(ξ) / 4 - sin(2ξ) / 4 - 3cos(3ξ) / 16 - sin(4ξ) / 8
case_periodic_profile_d2(ξ) = @. -sin(ξ) / 4 - cos(2ξ) / 2 + 9sin(3ξ) / 16 - cos(4ξ) / 2
const CASE_PERIODIC_PROFILE_NORM2 = 2133 // 2048

function case_grown_size(g, target)
    result = collect(size(g))
    for (dim, n) in zip(fft_storage_dims(g), target)
        result[dim] = n
    end
    return Tuple(result)
end

function test_one_inhomogeneous_fd_contract(g, target)
    y, w = only(g.xs), only(g.ws)
    N = length(y)

    @test length(y) == N
    @test length(w) == N
    @test size(only(g.D₁)) == (N, N)
    @test size(only(g.D₂)) == (N, N)
    @test weights(g) === w
    @test length(w) == N
    @test sum(w .* case_wall_poly(y)) ≈ 4 / 3 rtol=1e-10
    @test sum(w .* case_wall_sine(y) .^ 2) ≈ 1 rtol=1e-10

    @test maximum(abs, apply_dd(g, case_wall_poly(y), 1) .+ 2 .* y) < 1e-8
    @test maximum(abs, apply_dd(g, case_wall_sine(y), 1) .- π .* cos.(π .* y)) < 1e-6

    pairs = ((case_wall_poly(y), case_wall_sine(y)),
             (case_wall_sine(y), case_wall_poly2(y)),
             (case_wall_poly2(y), case_wall_poly(y)))
    for (u, v) in pairs
        @test weighted_dot(g, apply_dd(g, u, 1), v) ≈
              weighted_dot(g, u, apply_dd(g, v, 1; adjoint=true)) rtol=1e-12 atol=100eps(eltype(y))
    end
    for (u, v) in pairs
        @test weighted_dot(g, apply_lap(g, u), v) ≈
              weighted_dot(g, u, apply_lap(g, v; adjoint=true)) rtol=1e-12 atol=100eps(eltype(y))
    end

    grown = growto(g, target)
    @test only(grown.D₁) === only(g.D₁)
    @test only(grown.D₂) === only(g.D₂)
    @test only(grown.D₁⁺) === only(g.D₁⁺)
    @test only(grown.D₂⁺) === only(g.D₂⁺)
    @test only(grown.xs) === y
    @test only(grown.ws) === w
    @test size(grown) == case_grown_size(g, target)
end

function test_two_inhomogeneous_fd_contract(g, target)
    x = g.xs[1]
    p11, p22, s2π = case_poly11(x), case_poly22(x), case_sin2pi(x)
    u = p11 * transpose(p11)

    @test maximum(abs, apply_dd(g, u, 1) .- (1 .- 2 .* x) * transpose(p11)) < 1e-8
    @test maximum(abs, apply_dd(g, u, 2) .- p11 * transpose(1 .- 2 .* x)) < 1e-8
    @test maximum(abs, apply_dd(g, s2π * transpose(p22), 1) .- (2π .* cos.(2π .* x)) * transpose(p22)) < 1e-6

    derivative_pairs = ((case_poly21(x) * transpose(case_poly12(x)), case_poly12(x) * transpose(case_poly21(x))),
                        (p11 * transpose(case_poly21(x)), case_poly21(x) * transpose(case_poly12(x))),
                        (p22 * transpose(case_poly12(x)), case_poly21(x) * transpose(p11)))
    for (a, b) in derivative_pairs, dim in (1, 2)
        @test weighted_dot(g, apply_dd(g, a, dim), b) ≈
              weighted_dot(g, a, apply_dd(g, b, dim; adjoint=true)) rtol=1e-12 atol=100eps(eltype(x))
    end

    laplacian_pairs = ((p11 * transpose(s2π), s2π * transpose(p22)),
                       (s2π * transpose(p22), p22 * transpose(p11)),
                       (p22 * transpose(p11), p11 * transpose(s2π)),
                       (p11 * transpose(s2π), p11 * transpose(s2π)))
    for (a, b) in laplacian_pairs
        @test weighted_dot(g, apply_lap(g, a), b) ≈
              weighted_dot(g, a, apply_lap(g, b; adjoint=true)) rtol=1e-12 atol=100eps(eltype(x))
    end

    sine_product = sin.(π .* x) * transpose(sin.(π .* x))
    dx, dy, lap = apply_dd(g, sine_product, 1), apply_dd(g, sine_product, 2), apply_lap(g, sine_product)
    @test weighted_dot(g, sine_product, sine_product) ≈ 1 / 4 rtol=1e-12
    @test weighted_dot(g, dx, dx) ≈ π^2 / 4 rtol=1e-8
    @test weighted_dot(g, dy, dy) ≈ π^2 / 4 rtol=1e-8
    @test weighted_dot(g, lap, lap) ≈ π^4 rtol=1e-6

    grown = growto(g, target)
    @test grown.D₁ === g.D₁
    @test grown.D₂ === g.D₂
    @test grown.D₁⁺ === g.D₁⁺
    @test grown.D₂⁺ === g.D₂⁺
    @test grown.xs === g.xs
    @test grown.ws === g.ws
    @test weights(grown) === weights(g)
    @test size(grown) == case_grown_size(g, target)
end

function case_physical_to_spectral(g, plans, f)
    result = physical_to_ft(g, f; plans)
    return result.spectral, parent(result.physical)
end

function case_reference_dot(u::FTField, v::FTField)
    g = grid(u)
    ws, inhomogeneous_dims = weights(g), inhomogeneous_storage_dims(g)
    rfft_dim = first(fft_storage_dims(g))
    result = zero(real(eltype(parent(u))))
    @inbounds for I in CartesianIndices(parent(u))
        weight_index = ntuple(i -> I[inhomogeneous_dims[i]], length(inhomogeneous_dims))
        result += (I[rfft_dim] == 1 ? 1 : 2) * ws[weight_index...] * real(conj(parent(u)[I]) * parent(v)[I])
    end
    return result
end

function case_physical_dot(g, u, v=u)
    weight_shape = ntuple(dim -> dim in inhomogeneous_storage_dims(g) ? size(g, dim) : 1, ndims(u))
    normalization = prod(size(g, dim) for dim in fft_storage_dims(g))
    return sum(reshape(weights(g), weight_shape) .* conj.(u) .* v) / normalization
end

function case_polynomial_value(poly, x, y)
    result = zero(x + y)
    for (i, j, coefficient) in poly
        result += Float64(coefficient) * x^i * y^j
    end
    return result
end

function case_polynomial_product_integral(poly, test_poly)
    result = 0 // 1
    for (i, j, a) in poly, (k, l, b) in test_poly
        result += (a * b) / ((i + k + 1) * (j + l + 1))
    end
    return result
end

function case_analytic_velocity_field(g, velocity)
    components = map(velocity) do component
        FFT(Field(g, (x, y, _, _) -> case_polynomial_value(component, x, y)))
    end
    return VectorField(components...)
end

case_analytic_velocity_dot(velocity, test_velocity) =
    sum(case_polynomial_product_integral(u, v) for (u, v) in zip(velocity, test_velocity))
