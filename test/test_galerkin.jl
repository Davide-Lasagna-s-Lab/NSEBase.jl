function _fill_vector_field!(u::VectorField)
    for n in eachindex(u)
        data = parent(u[n])
        for k in axes(data, 2), j in axes(data, 1)
            data[j, k] = complex(0.2n + 0.3j - 0.1k, -0.4n + 0.2j + 0.5k)
        end
    end
    return u
end

function _fill_coefficients!(a::ProjectedField)
    data = parent(a)
    for k in axes(data, 2), m in axes(data, 1)
        data[m, k] = complex(0.7m - 0.2k, 0.1m + 0.3k)
    end
    return a
end

# kH-dependent modes for the 2D test grid:
#   axis 1   : Ny  — inhomogeneous (wall-normal)
#   axis 2   : Nm  — mode index
#   axis 3   : Nk  — rfft half-spectrum (homogeneous)
function _mode_arrays(Nc, Ny, Nm, Nk)
    ntuple(Nc) do n
        [complex(0.1n + 0.2j - 0.3m + 0.05k,
                 0.2n - 0.1j + 0.4m - 0.07k)
         for m in 1:Nm, j in 1:Ny, k in 1:Nk]
    end
end

function _expected_project(u::VectorField, modes, ws)
    Nc = length(modes)
    Nm, Ny, Nk = size(modes[1])
    out = zeros(ComplexF64, Nm, Nk)
    for n in 1:Nc, k in 1:Nk, m in 1:Nm, j in 1:Ny
        out[m, k] += ws[j] * conj(modes[n][m, j, k]) * parent(u[n])[j, k]
    end
    return out
end

function _expected_expand(a::ProjectedField, modes)
    Nc = length(modes)
    Nm, Ny, Nk = size(modes[1])
    out = ntuple(_ -> zeros(ComplexF64, Ny, Nk), Nc)
    for n in 1:Nc, k in 1:Nk, j in 1:Ny, m in 1:Nm
        out[n][j, k] += modes[n][m, j, k] * parent(a)[m, k]
    end
    return out
end

@testset verbose=true "Galerkin projection and expansion                                   " begin
    @testset "explicit formulas" begin
        Ny, Nx, Nm, Nc = 3, 5, 2, 2
        Nk = (Nx >> 1) + 1
        ws = [0.25, 0.5, 1.25]
        g = GalerkinGrid{(Ny, Nx)}(ws)
        modes = _mode_arrays(Nc, Ny, Nm, Nk)

        u = _fill_vector_field!(VectorField(g; N=Nc))

        expected_a = _expected_project(u, modes, ws)
        for alg in (LoopGalerkin(), GemmGalerkin())
            a = ProjectedField(g, modes)
            project!(a, u, alg)
            @test parent(a) ≈ expected_a
        end

        a = _fill_coefficients!(ProjectedField(g, modes))
        expected_u = _expected_expand(a, modes)
        for alg in (LoopGalerkin(), GemmGalerkin())
            uout = VectorField(g; N=Nc)
            expand!(uout, a, alg)
            for n in 1:Nc
                @test parent(uout[n]) ≈ expected_u[n]
            end
        end
    end

    @testset "LoopGalerkin and GemmGalerkin agree" begin
        Ny, Nx, Nm, Nc = 4, 7, 3, 2
        Nk = (Nx >> 1) + 1
        ws = [1.0, 0.5, 1.5, 2.0]
        g  = GalerkinGrid{(Ny, Nx)}(ws)

        modes = _mode_arrays(Nc, Ny, Nm, Nk)
        u     = _fill_vector_field!(VectorField(g; N=Nc))

        a_loop = ProjectedField(g, modes)
        a_gemm = ProjectedField(g, modes)
        project!(a_loop, u, LoopGalerkin())
        project!(a_gemm, u, GemmGalerkin())
        @test parent(a_gemm) ≈ parent(a_loop)

        _fill_coefficients!(a_loop)
        parent(a_gemm) .= parent(a_loop)

        u_loop = VectorField(g; N=Nc)
        u_gemm = VectorField(g; N=Nc)
        expand!(u_loop, a_loop, LoopGalerkin())
        expand!(u_gemm, a_gemm, GemmGalerkin())
        for n in 1:Nc
            @test parent(u_gemm[n]) ≈ parent(u_loop[n])
        end
    end

    @testset "project-expand round trip" begin
        Ny, Nx, Nc = 4, 5, 3
        Nk = (Nx >> 1) + 1
        g  = GalerkinGrid{(Ny, Nx)}(ones(Ny))

        # An orthonormal basis: the n-th block of `Nm = Nc*Ny` columns picks
        # out the j-th wall-normal point of component n, repeated identically
        # at every wavenumber.
        Nm = Nc * Ny
        eye_modes = ntuple(Nc) do n
            data = zeros(ComplexF64, Nm, Ny, Nk)
            for k in 1:Nk, j in 1:Ny
                data[(n - 1) * Ny + j, j, k] = 1
            end
            data
        end

        coeffs = [complex(0.3m - 0.2k, 0.4m + 0.1k) for m in 1:Nm, k in 1:Nk]

        for alg in (LoopGalerkin(), GemmGalerkin())
            a = ProjectedField(g, eye_modes)
            parent(a) .= coeffs

            u = expand(a, alg)
            b = project(u, eye_modes, alg)

            @test parent(b) ≈ coeffs
        end
    end
end
