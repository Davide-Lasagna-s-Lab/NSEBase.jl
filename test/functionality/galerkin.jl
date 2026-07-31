function _fill_vector_field!(u::VectorField)
    for component in eachindex(u), k in axes(u[component], 2), j in axes(u[component], 1)
        u[component][j, k] = complex(0.2component + 0.3j - 0.1k,
                                     -0.4component + 0.2j + 0.5k)
    end
    return u
end

function _fill_coefficients!(a::ProjectedField)
    for k in axes(a, 2), mode in axes(a, 1)
        a[mode, k] = complex(0.7mode - 0.2k, 0.1mode + 0.3k)
    end
    return a
end

function _mode_arrays(ncomponents, nbounded, nmodes, nwavenumbers)
    return ntuple(ncomponents) do component
        [complex(0.1component + 0.2j - 0.3mode + 0.05k,
                 0.2component - 0.1j + 0.4mode - 0.07k)
         for mode in 1:nmodes, j in 1:nbounded, k in 1:nwavenumbers]
    end
end

function _expected_project(u::VectorField, modes, quadrature_weights)
    ncomponents = length(modes)
    nmodes, nbounded, nwavenumbers = size(modes[1])
    coefficients = zeros(ComplexF64, nmodes, nwavenumbers)
    for component in 1:ncomponents, k in 1:nwavenumbers, mode in 1:nmodes, j in 1:nbounded
        coefficients[mode, k] += quadrature_weights[j] *
                                 conj(modes[component][mode, j, k]) * u[component][j, k]
    end
    return coefficients
end

function _expected_expand(a::ProjectedField, modes)
    ncomponents = length(modes)
    nmodes, nbounded, nwavenumbers = size(modes[1])
    fields = ntuple(_ -> zeros(ComplexF64, nbounded, nwavenumbers), ncomponents)
    for component in 1:ncomponents, k in 1:nwavenumbers, j in 1:nbounded, mode in 1:nmodes
        fields[component][j, k] += modes[component][mode, j, k] * a[mode, k]
    end
    return fields
end

@testset verbose=true "Galerkin projection and expansion                           " begin
    @testset verbose=true "Documented discrete contractions                            " begin
        g = line_grid(Nb=9, Nh=5)
        nmodes, ncomponents = 2, 2
        nwavenumbers = size(FTField(g), 2)
        modes = _mode_arrays(ncomponents, size(g, 1), nmodes, nwavenumbers)
        u = _fill_vector_field!(VectorField(g, FTField; N=ncomponents))
        expected_coefficients = _expected_project(u, modes, weights(g))

        for algorithm in (LoopGalerkin(), GemmGalerkin())
            a = ProjectedField(g, modes)
            parent(a) .= 3
            @test project!(a, u, algorithm) === a
            @test parent(a) ≈ expected_coefficients
        end

        a = _fill_coefficients!(ProjectedField(g, modes))
        expected_fields = _expected_expand(a, modes)
        for algorithm in (LoopGalerkin(), GemmGalerkin())
            output = VectorField(g, FTField; N=ncomponents)
            output .= 7
            @test expand!(output, a, algorithm) === output
            @test all(parent(output[n]) ≈ expected_fields[n] for n in 1:ncomponents)
        end
    end

    @testset verbose=true "Loop and GEMM algorithms agree                              " begin
        g = line_grid(Nb=9, Nh=7)
        modes = _mode_arrays(3, size(g, 1), 3, size(FTField(g), 2))
        u = _fill_vector_field!(VectorField(g, FTField; N=3))

        loop_coefficients = project(u, modes, LoopGalerkin())
        gemm_coefficients = project(u, modes, GemmGalerkin())
        @test parent(gemm_coefficients) ≈ parent(loop_coefficients)

        _fill_coefficients!(loop_coefficients)
        parent(gemm_coefficients) .= parent(loop_coefficients)
        loop_field = expand(loop_coefficients, LoopGalerkin())
        gemm_field = expand(gemm_coefficients, GemmGalerkin())
        @test all(parent(gemm_field[n]) ≈ parent(loop_field[n]) for n in eachindex(loop_field))
    end

    @testset verbose=true "Discrete orthonormal round trip                             " begin
        g = line_grid(Nb=9, Nh=5)
        ncomponents, nbounded = 3, size(g, 1)
        nwavenumbers = size(FTField(g), 2)
        nmodes = ncomponents * nbounded
        quadrature_weights = weights(g)

        modes = ntuple(ncomponents) do component
            data = zeros(ComplexF64, nmodes, nbounded, nwavenumbers)
            for k in 1:nwavenumbers, j in 1:nbounded
                data[(component - 1) * nbounded + j, j, k] = inv(sqrt(quadrature_weights[j]))
            end
            data
        end
        coefficients = [complex(0.3mode - 0.2k, 0.4mode + 0.1k)
                        for mode in 1:nmodes, k in 1:nwavenumbers]

        for algorithm in (LoopGalerkin(), GemmGalerkin())
            a = ProjectedField(g, modes)
            parent(a) .= coefficients
            reconstructed = expand(a, algorithm)
            projected = project(reconstructed, modes, algorithm)
            @test parent(projected) ≈ coefficients
        end
    end
end
