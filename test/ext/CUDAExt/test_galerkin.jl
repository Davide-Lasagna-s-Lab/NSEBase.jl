@testset verbose=true "CUDA Galerkin methods                                       " begin
    g = CUDA_CHANNEL_GRID
    Ny, Nx, Nz, Nt = size(g)

    # Construct a weighted-orthonormal basis at every homogeneous index.
    M = 5
    Ψ = ntuple(_ -> zeros(ComplexF64, M, Ny, (Nx >> 1) + 1, Nz, Nt), 1)
    rng = MersenneTwister(17)
    w = collect(weights(g))
    weighted_basis = T -> (Diagonal(1 ./ sqrt.(w))*qr(Diagonal(sqrt.(w))*randn(rng, T, Ny, M)).Q[:, 1:M])'

    for nt in 1:Nt, nz in 1:Nz, nx in 2:(Nx >> 1) + 1
        Ψ[1][:, :, nx, nz, nt] .= weighted_basis(ComplexF64)
    end
    for nz in 2:(Nz >> 1) + 1, nt in 2:Nt
        Ψ[1][:, :, 1,     nz,       nt]   .= weighted_basis(ComplexF64)
        Ψ[1][:, :, 1, end-nz+2, end-nt+2] .= conj.(Ψ[1][:, :, 1, nz, nt])
    end
    for nz in 2:Nz
        Ψ[1][:, :, 1,     nz,   1] .= weighted_basis(ComplexF64)
        Ψ[1][:, :, 1, end-nz+2, 1] .= conj.(Ψ[1][:, :, 1, nz, 1])
    end
    for nt in 2:Nt
        Ψ[1][:, :, 1, 1,     nt]   .= weighted_basis(ComplexF64)
        Ψ[1][:, :, 1, 1, end-nt+2] .= conj.(Ψ[1][:, :, 1, 1, nt])
    end
    Ψ[1][:, :, 1, 1, 1] .= weighted_basis(Float64)

    # Construct compatible host fields, then move the complete objects to the device.
    u = VectorField(g; N=1)
    coefficients = randn(rng, ComplexF64, M, (Nx >> 1) + 1, Nz, Nt)
    a = ProjectedField(g, coefficients, Ψ)
    ud = CUDA.cu(u)
    ad = CUDA.cu(a)

    # construct all galerkin methods
    project_methods = [
                       CUDAExt.ProjectLoop(),
                       CUDAExt.ProjectShared(ad, ud),
                    #    CUDAExt.ProjectSharedTiled(ad, ud), # ! not implemented yet
                    #    CUDAExt.ProjectSharedTree(ad, ud),  # ! not implemented yet
                       ]
    expand_methods  = [
                       CUDAExt.ExpandModal(over_vector=true),
                       CUDAExt.ExpandModal(over_vector=false),
                       ]

    @testset verbose=true "Projection and expansion methods are compatible             " for pmethod in project_methods, emethod in expand_methods
        out_d = project!(zero(ad), expand!(ud, ad, emethod), pmethod)
        @test norm(Array(parent(ad - out_d))) < 1e-5
    end

    # A distinct production-grid type provides a second autotuning cache key.
    g2 = channel_grid(Nx=3, Ny=16, Nz=1, Nt=1)
    Ψ2 = test_modes(g2; ncomponents=1, nmodes=M, seed=11)
    ud2 = CUDA.cu(VectorField(g2; N=1))
    ad2 = CUDA.cu(ProjectedField(g2, Ψ2))

    @testset verbose=true "Projection autotuning caches by concrete field types        " begin
        reset_project_cache!()
        @test isempty(CUDAExt.PROJECT_METHODS)
        CUDAExt.project_method(ad, ud)
        @test length(CUDAExt.PROJECT_METHODS) == 1
        reset_project_cache!()
        @test isempty(CUDAExt.PROJECT_METHODS)
        CUDAExt.project_method(ad, ud)
        CUDAExt.project_method(ad2, ud2)
        @test length(CUDAExt.PROJECT_METHODS) == 2
        reset_project_cache!(ad2, ud2)
        @test length(CUDAExt.PROJECT_METHODS) == 1
        CUDAExt.project_method(ad, ud)
        @test length(CUDAExt.PROJECT_METHODS) == 1
        CUDAExt.project_method(ad2, ud2)
        reset_project_cache!()
        @test isempty(CUDAExt.PROJECT_METHODS)
    end

    @testset verbose=true "Expansion autotuning caches by concrete field types         " begin
        reset_expand_cache!()
        @test isempty(CUDAExt.EXPAND_METHODS)
        CUDAExt.expand_method(ud, ad)
        @test length(CUDAExt.EXPAND_METHODS) == 1
        reset_expand_cache!()
        @test isempty(CUDAExt.EXPAND_METHODS)
        CUDAExt.expand_method(ud, ad)
        CUDAExt.expand_method(ud2, ad2)
        @test length(CUDAExt.EXPAND_METHODS) == 2
        reset_expand_cache!(ud2, ad2)
        @test length(CUDAExt.EXPAND_METHODS) == 1
        CUDAExt.expand_method(ud, ad)
        @test length(CUDAExt.EXPAND_METHODS) == 1
        CUDAExt.expand_method(ud2, ad2)
        reset_expand_cache!()
        @test isempty(CUDAExt.EXPAND_METHODS)
    end
end
