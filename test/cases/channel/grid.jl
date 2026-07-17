@testset verbose=true "Grid layout, coordinates, and growth                        " begin
    Ny, Nx, Nz, Nt = 11, 7, 7, 3
    α, β = 1.25, 0.75
    g = ChannelGrid(Nx, Ny, Nz; Nt, α, β, width=5)
    positional = ChannelGrid(Nx, Ny, Nz, Nt, 5, α, β)

    @test g isa AbstractChannel3DGrid
    @test g isa RectangularGrid{1} && points(positional) == points(g)
    @test size(g) == (Ny, Nx, Nz, Nt)
    @test CHANNEL_3D_AXES == (2, 1, 3, 4)
    @test CHANNEL_3D_FFT_ORDER == (2, 3, 4)
    @test CHANNEL_3D_INHOMOGENEOUS_DIMS == (1,)
    @test fft_physical_dims(g) == (:x, :z, :t)
    @test inhomogeneous_physical_dims(g) == (:y,)
    @test g.scales == (Float64(α), Float64(β), 2π)
    @test weights(g) === g.ws[1]
    @test g.D₁⁺[1] isa FDGrids.AdjointDiffMatrix
    @test g.D₂⁺[1] isa FDGrids.AdjointDiffMatrix

    y, x, z, t = points(g)
    @test map(size, (y, x, z, t)) == ((Ny, 1, 1, 1), (1, Nx, 1, 1), (1, 1, Nz, 1), (1, 1, 1, Nt))
    @test vec(y) == g.xs[1]
    @test vec(x) ≈ (0:Nx-1) .* (2π / (α * Nx))
    @test vec(z) ≈ (0:Nz-1) .* (2π / (β * Nz))
    @test vec(t) ≈ (0:Nt-1) ./ Nt

    padded_size = NSEBase.get_padded_size(size(g), fft_storage_dims(g))
    padded_homogeneous_size = map(dim -> padded_size[dim], fft_storage_dims(g))
    @test points(g; dealias=true) == points(g, padded_homogeneous_size)

    target = (Nx + 2, Nz + 2, Nt + 2)
    grown = growto(g, target)
    gx, gz, gt = points(g, target)[2:4]
    @test typeof(grown) != typeof(g)
    @test size(grown) == (Ny, target...)
    @test points(grown) == points(g, target)
    @test vec(gx) ≈ (0:target[1]-1) .* (2π / (α * target[1]))
    @test vec(gz) ≈ (0:target[2]-1) .* (2π / (β * target[2]))
    @test vec(gt) ≈ (0:target[3]-1) ./ target[3]
    @test grown.xs[1] === g.xs[1]
    @test grown.ws[1] === g.ws[1]
    @test grown.D₁[1] === g.D₁[1]
    @test grown.D₂[1] === g.D₂[1]
    @test grown.D₁⁺[1] === g.D₁⁺[1]
    @test grown.D₂⁺[1] === g.D₂⁺[1]

    supplied = ChannelGrid(g.xs[1], Nx, Nz, Nt, α, β, g.D₁[1], g.D₂[1], g.ws[1])
    forward_adjoint = ChannelGrid(g.xs[1], Nx, Nz, Nt, α, β, g.D₁[1], g.D₂[1], g.ws[1]; adjoint_diff=false)
    @test supplied.D₁[1] === g.D₁[1]
    @test supplied.D₂[1] === g.D₂[1]
    @test supplied.D₁⁺[1] ≈ g.D₁⁺[1]
    @test forward_adjoint.D₁⁺[1] === forward_adjoint.D₁[1] && forward_adjoint.D₂⁺[1] === forward_adjoint.D₂[1]
end
