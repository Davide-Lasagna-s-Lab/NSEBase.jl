using NSEBase

let
    grid = LidDrivenCavityGrid(13, 13; width=5)
    X, Y, _ = points(grid)

    # A divergence-free lifting with a regularised moving lid. Perturbations
    # can then satisfy homogeneous conditions on every wall.
    U = @. 16X^2 * (1 - X)^2 * (3Y^2 - 2Y)
    V = @. -32X * (1 - X) * (1 - 2X) * Y^2 * (Y - 1)
    equations = LidDrivenCavityFlow(grid, 1000; base=(U, V), fftw_flags=FFTW.ESTIMATE, dealias=false)
    state, residual = VectorField(grid; N=2), VectorField(grid; N=2)
    equations.nl(0.0, state, residual)

    println("2D cavity: size=$(size(grid)), state=(u,v), moving lid supplied by the base lifting")
    (; grid, equations, state, residual)
end
