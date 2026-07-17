using NSEBase

let
    grid = SquareDuctGrid(15, 9, 1, 0.5; width=5)
    equations = SquareDuctFlow(grid, 2000; f=1, fftw_flags=FFTW.ESTIMATE, dealias=false)

    # The historical positional-width constructor remains available.
    square_grid = SquareDuctGrid(13, 5, 9, 1, 0.5)
    square_equations = SquareDuctFlow(square_grid, 2000; fftw_flags=FFTW.ESTIMATE, dealias=false)
    state, residual = VectorField(grid; N=3), VectorField(grid; N=3)
    equations.nl(0.0, state, residual)
    println("square duct: size=$(size(grid)), state=(u,v,w), force=$(typeof(equations.nl.force))")
    (; grid, equations, square_grid, square_equations, state, residual)
end
