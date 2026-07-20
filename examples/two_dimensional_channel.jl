using NSEBase

let
    grid = TwoDimensionalChannelGrid(9, 17; Nt=1, α=0.5, width=5)
    equations = PlanePoiseuilleFlow(grid, 500; f=1, fftw_flags=FFTW.ESTIMATE, dealias=false)
    state, residual = VectorField(grid; N=2), VectorField(grid; N=2)
    equations.nl(0.0, state, residual)
    println("two-dimensional Poiseuille: size=$(size(grid)), state=(u,v), force=$(typeof(equations.nl.force))")
    (; grid, equations, state, residual)
end
