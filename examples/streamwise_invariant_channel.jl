using NSEBase

let
    grid = StreamwiseInvariantChannelGrid(17, 9; Nt=1, β=0.5, width=5)
    equations = PlaneCouetteFlow(grid, 500; Ro=0.2, fftw_flags=FFTW.ESTIMATE, dealias=false)
    state, residual = VectorField(grid; N=3), VectorField(grid; N=3)
    equations.nl(0.0, state, residual)
    println("streamwise-invariant Couette: size=$(size(grid)), state=(v,w,u), force=$(typeof(equations.nl.force))")
    (; grid, equations, state, residual)
end
