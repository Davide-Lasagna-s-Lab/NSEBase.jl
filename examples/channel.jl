using NSEBase

let
    grid = ChannelGrid(9, 17, 9; Nt=1, α=0.5, β=0.5, width=5)
    couette = PlaneCouetteFlow(grid, 500; Ro=0.1, fftw_flags=FFTW.ESTIMATE, dealias=false)
    poiseuille = PlanePoiseuilleFlow(grid, 500; f=1, fftw_flags=FFTW.ESTIMATE, dealias=false)
    state, residual = VectorField(grid; N=3), VectorField(grid; N=3)
    poiseuille.nl(0.0, state, residual)
    println("channel: size=$(size(grid)), state=(u,v,w), force=$(typeof(poiseuille.nl.force))")
    (; grid, couette, poiseuille, state, residual)
end
