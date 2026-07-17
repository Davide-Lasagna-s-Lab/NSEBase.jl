using NSEBase

let
    grid = ChannelGrid(9, 17, 9; Nt=1, α=0.5, β=0.5, width=5)
    equations = RayleighBenardFlow(grid, 1.0, 0.71, 1000.0; fftw_flags=FFTW.ESTIMATE, dealias=false)
    state, residual = VectorField(grid; N=4), VectorField(grid; N=4)
    equations.nl(0.0, state, residual)
    println("Rayleigh–Bénard: size=$(size(grid)), state=(u,v,w,θ), Ri=$(equations.nl.Ri)")
    (; grid, equations, state, residual)
end
