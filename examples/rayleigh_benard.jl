using NSEBase

let
    grid_2d = TwoDimensionalChannelGrid(9, 17; Nt=1, α=0.5, width=5)
    equations_2d = RayleighBenardFlow(grid_2d, 1.0, 0.71, 1000.0;
                                      fftw_flags=FFTW.ESTIMATE, dealias=false)
    state_2d, residual_2d = VectorField(grid_2d; N=3), VectorField(grid_2d; N=3)
    add_base_flow!(state_2d, equations_2d.base)
    equations_2d.nl(0.0, state_2d, residual_2d)

    grid_3d = ChannelGrid(9, 17, 9; Nt=1, α=0.5, β=0.5, width=5)
    equations_3d = RayleighBenardFlow(grid_3d, 1.0, 0.71, 1000.0;
                                      fftw_flags=FFTW.ESTIMATE, dealias=false)
    state_3d, residual_3d = VectorField(grid_3d; N=4), VectorField(grid_3d; N=4)
    add_base_flow!(state_3d, equations_3d.base)
    equations_3d.nl(0.0, state_3d, residual_3d)

    println("2D Rayleigh–Bénard: size=$(size(grid_2d)), state=(u,v,θ), Ri=$(equations_2d.nl.Ri)")
    println("3D Rayleigh–Bénard: size=$(size(grid_3d)), state=(u,v,w,θ), Ri=$(equations_3d.nl.Ri)")
    (; two_dimensional=(grid=grid_2d, equations=equations_2d, state=state_2d, residual=residual_2d),
       three_dimensional=(grid=grid_3d, equations=equations_3d, state=state_3d, residual=residual_3d))
end
