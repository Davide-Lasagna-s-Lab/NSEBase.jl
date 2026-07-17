function channel_wall_normal_modes(g; Nm=length(g.xs[1]), components=1)
    Ny = length(g.xs[1])
    Nm <= Ny || throw(ArgumentError("Nm must not exceed the wall-normal resolution"))
    homogeneous_size = map(dim -> transform_size(g)[dim], fft_storage_dims(g))
    first_component = zeros(Complex{eltype(g)}, Nm, Ny, homogeneous_size...)
    for I in CartesianIndices(homogeneous_size), m in 1:Nm
        first_component[m, m, Tuple(I)...] = inv(sqrt(g.ws[1][m]))
    end
    return ntuple(component -> component == 1 ? first_component : zero(first_component), components)
end
