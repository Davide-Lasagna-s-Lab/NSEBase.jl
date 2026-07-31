"""
    test_ftfield(grid; seed=1) -> FTField

Construct a deterministic spectral field with the exact storage shape required
by `grid`. The public `FTField` constructor performs the same Hermitian
sanitisation that production inputs receive.
"""
function test_ftfield(g::AbstractGrid; seed::Int=1)
    rng = MersenneTwister(seed)
    return FTField(g, randn(rng, Complex{eltype(g)}, transform_size(g)))
end

"""
    test_modes(grid; ncomponents=3, nmodes=2, seed=1) -> NTuple

Construct a deterministic, correctly laid-out modal basis. Each component has
shape `(mode, inhomogeneous..., homogeneous...)`, matching the public
`ProjectedField` and Galerkin contracts.
"""
function test_modes(g::AbstractGrid; ncomponents::Int=3, nmodes::Int=2, seed::Int=1)
    rng = MersenneTwister(seed)
    physical_size = size(g)
    spectral_size = transform_size(g)
    inhomogeneous_size = map(dim -> physical_size[dim], inhomogeneous_storage_dims(g))
    homogeneous_size = map(dim -> spectral_size[dim], fft_storage_dims(g))
    mode_size = (nmodes, inhomogeneous_size..., homogeneous_size...)
    return ntuple(_ -> randn(rng, Complex{eltype(g)}, mode_size), ncomponents)
end
