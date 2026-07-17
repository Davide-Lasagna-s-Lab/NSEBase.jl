"""Build a real `RectangularGrid` for tests from compact layout data."""
function rectangular_test_grid(grid_size::NTuple{D, Int}, axes::Tuple, fft_dims::Tuple;
                               scales=ntuple(_ -> 1.0, length(fft_dims)), limits=nothing,
                               quadrature_weights=nothing, width=3, T=Float64) where {D}
    inhomogeneous_dims = Tuple(dim for dim in 1:D if dim ∉ fft_dims)
    NI = length(inhomogeneous_dims)
    limits === nothing && (limits = ntuple(_ -> (-1, 1), NI))

    fdgrids = map(inhomogeneous_dims, limits) do dim, lim
        FDGrids.grid(grid_size[dim], T(lim[1]), T(lim[2]), FDGrids.UniformGrid())
    end
    xs = map(grid -> Vector{T}(grid.xs), fdgrids)
    ws = quadrature_weights === nothing ? map(grid -> Vector{T}(grid.ws), fdgrids) : map(w -> Vector{T}(w), quadrature_weights)
    D₁ = map(x -> FDGrids.DiffMatrix(x, width, 1; eltype=T), xs)
    D₂ = map(x -> FDGrids.DiffMatrix(x, width, 2; eltype=T), xs)
    D₁⁺ = map(LinearAlgebra.adjoint, D₁, ws)
    D₂⁺ = map(LinearAlgebra.adjoint, D₂, ws)
    return RectangularGrid(xs, D₁, D₂, D₁⁺, D₂⁺, ws, scales, grid_size, axes, fft_dims, T)
end

"""One finite-difference direction followed by one periodic direction."""
function planar_test_grid(Nx::Int, Ny::Int; L=2π, xlim=(-1, 1), ws=nothing, T=Float64)
    quadrature_weights = ws === nothing ? nothing : (ws,)
    return rectangular_test_grid((Nx, Ny), (1, 2, nothing, nothing), (2,);
                                 scales=(T(2π / L),), limits=(xlim,), quadrature_weights, T)
end

"""Two-dimensional `(y, x)` grid with periodic physical `x`."""
function shear_test_grid(Ny::Int, Nx::Int; Lx=2π, ylim=(-1, 1), width=3, T=Float64)
    return rectangular_test_grid((Ny, Nx), (2, 1, nothing, nothing), (2,);
                                 scales=(T(2π / Lx),), limits=(ylim,), width, T)
end

"""Channel-like steady grid stored as `(y, x, z)`."""
function channel_test_grid(Ny::Int, Nx::Int, Nz::Int; α=1.0, β=1.0, ylim=(-1, 1), T=Float64)
    return rectangular_test_grid((Ny, Nx, Nz), (2, 1, 3, nothing), (2, 3);
                                 scales=(T(α), T(β)), limits=(ylim,), T)
end

"""Channel-like steady grid with Fourier dimensions before the wall-normal dimension."""
function flipped_channel_test_grid(Nx::Int, Nz::Int, Ny::Int; α=1.0, β=1.0, ylim=(-1, 1), T=Float64)
    return rectangular_test_grid((Nx, Nz, Ny), (1, 3, 2, nothing), (1, 2);
                                 scales=(T(α), T(β)), limits=(ylim,), T)
end

"""Change only a concrete test grid's size parameter, bypassing outer-constructor validation."""
function unchecked_rectangular_test_size(g::RectangularGrid{NI, T, S0, D, AXES, FFT_DIMS, XS, D1, D2, A1, A2, WS, WP, NFFT},
                                         ::Val{S}) where {NI, T, S0, D, AXES, FFT_DIMS, XS, D1, D2, A1, A2, WS, WP, NFFT, S}
    return RectangularGrid{NI, T, S, D, AXES, FFT_DIMS, XS, D1, D2, A1, A2, WS, WP, NFFT}(
        g.xs, g.D₁, g.D₂, g.D₁⁺, g.D₂⁺, g.ws, g.w, g.scales)
end

"""Select all inhomogeneous points at the homogeneous zero mode."""
inhomogeneous_slice(g::AbstractGrid{<:Any, D}) where {D} = ntuple(dim -> dim in fft_storage_dims(g) ? 1 : Colon(), D)

function ftfield_from_inhomogeneous(g::AbstractGrid, values)
    u = FTField(g)
    view(parent(u), inhomogeneous_slice(g)...) .= values
    return u
end

function apply_dd(g::AbstractGrid, values, dim=only(inhomogeneous_storage_dims(g)); adjoint=false)
    out = FTField(g)
    dd!(out, ftfield_from_inhomogeneous(g, values), Val(dim); adjoint)
    return copy(view(parent(out), inhomogeneous_slice(g)...))
end

function apply_lap(g::AbstractGrid, values; adjoint=false)
    out = FTField(g)
    inhomogeneous_laplacian!(out, ftfield_from_inhomogeneous(g, values); adjoint)
    return copy(view(parent(out), inhomogeneous_slice(g)...))
end

weighted_dot(g::AbstractGrid, u, v) = dot(weights(g) .* u, v)

function physical_to_ft(g::AbstractGrid, f; flags=FFTW.ESTIMATE, dealias=false, plans=nothing)
    plans = isnothing(plans) ? FFTPlans(g; flags, dealias) : plans
    physical = Field(g, f; dealias)
    spectral = FTField(g)
    plans(spectral, physical)
    return (; physical, spectral, plans)
end

function deterministic_ftfield(g::AbstractGrid, phase)
    dims = transform_size(g)
    n = reshape(collect(1:prod(dims)), dims)
    return FTField(g, sin.(phase .+ 0.13 .* n) .+ im .* cos.(phase .+ 0.17 .* n))
end
