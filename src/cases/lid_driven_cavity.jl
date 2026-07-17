# ////////////////////////////////////////////////////////////////////////////////////////////// #
# ///   Two-dimensional lid-driven-cavity layout                                             /// #
# ////////////////////////////////////////////////////////////////////////////////////////////// #

"""
    LID_DRIVEN_CAVITY_2D_AXES = (1, 2, nothing, 3)

Map logical coordinates `(x, y, z, t)` to the 2D lid-driven-cavity storage
order `(x, y, t)`. The spanwise coordinate is absent and time occupies storage
dimension 3.
"""
const LID_DRIVEN_CAVITY_2D_AXES = (1, 2, nothing, 3)

"""
    LID_DRIVEN_CAVITY_2D_FFT_ORDER = (3,)

The 2D cavity's only Fourier storage dimension: optional unit-period time or
phase. Both spatial directions use FDGrids matrices.
"""
const LID_DRIVEN_CAVITY_2D_FFT_ORDER = (3,)

"""
    LID_DRIVEN_CAVITY_2D_INHOMOGENEOUS_DIMS = (1, 2)

The bounded 2D cavity storage dimensions `(x, y)` discretised with FDGrids.
"""
const LID_DRIVEN_CAVITY_2D_INHOMOGENEOUS_DIMS = (1, 2)

# ////////////////////////////////////////////////////////////////////////////////////////////// #
# ///   Three-dimensional lid-driven-cavity layouts                                          /// #
# ////////////////////////////////////////////////////////////////////////////////////////////// #

"""
    LID_DRIVEN_CAVITY_3D_AXES = (1, 2, 3, 4)

Map logical coordinates `(x, y, z, t)` to the 3D lid-driven-cavity storage
order `(x, y, z, t)`. The spanwise `z` direction may be bounded or periodic;
the choice changes the FFT order but not this axis map.
"""
const LID_DRIVEN_CAVITY_3D_AXES = (1, 2, 3, 4)

"""
    LID_DRIVEN_CAVITY_3D_PERIODIC_FFT_ORDER = (3, 4)

Fourier storage order for a 3D cavity with periodic spanwise `z`: dimension 3
uses the real-to-complex transform and unit-period time uses a complex transform.
"""
const LID_DRIVEN_CAVITY_3D_PERIODIC_FFT_ORDER = (3, 4)

"""
    LID_DRIVEN_CAVITY_3D_BOUNDED_FFT_ORDER = (4,)

Fourier storage order for a 3D cavity with bounded spanwise `z`. Only the
optional unit-period time or phase direction is transformed.
"""
const LID_DRIVEN_CAVITY_3D_BOUNDED_FFT_ORDER = (4,)

# ////////////////////////////////////////////////////////////////////////////////////////////// #
# ///   Lid-driven-cavity layout contracts                                                   /// #
# ////////////////////////////////////////////////////////////////////////////////////////////// #

"""
    AbstractLidDrivenCavity2DGrid{T}

Layout contract for serial, MPI-decomposed, and downstream 2D cavity grids with
real scalar type `T`, storage order `(x, y, t)`, and Fourier time.
"""
const AbstractLidDrivenCavity2DGrid{T} =
    AbstractGrid{T, 3, LID_DRIVEN_CAVITY_2D_AXES, LID_DRIVEN_CAVITY_2D_FFT_ORDER}

"""
    AbstractLidDrivenCavity3DPeriodicGrid{T}

Layout contract for 3D cavity grids with bounded `(x,y)`, periodic `z`, and
Fourier time.
"""
const AbstractLidDrivenCavity3DPeriodicGrid{T} =
    AbstractGrid{T, 4, LID_DRIVEN_CAVITY_3D_AXES, LID_DRIVEN_CAVITY_3D_PERIODIC_FFT_ORDER}

"""
    AbstractLidDrivenCavity3DBoundedGrid{T}

Layout contract for 3D cavity grids whose three spatial directions are bounded
and use FDGrids, with only time transformed.
"""
const AbstractLidDrivenCavity3DBoundedGrid{T} =
    AbstractGrid{T, 4, LID_DRIVEN_CAVITY_3D_AXES, LID_DRIVEN_CAVITY_3D_BOUNDED_FFT_ORDER}

"""
    AbstractLidDrivenCavity3DGrid{T}

Union of the periodic- and bounded-spanwise 3D cavity layout contracts.
"""
const AbstractLidDrivenCavity3DGrid{T} = Union{
    AbstractLidDrivenCavity3DPeriodicGrid{T},
    AbstractLidDrivenCavity3DBoundedGrid{T},
}

"""
    AbstractLidDrivenCavityGrid{T}

Union of every supported 2D and 3D lid-driven-cavity layout. Flow constructors
use this structural contract so the bundled [`LidDrivenCavityGrid`](@ref),
MPI-decomposed grids, and compatible downstream grids share the same API.
"""
const AbstractLidDrivenCavityGrid{T} = Union{
    AbstractLidDrivenCavity2DGrid{T},
    AbstractLidDrivenCavity3DPeriodicGrid{T},
    AbstractLidDrivenCavity3DBoundedGrid{T},
}

# ////////////////////////////////////////////////////////////////////////////////////////////// #
# ///   Finite-difference construction                                                       /// #
# ////////////////////////////////////////////////////////////////////////////////////////////// #

"""
    _lid_driven_cavity_fd(N, lim, dist, width, T)

Build one bounded FDGrids direction and its quadrature-weighted adjoints. The
returned tuple is ordered as `(points, D₁, D₂, D₁⁺, D₂⁺, weights)`.
"""
function _lid_driven_cavity_fd(N, lim, dist, width, ::Type{T}) where {T}
    grid = FDGrids.grid(N, lim[1], lim[2], dist)
    xs, ws = Vector{T}(grid.xs), Vector{T}(grid.ws)
    D₁ = FDGrids.DiffMatrix(xs, width, 1; eltype=T)
    D₂ = FDGrids.DiffMatrix(xs, width, 2; eltype=T)
    return xs, D₁, D₂, LinearAlgebra.adjoint(D₁, ws), LinearAlgebra.adjoint(D₂, ws), ws
end

# ////////////////////////////////////////////////////////////////////////////////////////////// #
# ///   Two-dimensional lid-driven-cavity grid factory                                      /// #
# ////////////////////////////////////////////////////////////////////////////////////////////// #

"""
    LidDrivenCavityGrid(Nx, Ny; Nt=1, lim=(0, 1), xlim=lim, ylim=lim,
                        dist=FDGrids.UniformGrid(), xdist=dist, ydist=dist,
                        width=5, xwidth=width, ywidth=width, T=Float64)
    LidDrivenCavityGrid(Nx, Ny, Nz; Nt=1, spanwise=:bounded, Lz=1,
                        lim=(0, 1), xlim=lim, ylim=lim, zlim=lim,
                        dist=FDGrids.UniformGrid(), xdist=dist, ydist=dist,
                        zdist=dist, width=5, xwidth=width, ywidth=width,
                        zwidth=width, T=Float64)

Construct a 2D or 3D lid-driven-cavity [`RectangularGrid`](@ref). The number of
positional resolutions is the number of spatial dimensions; `Nt` is a keyword
so `LidDrivenCavityGrid(Nx,Ny,Nz)` unambiguously means a steady 3D cavity.

The 2D layout is `(x,y,t)`. A 3D grid uses `(x,y,z,t)` and accepts
`spanwise=:bounded` or `spanwise=:periodic`:

- `:bounded` builds an independent FDGrids `z` direction on `zlim`. The grid
  supplies boundary points and operators but does not impose no-slip values.
- `:periodic` uses a Fourier `z` direction of physical length `Lz`; its
  wavenumber scale is computed internally as `2π/Lz`.

# Common keywords

- `lim`: default interval for every bounded spatial direction.
- `xlim`, `ylim`, `zlim`: per-direction bounded intervals.
- `dist`: default FDGrids distribution; the default is `UniformGrid()`.
- `xdist`, `ydist`, `zdist`: per-direction distribution overrides.
- `width`: default finite-difference stencil width.
- `xwidth`, `ywidth`, `zwidth`: per-direction stencil-width overrides.
- `Nt`: positive odd temporal resolution, defaulting to `1`.
- `T`: real scalar type for all grid data.

Every finite-difference width must be odd and at least three, and its point
count must exceed twice the width. Periodic resolutions must be positive and
odd. Boundary values are deliberately not encoded by the grid.

# Examples

```julia
g2 = LidDrivenCavityGrid(65, 49; Nt=1, xlim=(-1, 1), width=7)
g3 = LidDrivenCavityGrid(65, 49, 33; spanwise=:bounded, width=7)
g3p = LidDrivenCavityGrid(65, 49, 31; spanwise=:periodic, Lz=2)
```
"""
function LidDrivenCavityGrid(Nx::Int, Ny::Int; Nt::Int=1, lim=(0, 1), xlim=lim, ylim=lim,
                             dist=FDGrids.UniformGrid(), xdist=dist, ydist=dist,
                             width=5, xwidth=width, ywidth=width, T::Type{<:Real}=Float64)
    x, Dₓ, Dₓₓ, Dₓ⁺, Dₓₓ⁺, wₓ = _lid_driven_cavity_fd(Nx, xlim, xdist, xwidth, T)
    y, Dᵧ, Dᵧᵧ, Dᵧ⁺, Dᵧᵧ⁺, wᵧ = _lid_driven_cavity_fd(Ny, ylim, ydist, ywidth, T)
    return RectangularGrid((x, y), (Dₓ, Dᵧ), (Dₓₓ, Dᵧᵧ), (Dₓ⁺, Dᵧ⁺), (Dₓₓ⁺, Dᵧᵧ⁺), (wₓ, wᵧ),
                           (2π,), (Nx, Ny, Nt), LID_DRIVEN_CAVITY_2D_AXES, LID_DRIVEN_CAVITY_2D_FFT_ORDER, T)
end

# ////////////////////////////////////////////////////////////////////////////////////////////// #
# ///   Three-dimensional lid-driven-cavity grid factory                                    /// #
# ////////////////////////////////////////////////////////////////////////////////////////////// #

function LidDrivenCavityGrid(Nx::Int, Ny::Int, Nz::Int; Nt::Int=1, spanwise=:bounded, Lz=1,
                             lim=(0, 1), xlim=lim, ylim=lim, zlim=lim,
                             dist=FDGrids.UniformGrid(), xdist=dist, ydist=dist, zdist=dist,
                             width=5, xwidth=width, ywidth=width, zwidth=width, T::Type{<:Real}=Float64)
    x, Dₓ, Dₓₓ, Dₓ⁺, Dₓₓ⁺, wₓ = _lid_driven_cavity_fd(Nx, xlim, xdist, xwidth, T)
    y, Dᵧ, Dᵧᵧ, Dᵧ⁺, Dᵧᵧ⁺, wᵧ = _lid_driven_cavity_fd(Ny, ylim, ydist, ywidth, T)
    if spanwise === :periodic
        Lz > 0 || throw(ArgumentError("Lz must be positive"))
        return RectangularGrid((x, y), (Dₓ, Dᵧ), (Dₓₓ, Dᵧᵧ), (Dₓ⁺, Dᵧ⁺), (Dₓₓ⁺, Dᵧᵧ⁺), (wₓ, wᵧ),
                               (2π / Lz, 2π), (Nx, Ny, Nz, Nt), LID_DRIVEN_CAVITY_3D_AXES, LID_DRIVEN_CAVITY_3D_PERIODIC_FFT_ORDER, T)
    elseif spanwise === :bounded
        z, D_z, D_zz, D_z⁺, D_zz⁺, w_z = _lid_driven_cavity_fd(Nz, zlim, zdist, zwidth, T)
        return RectangularGrid((x, y, z), (Dₓ, Dᵧ, D_z), (Dₓₓ, Dᵧᵧ, D_zz), (Dₓ⁺, Dᵧ⁺, D_z⁺), (Dₓₓ⁺, Dᵧᵧ⁺, D_zz⁺),
                               (wₓ, wᵧ, w_z), (2π,), (Nx, Ny, Nz, Nt), LID_DRIVEN_CAVITY_3D_AXES, LID_DRIVEN_CAVITY_3D_BOUNDED_FFT_ORDER, T)
    end
    throw(ArgumentError("spanwise must be :bounded or :periodic, got $spanwise"))
end

# ////////////////////////////////////////////////////////////////////////////////////////////// #
# ///   Two-dimensional lid-driven-cavity flow                                               /// #
# ////////////////////////////////////////////////////////////////////////////////////////////// #

"""
    LidDrivenCavityFlow(g, Re; base, mode=AdjointDiscrete(),
                        fftw_flags=FFTW.EXHAUSTIVE, dealias=true) -> ProjectedNSE

Construct the primitive-variable equations for a 2D or 3D lid-driven cavity at
Reynolds number `Re`. The formulation is selected from the grid layout:
[`CartesianPrimitive2D`](@ref) for `(x,y,t)` and
[`CartesianPrimitive3D`](@ref) for `(x,y,z,t)`.

`base` is required. It must contain `(U,V)` for a 2D grid or `(U,V,W)` for a
3D grid. This explicit lifting carries the inhomogeneous moving-lid values;
the perturbation basis or residual must enforce homogeneous wall conditions.
Individual components may be `nothing`, but NSEBase never silently substitutes
a physically inappropriate zero reference state. On a decomposed grid, base
arrays use that rank's local inhomogeneous shape.

`mode` selects the continuous or quadrature-consistent discrete adjoint,
`fftw_flags` controls FFTW planning, and `dealias=true` pads only Fourier
directions for nonlinear products.

# Example

```julia
g = LidDrivenCavityGrid(65, 49; width=7)
X, Y, _ = points(g)
ψₓ, ψᵧ = @. 32X * (1 - X) * (1 - 2X) * Y^2 * (Y - 1), @. 16X^2 * (1 - X)^2 * (3Y^2 - 2Y)
eq = LidDrivenCavityFlow(g, 1000; base=(ψᵧ, -ψₓ), fftw_flags=FFTW.ESTIMATE)
```
"""
function LidDrivenCavityFlow(g::AbstractLidDrivenCavity2DGrid, Re; base, mode=AdjointDiscrete(),
                             fftw_flags=FFTW.EXHAUSTIVE, dealias=true)
    length(base) == 2 || throw(ArgumentError("a 2D cavity base must contain (U, V)"))
    construct_equations(g, Re, base, CartesianPrimitive2D(); mode, flags=fftw_flags, dealias)
end

# ////////////////////////////////////////////////////////////////////////////////////////////// #
# ///   Three-dimensional lid-driven-cavity flow                                             /// #
# ////////////////////////////////////////////////////////////////////////////////////////////// #

function LidDrivenCavityFlow(g::AbstractLidDrivenCavity3DGrid, Re; base, mode=AdjointDiscrete(),
                             fftw_flags=FFTW.EXHAUSTIVE, dealias=true)
    length(base) == 3 || throw(ArgumentError("a 3D cavity base flow must contain (U, V, W)"))
    construct_equations(g, Re, base, CartesianPrimitive3D(); mode, flags=fftw_flags, dealias)
end
