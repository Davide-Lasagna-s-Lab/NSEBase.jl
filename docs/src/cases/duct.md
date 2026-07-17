# Square-duct flow

`SquareDuctGrid` represents the square cross-section `[0,1]²`, with periodic
streamwise `z` and optional phase or time `t`.

| Storage dimension | Physical direction | Method |
|:--:|:--:|:--|
| 1 | `x` | FDGrids |
| 2 | `y` | FDGrids |
| 3 | `z` | rFFT, period `2π/α` |
| 4 | `t` | FFT, period `1` |

## Grid

```julia
using NSEBase

grid = SquareDuctGrid(49, 63, 1, 0.5; width=7)
```

The preferred arguments are `(N,Nz,Nt,α)`, with `width` as a keyword. The
historical ReSolver-SquareDuct positional form remains available:

```julia
grid = SquareDuctGrid(49, 7, 63, 1, 0.5)
```

One FDGrids discretisation is deliberately shared by the two cross-section
directions, including points, weights, derivatives, and adjoints. Product
quadrature is cached in `grid.w` and returned by `weights(grid)`.

`Nz` and `Nt` must be positive and odd. `width` must be odd and at least three,
with `N > 2width` for the compact weighted adjoints.

## Pressure-driven equations

```julia
equations = SquareDuctFlow(grid, 2000; f=1, fftw_flags=FFTW.ESTIMATE, dealias=false)
```

`f` acts on velocity component three, the streamwise direction, and only at
the zero `(z,t)` Fourier mode. `base` defaults to
`(nothing,nothing,nothing)`; place a broadcast-compatible `N×N` streamwise
base array in component three to linearise about a laminar solution.

The grid includes wall points when the selected distribution provides them,
but the basis or residual remains responsible for imposing no slip.
