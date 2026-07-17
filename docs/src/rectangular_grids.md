# Rectangular grids

`RectangularGrid{NI}` is the shared concrete grid for tensor-product Cartesian
domains with between one and three inhomogeneous FDGrids directions. It replaces
the near-identical grid types formerly maintained by each flow package.

## Prefer a case constructor

Case constructors fix low-level axis and FFT parameters and provide physical
defaults:

```julia
channel = ChannelGrid(63, 65, 63; Nt=1, α=0.5, β=1.0, width=7)
reduced_channel = ChannelGrid(65, 63; Nt=1, β=1.0, width=7)
duct = SquareDuctGrid(49, 63, 1, 0.5; width=7)
cavity = LidDrivenCavityGrid(65, 49; xlim=(-1,1), ylim=(0,1))
cavity3d = LidDrivenCavityGrid(65, 49, 31; spanwise=:periodic, Lz=2)
```

The constructors build collocation points, quadrature, first and second
derivatives, and quadrature-weighted adjoints. `dist` and `width` set defaults;
multi-direction cavity grids also accept per-direction forms such as `xdist`,
`ydist`, `xwidth`, and `ywidth`. The channel factory selects its 2D3C or 3D
layout from the number of positional spatial resolutions.

## Stored data

For `NI` inhomogeneous directions, these fields are explicit `NTuple{NI}`s:

| Field | Meaning |
|:--|:--|
| `xs` | Collocation vectors |
| `D₁`, `D₂` | First- and second-derivative operators |
| `D₁⁺`, `D₂⁺` | Supplied discrete weighted adjoints |
| `ws` | One-dimensional quadrature vectors |

`w` is the tensor product returned by `weights(g)`, and `scales` follows
`fft_storage_dims(g)` order. Different directions may use different grid sizes,
distributions, and stencil widths. A square duct intentionally shares the same
objects in both tuple entries.

## Custom construction

Use the low-level constructor when operators have already been prepared:

```julia
using FDGrids, LinearAlgebra, NSEBase

fd = FDGrids.grid(65, -1, 1, FDGrids.GaussLobattoGrid())
y, wy = fd.xs, fd.ws
D₁ = FDGrids.DiffMatrix(y, 7, 1)
D₂ = FDGrids.DiffMatrix(y, 7, 2)
D₁⁺ = LinearAlgebra.adjoint(D₁, wy)
D₂⁺ = LinearAlgebra.adjoint(D₂, wy)

grid = RectangularGrid((y,), (D₁,), (D₂,), (D₁⁺,), (D₂⁺,), (wy,),
                       (0.5, 1.0, 2π), (65, 63, 63, 1),
                       (2, 1, 3, 4), (2, 3, 4))
```

Direction-dependent tuples must follow `inhomogeneous_storage_dims(g)` order.
Scales must follow FFT order. Homogeneous physical sizes must be odd so 3/2
padding has the expected parity. A present logical time coordinate must be one
of the Fourier directions; the one to three FD directions are spatial.

## Ownership and conversion

Dense `Vector{T}` points and weights and same-typed matrix operators are retained
rather than copied; other vector containers are materialised as `Vector{T}`.
This preserves specialised FDGrids matrices and intentional sharing.
Caller-supplied adjoints are trusted: conversion converts them directly instead
of recomputing them from the forward matrices. Repeated identical objects are
converted once, so square-grid sharing survives precision conversion.

`convert(Float32, g)` converts every numerical object consistently.
`growto(g, sizes)` changes only homogeneous sizes and retains one-dimensional
FD data. Product quadrature is reconstructed from the retained `ws` tuples.

## Coordinates and scales

`points(g)` returns one broadcast-compatible array per storage dimension.
Inhomogeneous arrays contain the stored points; homogeneous arrays are uniform
on `[0,L)`, with `L=2π/wavenumber_scale(g,dim)`.

`points(g; dealias=true)` uses padded homogeneous sizes. Passing an explicit
tuple to `points(g, sizes)` or `growto(g, sizes)` uses FFT-order sizes, never
physical-coordinate order.

## Weighted adjoints

For one-dimensional weights `w`, FDGrids constructs the discrete adjoint that
satisfies

```math
\langle D u,v\rangle_w = \langle u,D^+v\rangle_w.
```

The rectangular product inner product is separable, so each directional
adjoint remains valid without forming Kronecker matrices. NSEBase applies the
one-dimensional operator directly along its array dimension.

## Boundary points are not boundary conditions

Gauss–Lobatto and uniform distributions include interval endpoints, allowing a
basis or residual to enforce wall conditions there. `RectangularGrid` itself
does not clamp, eliminate, or prescribe any boundary value.
