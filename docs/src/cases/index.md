# Bundled flow cases

All bundled cases return the same `RectangularGrid{NI}` implementation while
fixing geometry-specific layout and equation conventions.

| Case | Storage order | FD directions | Fourier directions | Formulation |
|:--|:--|:--|:--|:--|
| Full channel | `(y,x,z,t)` | `y` | `x,z,t` | 3D Cartesian |
| Square duct | `(x,y,z,t)` | `x,y` | `z,t` | 3D Cartesian |
| 2D cavity | `(x,y,t)` | `x,y` | `t` | 2D Cartesian |
| 3D cavity | `(x,y,z,t)` | `x,y` and optionally `z` | `t` and optionally `z` | 3D Cartesian |
| Streamwise-independent channel (RPCF) | `(y,z,t)` | wall-normal | spanwise, `t` | 2D3C Cartesian |
| Rayleigh–Bénard | `(y,x,z,t)` | `y` | `x,z,t` | 3D Boussinesq |

| Flow constructor | Default base | Body force |
|:--|:--|:--|
| `PlaneCouetteFlow` | layout-aware `U(η)=η` | optional layout-aware Coriolis |
| `PlanePoiseuilleFlow` | `U(η)=1-η²` | streamwise pressure gradient; optional Coriolis |
| `SquareDuctFlow` | zero | streamwise pressure gradient |
| `LidDrivenCavityFlow` | required user lifting | none |
| `RayleighBenardFlow` | conduction temperature `(1-η)/2` | Boussinesq buoyancy in the formulation |

The examples in the following pages show the recommended high-level
constructors. Runnable equivalents live in the repository's `examples/`
directory and are executed by the test suite.

Boundary conditions are deliberately absent from this table: every case leaves
wall values to its basis or residual formulation.
