# NSEBase case examples

Each script constructs a small grid and its equation bundle, applies the
nonlinear operator once to a zero state, and prints the layout/state convention.
Run one from the repository root with, for example:

```sh
julia --project=. examples/channel.jl
```

The examples use `FFTW.ESTIMATE` and disable de-aliasing to keep setup fast.
Production calculations normally retain the case defaults
`fftw_flags=FFTW.EXHAUSTIVE` and `dealias=true`.

- `channel.jl`: plane Couette and plane Poiseuille flow
- `duct.jl`: square-duct flow and the historical positional-width constructor
- `lid_driven_cavity.jl`: 2D cavity construction with an explicit lid lifting
- `rpcf.jl`: streamwise-independent rotating plane Couette flow
- `rayleigh_benard.jl`: Boussinesq convection around the conduction state
