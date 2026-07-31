# Test-suite organization

The suite separates contracts owned by NSEBase from numerical accuracy owned
by concrete grid packages:

- `interface/` tests public metadata, indexing, construction, and container
  behavior.
- `functionality/` tests reusable algorithms such as FFT execution,
  differentiation routing, norms, shifts, Galerkin operations, and persistence.
- `performance/` records allocation contracts after compilation warm-up.
- `ext/` contains MPI and CUDA integration tests.
- `support/` contains deterministic fixtures built exclusively from
  `ReSolverRectangularGrids`; it defines no test-only `AbstractGrid` subtype.

Analytical derivative, Laplacian, quadrature, norm, and shift tests for channel,
lid-driven-cavity, and square-duct grids live in ReSolverRectangularGrids. This
suite checks only how NSEBase consumes those production grids, avoiding a
second copy of the grid package's numerical tests.

Add a test to the narrowest folder that owns its contract. Shared data builders
belong in `support/`; a helper must not implement or extend the `AbstractGrid`
interface. Every testset uses `verbose=true` and a literal description padded
on the right to 60 characters so nested results remain readable in CI.

Run the complete suite from the repository root with:

```sh
julia --project=test test/runtests.jl
```
