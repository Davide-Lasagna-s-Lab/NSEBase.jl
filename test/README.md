# NSEBase test-suite organization

The suite separates implementation contracts from physical-case regressions. Generic tests exercise one public behavior at a time on production `RectangularGrid` objects. Case tests then verify the coordinates, quadrature, derivatives, adjoints, norms, shifts, and equation constructors that users rely on for each bundled flow.

## Layout

- `helpers/rectangular.jl` contains concrete-grid factories and small numerical reference helpers. It defines no grid subtype.
- `helpers/case_contracts.jl` contains shared assertion harnesses. Calling a harness from two cases executes the assertions twice, so common code does not reduce case coverage.
- `implementation/` tests accessors, fields, transforms, operators, and equation wrappers independently of a particular flow.
- `cases/` owns the numerical regressions migrated from the channel, square-duct, lid-driven-cavity, RPCF, and Rayleigh–Bénard packages. Channel tests are split by concern because that legacy suite already had six focused files.
- `integration/` covers persistence and runnable examples.
- `performance/` contains warmed allocation contracts.
- `ext/MPIExt/` runs the production `ChannelGrid` through decomposed-grid subprocess tests.

`IncompleteGrid` in `implementation/grids/abstractgrid.jl` and `AllocationIncompleteGrid` in the allocation suite are deliberate metadata-only probes. They exist solely to verify required-interface failures; neither supplies numerical grid behavior.

## Adding or changing tests

Put API mechanics in `implementation/` and physical normalization or accuracy checks in the owning case. Reuse a helper for setup or references when behavior is identical, but invoke the assertion from every case whose contract it represents. Allocation tests must construct fixtures outside the measured closure and warm the closure before measuring it. MPI tests remain subprocess-isolated and run last.

Every `@testset` uses `verbose=true` so nested results are visible. Descriptions are right-padded directly in their string literals to exactly 60 characters; keep the meaningful description at 60 characters or fewer and do not compute the padding at runtime. MPI descriptions follow the same rule in `ext/MPIExt/runtests.jl`'s test table.

Fourier resolutions in validated rectangular fixtures are odd by contract. The two raw even-length Nyquist regressions use `unchecked_rectangular_test_size`, which retains the concrete production type and FDGrids operators while changing only its size type parameter. This narrow bypass must not be used for ordinary case or implementation tests.

## Migration provenance

Every active regression from the former flow packages has an explicit home:

- ReSolver-ChannelFlow's grid, derivative, shift, norm, weighting, and Cartesian-primitive files map to `cases/channel/`; its hot-loop assertions map to `performance/allocations.jl`, and forcing behavior maps to `cases/forcings.jl`.
- Resolver-LidDrivenCavity's suite maps to `cases/lid_driven_cavity.jl`.
- ReSolver-RPCF's suite maps to `cases/rpcf.jl`.
- ReSolver-RayleighBenard's suite maps to `cases/rayleigh_benard.jl`.
- ReSolver-SquareDuct's suite maps to `cases/square_duct.jl`.

Where several packages asserted the same finite-difference or adjoint identity, the implementation lives in `helpers/case_contracts.jl`, but each owning case invokes it independently. Legacy assertions that depended on a superseded package wrapper were translated to the corresponding `RectangularGrid` contract; for example, rectangular cavities now verify unequal resolutions instead of expecting their constructor to reject them.
