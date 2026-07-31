# Allocation contracts, organised to mirror `src/`.
#
# The rule of thumb in this file is:
#   - functions ending in `!`, cheap accessors, and hot scalar/loop utilities
#     should allocate zero bytes after warmup;
#   - constructors and explicitly allocating APIs (`copy`, `zero`, `FFT`,
#     `project`, `save_*`, ...) are smoke-tested as allocating operations.
#
# All measurements are taken inside helper functions. Top-level `@allocated`
# can measure allocations from global captures in the test itself, which is not
# the package behaviour we want to pin down.

function allocs_after_warmup(f)
    f()
    f()
    return @allocated f()
end

function alloc_fixture()
    g = steady_channel_grid(; Nx=9, Ny=9, Nz=7, α=1.5, β=2.0)
    u = test_ftfield(g; seed=21)
    v = test_ftfield(g; seed=22)

    q = VectorField(u, copy(u), zero(u))
    p = copy(q)

    modes = test_modes(g; ncomponents=3, nmodes=4, seed=23)
    coefficient_size = (size(modes[1], 1), map(dim -> transform_size(g)[dim], fft_storage_dims(g))...)
    a = ProjectedField(g, randn(MersenneTwister(24), ComplexF64, coefficient_size), modes)
    b = copy(a)

    return (; g, u, v, q, p, modes, a, b)
end

function alloc_planar_fixture()
    g = planar_grid(; Nb=9, Nh=9)
    u = test_ftfield(g; seed=31)
    v = test_ftfield(g; seed=32)
    q = VectorField(u, copy(u))
    p = copy(q)
    return (; g, u, v, q, p)
end

function alloc_projected_2d_fixture()
    g = planar_grid(; Nb=9, Nh=9)
    modes = test_modes(g; ncomponents=2, nmodes=3, seed=41)
    coefficient_size = (size(modes[1], 1), map(dim -> transform_size(g)[dim], fft_storage_dims(g))...)
    a = ProjectedField(g, randn(MersenneTwister(42), ComplexF64, coefficient_size), modes)
    b = copy(a)
    q = VectorField(FTField(g), FTField(g))
    parent(q[1]) .= parent(test_ftfield(g; seed=43))
    parent(q[2]) .= parent(test_ftfield(g; seed=44))
    return (; g, q, modes, a, b)
end

function alloc_plans_fixture(; dealias=false)
    g = planar_grid(; Nb=9, Nh=9)
    plans = FFTPlans(g; dealias=dealias, flags=FFTW.ESTIMATE)
    u = Field(g, (x, y) -> x + sin(y))
    u_dealiased = Field(g, (x, y) -> x + sin(y); dealias=dealias)
    uhat = test_ftfield(g; seed=51)
    return (; g, plans, u, u_dealiased, uhat)
end

alloc_unimplemented_error(g) = NSEBase.NotImplementedError(g)

function alloc_noop_force!(out, _, _)
    return out
end

@testset verbose=true "Allocation contracts                                        " begin

    @testset verbose=true "src/NSEBase.jl                                              " begin
        @test isdefined(NSEBase, :AbstractGrid)
        @test isdefined(NSEBase, :FTField)
        @test isdefined(NSEBase, :ProjectedNSE)
    end

    @testset verbose=true "src/notimplementederror.jl                                  " begin
        err = alloc_unimplemented_error(line_grid())

        @test err isa NSEBase.NotImplementedError
        @test allocs_after_warmup(() -> sprint(showerror, err)) > 0
    end

    @testset verbose=true "src/abstractgrid.jl                                         " begin
        (; g) = alloc_fixture()
        values = (:x, :y, :z, :t)

        @test allocs_after_warmup(() -> NSEBase.fft_storage_dims(g)) == 0
        @test allocs_after_warmup(() -> NSEBase.inhomogeneous_storage_dims(g)) == 0
        @test allocs_after_warmup(() -> NSEBase.to_storage_order(values, g)) == 0
        @test allocs_after_warmup(() -> size(g)) == 0
        @test allocs_after_warmup(() -> size(g, 2)) == 0
        @test allocs_after_warmup(() -> NSEBase.transform_size(g)) == 0
        @test allocs_after_warmup(() -> NSEBase.fft_norm(g)) == 0
        @test allocs_after_warmup(() -> NSEBase.weights(g)) == 0
        @test allocs_after_warmup(() -> NSEBase.wavenumber_scale(g, 2)) == 0
        @test allocs_after_warmup(() -> convert(Float64, g)) == 0

        # Coordinate arrays are newly constructed by this fixture.
        @test allocs_after_warmup(() -> NSEBase.points(g)) > 0
        grown = NSEBase.growto(g, (11, 9))
        @test size(grown) == (size(g, 1), 11, 9)
        @test allocs_after_warmup(() -> NSEBase.growto(g, (11, 9))) > 0
    end

    @testset verbose=true "src/wavenumbervector.jl                                     " begin
        (; g) = alloc_fixture()
        k = WaveNumberVector(1, -2)
        storage_indices = (2, 6)

        @test allocs_after_warmup(() -> WaveNumberVector(1, -2)) == 0
        @test allocs_after_warmup(() -> length(k)) == 0
        @test allocs_after_warmup(() -> k[2]) == 0
        @test allocs_after_warmup(() -> NSEBase._fftw_index(-1, 6)) == 0
        @test allocs_after_warmup(() -> NSEBase._fftw_sym_index(3, 6)) == 0
        @test allocs_after_warmup(() -> NSEBase.to_homogeneous_indices(g, k)) == 0
        @test allocs_after_warmup(() -> NSEBase.to_wavenumber_vector(g, storage_indices)) == 0
    end

    @testset verbose=true "src/ftfield.jl                                              " begin
        (; g, u) = alloc_fixture()
        k = WaveNumberVector(1, -1)

        @test allocs_after_warmup(() -> FTField(g)) > 0
        @test allocs_after_warmup(() -> FTField(g, randn(ComplexF64, size(parent(u))))) > 0
        @test allocs_after_warmup(() -> parent(u)) == 0
        @test allocs_after_warmup(() -> grid(u)) == 0
        @test allocs_after_warmup(() -> size(u)) == 0
        @test allocs_after_warmup(() -> eltype(u)) == 0
        @test allocs_after_warmup(() -> u[1]) == 0
        @test allocs_after_warmup(() -> (u[1] = 1 + 0im)) == 0
        @test allocs_after_warmup(() -> NSEBase.combine_indices(g, CartesianIndex(2), CartesianIndex(3, 4))) == 0
        @test allocs_after_warmup(() -> NSEBase._average_complex(1 + 2im, 3 + 4im)) == 0
        @test allocs_after_warmup(() -> (u[k, 2] = 2 - 1im)) == 0
        @test allocs_after_warmup(() -> u[k, 2]) == 0

        data = randn(ComplexF64, size(parent(u)))
        @test allocs_after_warmup(() -> NSEBase.apply_symmetry!(data, Val(NSEBase.fft_storage_dims(g)))) == 0
        @test allocs_after_warmup(() -> NSEBase.normalise_mean!(data, Val(NSEBase.fft_storage_dims(g)))) == 0

        @test allocs_after_warmup(() -> similar(u)) > 0
        @test allocs_after_warmup(() -> copy(u)) > 0
        @test allocs_after_warmup(() -> zero(u)) > 0
        @test allocs_after_warmup(() -> u[k]) == 0
    end

    @testset verbose=true "src/field.jl                                                " begin
        (; g) = alloc_planar_fixture()
        u = Field(g, (x, y) -> x + sin(y))

        @test allocs_after_warmup(() -> Field(g)) > 0
        @test allocs_after_warmup(() -> Field(g, (x, y) -> x + sin(y))) > 0
        @test allocs_after_warmup(() -> Field(g, parent(u))) == 0
        @test allocs_after_warmup(() -> parent(u)) == 0
        @test allocs_after_warmup(() -> grid(u)) == 0
        @test allocs_after_warmup(() -> size(u)) == 0
        @test allocs_after_warmup(() -> eltype(u)) == 0
        @test allocs_after_warmup(() -> u[1]) == 0
        @test allocs_after_warmup(() -> (u[1] = 0.5)) == 0
        @test allocs_after_warmup(() -> similar(u)) > 0
        @test allocs_after_warmup(() -> copy(u)) > 0
        @test allocs_after_warmup(() -> zero(u)) > 0
    end

    @testset verbose=true "src/vectorfield.jl                                          " begin
        (; g, q) = alloc_planar_fixture()
        base = (range(-1.0, 1.0, length=size(g, 1)) |> collect, nothing)

        @test allocs_after_warmup(() -> VectorField(g, FTField; N=2)) > 0
        @test allocs_after_warmup(() -> VectorField(g, (x, y) -> x, (x, y) -> sin(y))) > 0
        @test allocs_after_warmup(() -> parent(q)) == 0
        @test allocs_after_warmup(() -> grid(q)) == 0
        @test allocs_after_warmup(() -> q[1]) == 0
        @test allocs_after_warmup(() -> size(q)) == 0
        @test allocs_after_warmup(() -> eltype(q)) == 0
        @test allocs_after_warmup(() -> (q[1] = q[2])) == 0
        @test allocs_after_warmup(() -> add_base_flow!(q, base)) == 0
        @test allocs_after_warmup(() -> similar(q)) > 0
        @test allocs_after_warmup(() -> copy(q)) > 0
        @test allocs_after_warmup(() -> zero(q)) > 0
        @test allocs_after_warmup(() -> NSEBase.growto(q, (11,))) > 0
    end

    @testset verbose=true "src/fft.jl                                                  " begin
        (; plans, u, u_dealiased, uhat) = alloc_plans_fixture(dealias=false)
        dealiased = alloc_plans_fixture(dealias=true)
        dealias_plans = dealiased.plans
        physical_dealiased = dealiased.u_dealiased
        uhat_dealiased = dealiased.uhat
        cache = similar(dealias_plans.cache)
        compact = similar(parent(uhat))

        @test allocs_after_warmup(() -> FFTPlans(size(u), NSEBase.fft_storage_dims(grid(u)), Float64; dealias=false, flags=FFTW.ESTIMATE)) > 0
        @test allocs_after_warmup(() -> NSEBase.get_padded_size((5, 8), (2,))) == 0
        @test allocs_after_warmup(() -> NSEBase._get_transform_size((5, 8), 2)) == 0
        @test allocs_after_warmup(() -> NSEBase._loopblk!(compact, axes(compact), compact, axes(compact), Val(false))) == 0
        @test allocs_after_warmup(() -> NSEBase._apply_mask!(cache)) == 0
        @test allocs_after_warmup(() -> NSEBase._copy_to_padded!(cache, compact, (2,))) == 0
        @test allocs_after_warmup(() -> NSEBase._copy_from_padded!(compact, cache, (2,))) == 0
        @test allocs_after_warmup(() -> NSEBase._add_from_padded!(compact, cache, (2,))) == 0
        @test allocs_after_warmup(() -> plans(uhat, u; add=false, use_cache=false)) == 0
        @test allocs_after_warmup(() -> plans(parent(uhat), parent(u), false, false)) == 0
        @test allocs_after_warmup(() -> dealias_plans(uhat_dealiased, physical_dealiased; add=false)) == 0
        @test allocs_after_warmup(() -> plans(u, uhat; preserve_input=false, use_cache=false)) == 0
        @test allocs_after_warmup(() -> plans(parent(u), parent(uhat), false, false)) == 0
        @test allocs_after_warmup(() -> FFT(u)) > 0
        @test allocs_after_warmup(() -> IFFT(uhat)) > 0
    end

    @testset verbose=true "src/projectedfield.jl                                       " begin
        (; g, modes, a) = alloc_fixture()
        k = WaveNumberVector(1, -1)

        @test allocs_after_warmup(() -> ProjectedField(g, modes)) > 0
        @test allocs_after_warmup(() -> ProjectedField(g, parent(a), modes)) > 0
        @test allocs_after_warmup(() -> ProjectedField(g, modes[1])) > 0
        @test allocs_after_warmup(() -> ProjectedField(FTField(g), modes)) > 0
        @test allocs_after_warmup(() -> parent(a)) == 0
        @test allocs_after_warmup(() -> grid(a)) == 0
        @test allocs_after_warmup(() -> NSEBase.modes(a)) == 0
        @test allocs_after_warmup(() -> size(a)) == 0
        @test allocs_after_warmup(() -> eltype(a)) == 0
        @test allocs_after_warmup(() -> a[1]) == 0
        @test allocs_after_warmup(() -> (a[1] = 0.5 + 0im)) == 0
        @test allocs_after_warmup(() -> a[1, 2, 3]) == 0
        @test allocs_after_warmup(() -> (a[1, 2, 3] = 0.25 - 0.5im)) == 0
        @test allocs_after_warmup(() -> a[1, k]) == 0
        @test allocs_after_warmup(() -> (a[1, k] = 1 + 2im)) == 0
        @test allocs_after_warmup(() -> similar(a)) > 0
        @test allocs_after_warmup(() -> copy(a)) > 0
        @test allocs_after_warmup(() -> zero(a)) > 0
        @test allocs_after_warmup(() -> abs(a)) > 0
    end

    @testset verbose=true "src/galerkin.jl                                             " begin
        (; q, modes, a) = alloc_projected_2d_fixture()
        out = zero(q)

        @test allocs_after_warmup(() -> LoopGalerkin()) == 0
        @test allocs_after_warmup(() -> GemmGalerkin()) == 0
        @test allocs_after_warmup(() -> project!(a, q, LoopGalerkin())) == 0
        @test allocs_after_warmup(() -> expand!(out, a, LoopGalerkin())) == 0
        @test allocs_after_warmup(() -> project(q, modes, LoopGalerkin())) > 0
        a3 = alloc_fixture().a
        @test allocs_after_warmup(() -> expand(a3, LoopGalerkin())) > 0
        @test allocs_after_warmup(() -> project!(a, q, GemmGalerkin())) >= 0
        @test allocs_after_warmup(() -> expand!(out, a, GemmGalerkin())) >= 0
    end

    @testset verbose=true "src/shifts.jl                                               " begin
        (; u, q, a) = alloc_fixture()

        @test allocs_after_warmup(() -> shift!(u, (0.13, -0.21))) == 0
        @test allocs_after_warmup(() -> shift!(q, (0.13, -0.21))) == 0
        @test allocs_after_warmup(() -> shift!(a, (0.13, -0.21))) == 0
        @test allocs_after_warmup(() -> shift(u, (0.13, -0.21))) > 0
        @test allocs_after_warmup(() -> NSEBase._shift_phase(grid(u), (0.13, -0.21), WaveNumberVector(1, -1))) == 0
    end

    @testset verbose=true "src/norms.jl                                                " begin
        (; u, v, q, p, a, b) = alloc_fixture()
        tmp_ft = zero(v)
        tmp_vec1 = zero(p)
        tmp_a = zero(b)

        # These routines use one Ref accumulator internally.  BenchmarkTools
        # sees this as one 16-byte allocation; through `allocs_after_warmup`
        # the closure wrapper reports 32 bytes.
        small_accumulator_alloc = 32
        @test allocs_after_warmup(() -> dot(u, v)) <= small_accumulator_alloc
        @test allocs_after_warmup(() -> norm(u)) <= small_accumulator_alloc
        @test allocs_after_warmup(() -> normdiff(u, v)) <= small_accumulator_alloc
        @test allocs_after_warmup(() -> normdiff(u, v, (0.13, -0.21), tmp_ft)) <= small_accumulator_alloc
        @test allocs_after_warmup(() -> dot(q, p)) <= small_accumulator_alloc
        @test allocs_after_warmup(() -> norm(q)) <= small_accumulator_alloc
        @test allocs_after_warmup(() -> normdiff(q, p)) <= small_accumulator_alloc
        @test allocs_after_warmup(() -> normdiff(q, p, (0.13, -0.21), tmp_ft)) <= small_accumulator_alloc
        @test allocs_after_warmup(() -> minnormdiff(q, p, (2, 2), tmp_vec1)) <= 48
        @test allocs_after_warmup(() -> dot(a, b)) <= small_accumulator_alloc
        @test allocs_after_warmup(() -> norm(a)) <= small_accumulator_alloc
        @test allocs_after_warmup(() -> normdiff(a, b)) <= small_accumulator_alloc
        @test allocs_after_warmup(() -> normdiff(a, b, (0.13, -0.21), tmp_a)) <= small_accumulator_alloc
    end

    @testset verbose=true "src/weighting.jl                                            " begin
        (; g, a, b) = alloc_fixture()
        A = FarazmandWeight(g)
        k = WaveNumberVector(1, -1)

        @test allocs_after_warmup(() -> FarazmandWeight(g)) == 0
        @test allocs_after_warmup(() -> FarazmandWeight(1.0, 2.0)) == 0
        @test allocs_after_warmup(() -> A[k]) == 0
        @test allocs_after_warmup(() -> lmul!(A, a)) == 0
        @test allocs_after_warmup(() -> dot(a, A, b)) <= 32
    end

    @testset verbose=true "src/broadcasting.jl                                         " begin
        (; u, v, q, p) = alloc_fixture()
        out = zero(u)
        qout = zero(q)

        @test allocs_after_warmup(() -> (out .= 2 .* u .- v)) == 0
        @test allocs_after_warmup(() -> (qout .= 2 .* q .- p)) == 0
        @test allocs_after_warmup(() -> (qout .= 0)) == 0
        @test allocs_after_warmup(() -> u .+ v) > 0
        @test allocs_after_warmup(() -> q .+ p) > 0
    end

    @testset verbose=true "src/derivatives.jl                                          " begin
        (; u, q) = alloc_planar_fixture()
        out = zero(u)
        qout = zero(q)

        # The finite-difference direction dispatches to FDGrids' `mul!`. On
        # Julia < 1.11 it allocates with `--check-bounds=yes`, so skip it there.
        if VERSION >= v"1.11"
            @test allocs_after_warmup(() -> ddx!(out, u)) == 0
        end
        @test allocs_after_warmup(() -> ddy!(out, u)) == 0
        @test allocs_after_warmup(() -> ddz!(out, u)) == 0
        @test allocs_after_warmup(() -> ddt!(out, u)) == 0
        @test allocs_after_warmup(() -> NSEBase.dd!(out, u, Val(2))) == 0
        @test allocs_after_warmup(() -> NSEBase.dd!(qout, q, Val(2))) == 0
        # The inhomogeneous Laplacian follows the same FDGrids path.
        if VERSION >= v"1.11"
            @test allocs_after_warmup(() -> NSEBase._inhomogeneous_laplacian!(out, u)) == 0
        end
        @test allocs_after_warmup(() -> NSEBase._add_homogeneous_laplacian!(out, u)) == 0
        @test allocs_after_warmup(() -> NSEBase._add_homogeneous_laplacian!(qout, q)) == 0
        if VERSION >= v"1.11"
            @test allocs_after_warmup(() -> NSEBase.laplacian!(out, u)) == 0
            @test allocs_after_warmup(() -> NSEBase.laplacian!(qout, q)) == 0
        end
    end

    @testset verbose=true "src/io.jl                                                   " begin
        (; g, a, modes) = alloc_fixture()
        mktempdir() do dir
            grid_path = joinpath(dir, "grid.jld2")
            field_path = joinpath(dir, "field.jld2")
            @test allocs_after_warmup(() -> save_grid(g; path=grid_path)) > 0
            @test allocs_after_warmup(() -> load_grid(grid_path)) > 0
            @test allocs_after_warmup(() -> save_field(a; path=field_path)) > 0
            @test allocs_after_warmup(() -> load_field(g, modes, field_path)) > 0
        end
    end

    @testset verbose=true "src/equations/types.jl                                      " begin
        out = Ref(0)
        cf = CompoundForcing(NoForce(), alloc_noop_force!)

        @test allocs_after_warmup(() -> Forward()) == 0
        @test allocs_after_warmup(() -> AdjointDiscrete()) == 0
        @test allocs_after_warmup(() -> AdjointContinuous()) == 0
        @test allocs_after_warmup(() -> NoForce()(out, nothing, Forward())) == 0
        @test allocs_after_warmup(() -> CompoundForcing(NoForce(), alloc_noop_force!)) == 0
        @test allocs_after_warmup(() -> cf(out, nothing, Forward())) == 0
    end

    @testset verbose=true "src/equations/shared.jl                                     " begin
        (; g) = alloc_planar_fixture()
        base = (zeros(size(g, 1)), nothing)

        @test allocs_after_warmup(() -> NSEBase.ncomp(CartesianPrimitive3D())) == 0
        @test allocs_after_warmup(() -> NSEBase.cache_length(CartesianPrimitive3D(), FTField)) == 0
        @test allocs_after_warmup(() -> NSEBase.cache_length(CartesianPrimitive3D(), Field)) == 0
        @test allocs_after_warmup(() -> NSEBase.nonlinear_operator(CartesianPrimitive2D())) == 0
        @test allocs_after_warmup(() -> NSEBase.linearised_operator(CartesianPrimitive2D(), AdjointDiscrete())) == 0
        @test allocs_after_warmup(() -> construct_equations(g, 100.0, base, CartesianPrimitive2D(); flags=FFTW.ESTIMATE, dealias=false)) > 0
    end

    @testset verbose=true "src/equations/cartesianprimitive_2d.jl                      " begin
        (; g, q) = alloc_planar_fixture()
        out = zero(q)
        eq = CartesianPrimitive2DNSE(g, 100.0; flags=FFTW.ESTIMATE)
        ln = CartesianPrimitive2DLNSE(g, 100.0; mode=AdjointDiscrete(), flags=FFTW.ESTIMATE)
        ln(0.0, q, q, out) # Populate the base-state cache required by the shorter call.

        @test allocs_after_warmup(() -> CartesianPrimitive2DNSE(g, 100.0; flags=FFTW.ESTIMATE)) > 0
        @test allocs_after_warmup(() -> CartesianPrimitive2DLNSE(g, 100.0; mode=AdjointDiscrete(), flags=FFTW.ESTIMATE)) > 0
        # Equation actions call FDGrids' `mul!`-based derivatives internally;
        # on Julia < 1.11 `mul!` allocates with `--check-bounds=yes`.
        if VERSION >= v"1.11"
            @test allocs_after_warmup(() -> eq(0.0, q, out)) == 0
            @test allocs_after_warmup(() -> ln(0.0, q, out)) == 0
            @test allocs_after_warmup(() -> ln(0.0, q, q, out)) == 0
        end
    end

    @testset verbose=true "src/equations/cartesianprimitive_3d.jl                      " begin
        (; g) = alloc_fixture()

        # Construction allocates production-grid caches and FFTW plans. Full
        # operator behavior is exercised in `test_operators.jl`.
        @test allocs_after_warmup(() -> CartesianPrimitive3DNSE(g, 100.0; flags=FFTW.ESTIMATE)) > 0
        @test allocs_after_warmup(() -> CartesianPrimitive3DLNSE(g, 100.0; mode=AdjointDiscrete(), flags=FFTW.ESTIMATE)) > 0
    end

    @testset verbose=true "src/equations/projectednse.jl                               " begin
        (; g, modes, a, b) = alloc_projected_2d_fixture()
        base = (zeros(size(g, 1)), nothing)
        eq = construct_equations(g, 100.0, base, CartesianPrimitive2D(); flags=FFTW.ESTIMATE, dealias=true)

        @test allocs_after_warmup(() -> ProjectedNSE(g, 2, eq.nl, eq.ln, base)) > 0
        if VERSION >= v"1.11"
            @test allocs_after_warmup(() -> eq(a, b)) == 0
            @test allocs_after_warmup(() -> eq(a, b, b)) == 0
        end
    end
end
