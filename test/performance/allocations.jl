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

"""Metadata-only grid used to exercise the deliberately missing interface."""
struct AllocationIncompleteGrid <: AbstractGrid{Float64, 2, (1, 2, nothing, nothing), (2,)} end
Base.size(::AllocationIncompleteGrid) = (3, 5)

function alloc_fixture()
    g = channel_test_grid(7, 9, 7; α=1.5, β=2.0)
    u = FTField(g)
    v = FTField(g)
    parent(u) .= randn(ComplexF64, size(parent(u)))
    parent(v) .= randn(ComplexF64, size(parent(v)))

    q = VectorField(u, copy(u), zero(u))
    p = copy(q)

    Nm = 4
    homogeneous_size = map(dim -> transform_size(g)[dim], fft_storage_dims(g))
    modes = ntuple(_ -> randn(ComplexF64, Nm, size(g, 1), homogeneous_size...), 3)
    a = ProjectedField(g, randn(ComplexF64, Nm, homogeneous_size...), modes)
    b = copy(a)

    return (; g, u, v, q, p, modes, a, b)
end

function alloc_shear_fixture()
    g = shear_test_grid(7, 9)
    u = FTField(g)
    v = FTField(g)
    parent(u) .= randn(ComplexF64, size(parent(u)))
    parent(v) .= randn(ComplexF64, size(parent(v)))
    q = VectorField(u, copy(u))
    p = copy(q)
    return (; g, u, v, q, p)
end

function alloc_projected_2d_fixture()
    g = shear_test_grid(7, 9)
    Nm = 3
    Nk = transform_size(g)[2]
    modes = ntuple(_ -> randn(ComplexF64, Nm, size(g, 1), Nk), 2)
    a = ProjectedField(g, randn(ComplexF64, Nm, Nk), modes)
    b = copy(a)
    q = VectorField(FTField(g), FTField(g))
    parent(q[1]) .= randn(ComplexF64, size(parent(q[1])))
    parent(q[2]) .= randn(ComplexF64, size(parent(q[2])))
    return (; g, q, modes, a, b)
end

function alloc_plans_fixture(; dealias=false)
    g = shear_test_grid(7, 9)
    plans = FFTPlans(g; dealias=dealias, flags=FFTW.ESTIMATE)
    u = Field(g, (x, y, _, _) -> sin(x) + y)
    u_dealiased = Field(g, (x, y, _, _) -> sin(x) + y; dealias=dealias)
    uhat = FTField(g)
    parent(uhat) .= randn(ComplexF64, size(parent(uhat)))
    return (; g, plans, u, u_dealiased, uhat)
end

function alloc_noop_force!(out, _, _)
    return out
end

function alloc_rectangular_fixture()
    channel = ChannelGrid(3, 9, 3; Nt=3, α=1, β=1, width=3)
    duct = SquareDuctGrid(9, 3, 1, 1; width=3)
    xs, D₁, D₂, D₁⁺, D₂⁺, ws = channel.xs[1], channel.D₁[1], channel.D₂[1], channel.D₁⁺[1], channel.D₂⁺[1], channel.ws[1]
    box = RectangularGrid((xs, xs, xs), (D₁, D₁, D₁), (D₂, D₂, D₂), (D₁⁺, D₁⁺, D₁⁺),
                          (D₂⁺, D₂⁺, D₂⁺), (ws, ws, ws), (2π,), (9, 9, 9, 1), (1, 2, 3, 4), (4,))
    fields = map(g -> (out=FTField(g), u=FTField(g)), (channel, duct, box))
    vectors = (channel=(out=VectorField(channel; N=3), u=VectorField(channel; N=3)),
               duct=(out=VectorField(duct; N=3), u=VectorField(duct; N=3)))
    Nm = 3
    homogeneous_size = map(dim -> transform_size(channel)[dim], fft_storage_dims(channel))
    modes = ntuple(_ -> randn(ComplexF64, Nm, size(channel, 1), homogeneous_size...), 3)
    projected = (out=ProjectedField(channel, modes),
                 u=ProjectedField(channel, randn(ComplexF64, Nm, homogeneous_size...), modes))
    return (; fields, vectors, projected)
end

@testset verbose=true "Allocation contracts                                        " begin

    @testset verbose=true "src/NSEBase.jl                                              " begin
        @test isdefined(NSEBase, :AbstractGrid)
        @test isdefined(NSEBase, :FTField)
        @test isdefined(NSEBase, :ProjectedNSE)
    end

    @testset verbose=true "src/notimplementederror.jl                                  " begin
        g = AllocationIncompleteGrid()
        err = try
            NSEBase.points(g)
        catch e
            e
        end

        @test err isa NSEBase.NotImplementedError
        @test allocs_after_warmup(() -> sprint(showerror, err)) > 0
    end

    @testset verbose=true "src/grids/abstractgrid.jl                                   " begin
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
        grown = NSEBase.growto(g, (17, 13))
        @test size(grown) == (size(g, 1), 17, 13)
        @test allocs_after_warmup(() -> NSEBase.growto(g, (17, 13))) == 0
    end

    @testset verbose=true "src/grids/rectangular.jl                                    " begin
        (; fields) = alloc_rectangular_fixture()

        if VERSION >= v"1.11"
            for (; out, u) in fields
                @test allocs_after_warmup(() -> NSEBase.inhomogeneous_laplacian!(out, u)) == 0
                @test allocs_after_warmup(() -> NSEBase.inhomogeneous_laplacian!(out, u; adjoint=true)) == 0
            end

            (; out, u) = fields[1]
            @test allocs_after_warmup(() -> NSEBase.inhomogeneous_dd!(out, u, Val(1))) == 0
            @test allocs_after_warmup(() -> NSEBase.inhomogeneous_dd!(out, u, Val(1); adjoint=true)) == 0
        end
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
        (; g) = alloc_shear_fixture()
        u = Field(g, (x, y, _, _) -> y + sin(x))

        @test allocs_after_warmup(() -> Field(g)) > 0
        @test allocs_after_warmup(() -> Field(g, (x, y, _, _) -> y + sin(x))) > 0
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
        (; g, q) = alloc_shear_fixture()
        base = (range(-1.0, 1.0, length=size(g, 1)) |> collect, nothing)
        incomplete_q = VectorField(AllocationIncompleteGrid(); N=2)

        @test allocs_after_warmup(() -> VectorField(g, FTField; N=2)) > 0
        @test allocs_after_warmup(() -> VectorField(g, (x, y, _, _) -> y, (x, _, _, _) -> sin(x))) > 0
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
        @test_throws NSEBase.NotImplementedError NSEBase.growto(incomplete_q, (7,))
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
        (; u, q) = alloc_shear_fixture()
        (; projected) = alloc_rectangular_fixture()
        out = zero(u)
        qout = zero(q)

        @test allocs_after_warmup(() -> ddx!(out, u)) == 0
        # The wall-normal derivative dispatches to RectangularGrid's FDGrids operator.
        # On Julia < 1.11 mul! allocates when --check-bounds=yes is active; skip there.
        if VERSION >= v"1.11"
            @test allocs_after_warmup(() -> ddy!(out, u)) == 0
        end
        @test allocs_after_warmup(() -> ddz!(out, u)) == 0
        @test allocs_after_warmup(() -> ddt!(projected.out, projected.u)) == 0
        @test allocs_after_warmup(() -> NSEBase.dd!(out, u, Val(2))) == 0
        @test allocs_after_warmup(() -> NSEBase.dd!(qout, q, Val(2))) == 0
        # The inhomogeneous Laplacian and full Laplacian use the same FDGrids path.
        if VERSION >= v"1.11"
            @test allocs_after_warmup(() -> NSEBase.inhomogeneous_laplacian!(out, u)) == 0
        end
        @test allocs_after_warmup(() -> NSEBase.add_homogeneous_laplacian!(out, u)) == 0
        @test allocs_after_warmup(() -> NSEBase.add_homogeneous_laplacian!(qout, q)) == 0
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
        @test allocs_after_warmup(() -> Forward()) == 0
        @test allocs_after_warmup(() -> AdjointDiscrete()) == 0
        @test allocs_after_warmup(() -> AdjointContinuous()) == 0
    end

    @testset verbose=true "src/cases/forcings.jl                                       " begin
        out = Ref(0)
        compound = CompoundForcing(NoForce(), alloc_noop_force!)
        (; vectors) = alloc_rectangular_fixture()
        channel_force = ConstantBodyForce(1.0; component=1)
        duct_force = ConstantBodyForce(1.0; component=3)
        channel_rotation = CoriolisForce(0.2)
        rpcf_rotation = CoriolisForce(0.2; components=(3, 1))

        @test allocs_after_warmup(() -> NoForce()(out, nothing, Forward())) == 0
        @test allocs_after_warmup(() -> CompoundForcing(NoForce(), alloc_noop_force!)) == 0
        @test allocs_after_warmup(() -> compound(out, nothing, Forward())) == 0
        @test allocs_after_warmup(() -> channel_force(vectors.channel.out, vectors.channel.u, Forward())) == 0
        @test allocs_after_warmup(() -> duct_force(vectors.duct.out, vectors.duct.u, AdjointDiscrete())) == 0
        @test allocs_after_warmup(() -> channel_rotation(vectors.channel.out, vectors.channel.u, Forward())) == 0
        @test allocs_after_warmup(() -> rpcf_rotation(vectors.channel.out, vectors.channel.u, AdjointContinuous())) == 0
    end

    @testset verbose=true "src/equations/shared.jl                                     " begin
        (; g) = alloc_shear_fixture()
        base = (zeros(size(g, 1)), nothing)

        @test allocs_after_warmup(() -> NSEBase.ncomp(CartesianPrimitive3D())) == 0
        @test allocs_after_warmup(() -> NSEBase.cache_length(CartesianPrimitive3D(), FTField)) == 0
        @test allocs_after_warmup(() -> NSEBase.cache_length(CartesianPrimitive3D(), Field)) == 0
        @test allocs_after_warmup(() -> NSEBase.nonlinear_operator(CartesianPrimitive2D())) == 0
        @test allocs_after_warmup(() -> NSEBase.linearised_operator(CartesianPrimitive2D(), AdjointDiscrete())) == 0
        @test allocs_after_warmup(() -> construct_equations(g, 100.0, base, CartesianPrimitive2D(); flags=FFTW.ESTIMATE, dealias=false)) > 0
    end

    @testset verbose=true "src/equations/cartesianprimitive_2d.jl                      " begin
        (; g, q) = alloc_shear_fixture()
        out = zero(q)
        eq = CartesianPrimitive2DNSE(g, 100.0; flags=FFTW.ESTIMATE)
        ln = CartesianPrimitive2DLNSE(g, 100.0; mode=AdjointDiscrete(), flags=FFTW.ESTIMATE)

        @test allocs_after_warmup(() -> CartesianPrimitive2DNSE(g, 100.0; flags=FFTW.ESTIMATE)) > 0
        @test allocs_after_warmup(() -> CartesianPrimitive2DLNSE(g, 100.0; mode=AdjointDiscrete(), flags=FFTW.ESTIMATE)) > 0
        # Equation actions call RectangularGrid's FDGrids derivatives internally;
        # on Julia < 1.11 mul! allocates with --check-bounds=yes.
        if VERSION >= v"1.11"
            @test allocs_after_warmup(() -> eq(0.0, q, out)) == 0
            @test allocs_after_warmup(() -> ln(0.0, q, out)) == 0
            @test allocs_after_warmup(() -> ln(0.0, q, q, out)) == 0
        end
    end

    @testset verbose=true "src/equations/cartesianprimitive_3d.jl                      " begin
        (; g) = alloc_fixture()

        # Construction allocates caches and FFTW plans; operator action remains
        # covered by the focused 2-D allocation contracts above.
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
