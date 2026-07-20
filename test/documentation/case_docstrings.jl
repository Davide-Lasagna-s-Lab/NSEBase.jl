@testset verbose=true "Migrated case docstrings                                    " begin
    # Keep Julia's `?name` help available for every public binding migrated
    # from the five former flow packages, including renamed shared APIs.
    migrated_bindings = (
        :STREAMWISE_INVARIANT_CHANNEL_AXES, :STREAMWISE_INVARIANT_CHANNEL_FFT_ORDER,
        :STREAMWISE_INVARIANT_CHANNEL_INHOMOGENEOUS_DIMS,
        :TWO_DIMENSIONAL_CHANNEL_AXES, :TWO_DIMENSIONAL_CHANNEL_FFT_ORDER,
        :TWO_DIMENSIONAL_CHANNEL_INHOMOGENEOUS_DIMS,
        :CHANNEL_3D_AXES, :CHANNEL_3D_FFT_ORDER, :CHANNEL_3D_INHOMOGENEOUS_DIMS,
        :SQUARE_DUCT_AXES, :SQUARE_DUCT_FFT_ORDER, :SQUARE_DUCT_INHOMOGENEOUS_DIMS,
        :LID_DRIVEN_CAVITY_2D_AXES, :LID_DRIVEN_CAVITY_2D_FFT_ORDER, :LID_DRIVEN_CAVITY_2D_INHOMOGENEOUS_DIMS,
        :LID_DRIVEN_CAVITY_3D_AXES, :LID_DRIVEN_CAVITY_3D_PERIODIC_FFT_ORDER, :LID_DRIVEN_CAVITY_3D_BOUNDED_FFT_ORDER,
        :AbstractChannelGrid, :AbstractStreamwiseInvariantChannelGrid,
        :AbstractTwoDimensionalChannelGrid, :AbstractChannel3DGrid,
        :AbstractSquareDuctGrid, :AbstractLidDrivenCavityGrid, :AbstractLidDrivenCavity3DGrid,
        :RectangularGrid, :ChannelGrid, :StreamwiseInvariantChannelGrid,
        :TwoDimensionalChannelGrid, :SquareDuctGrid, :LidDrivenCavityGrid,
        :points, :growto, :weights, :wavenumber_scale, :dd!,
        :inhomogeneous_dd!, :inhomogeneous_laplacian!, :derivative_matrix,
        :fft_storage_dims, :inhomogeneous_storage_dims, :spatial_inhomogeneous_physical_dims,
        :Mode, :Forward, :AdjointContinuous, :AdjointDiscrete,
        :CartesianPrimitive3D, :CartesianPrimitive3DNSE, :CartesianPrimitive3DLNSE,
        :CartesianPrimitive2D, :CartesianPrimitive2DNSE, :CartesianPrimitive2DLNSE,
        :CartesianPrimitive2D3C, :CartesianPrimitive2D3CNSE, :CartesianPrimitive2D3CLNSE,
        :CartesianPrimitive2DBoussinesq, :CartesianPrimitive2DBoussinesqNSE,
        :CartesianPrimitive2DBoussinesqLNSE, :CartesianPrimitive3DBoussinesq,
        :CartesianPrimitive3DBoussinesqNSE, :CartesianPrimitive3DBoussinesqLNSE,
        :PolarPrimitive, :ProjectedNSE, :construct_equations,
        :CoriolisForce, :ConstantBodyForce,
        :plane_couette_base, :plane_poiseuille_base,
        :rbc_base_temperature, :PlaneCouetteFlow, :PlanePoiseuilleFlow,
        :SquareDuctFlow, :LidDrivenCavityFlow, :RayleighBenardFlow,
        :NoForce,
    )

    for name in migrated_bindings
        @test Base.Docs.hasdoc(NSEBase, name)
    end

    # Binding-level documentation can survive after a documented overload is
    # removed. Check that the unified RectangularGrid methods themselves still
    # have entries in Julia's documentation metadata.
    function has_rectangular_method_doc(binding_module, name)
        binding = Base.Docs.Binding(binding_module, name)
        metadata = Base.Docs.meta(NSEBase)
        haskey(metadata, binding) || return false
        return any(signature -> occursin("RectangularGrid", string(signature)), metadata[binding].order)
    end

    for name in (:points, :growto, :weights, :wavenumber_scale,
                 :inhomogeneous_dd!, :inhomogeneous_laplacian!, :derivative_matrix)
        @test has_rectangular_method_doc(NSEBase, name)
    end
    @test has_rectangular_method_doc(Base, :size)
    @test has_rectangular_method_doc(Base, :convert)
end
