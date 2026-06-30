# ------------------------------------------------------------------ #
# GPUGrid - Generic wrapper for grid stored on device                #
# ------------------------------------------------------------------ #
# 
# `GPUGrid` is a utility type that wraps an `AbstractGrid` which has already
# been moved onto device via `Adapt.adapt_structure`. This special type
# facilitates simpler dispatch for fields that use this grid such that CUDA
# kernels are used instead of the standard methods.
# 
#   ChannelGrid              - host-side grid defined by the user
#   CUDA.cu(g)               - returns a `GPUGrid` wrapping the parent `g`
# 
# ! these methods need to be defined by the user:
# !     - Adapt.adapt_structure(to, g::AbstractGrid)

"""
    GPUGrid{T, D, AXES, FFT_DIMS_ORDER, GP} <:
        AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}

Generic wrapper for an instance of an [`AbstractGrid`](@ref) that lives on
the device.

This wrapper is very lightweight, only defining the necessary parts of the
[`AbstractGrid`](@ref) interface for correct dispatch in hot loops. Otherwise,
this wrapper type allows the simpler definition of specialised constructors
and methods for the various fields, plans, and operators.
"""
struct GPUGrid{T,
               D,
               AXES,
               FFT_DIMS_ORDER,
               GP<:NSEBase.AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}
              } <: NSEBase.AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}
    parent::GP

    GPUGrid(g::GP) where {T, D, AXES, FFT_DIMS_ORDER, GP<:NSEBase.AbstractGrid{T, D, AXES, FFT_DIMS_ORDER}} =
        new{T, D, AXES, FFT_DIMS_ORDER, GP}(g)
end

CUDA.cu(g::NSEBase.AbstractGrid) = GPUGrid(adapt_structure(CuArray{Float32}, convert(Float32, g)))
Adapt.adapt_structure(to, g::NSEBase.AbstractGrid) = throw(NSEBase.NotImplementedError(g))
# ! could I do Adapt.@adapt_structure(ChannelFlow{S, T} where {S, T}) in the child package?

"""
    parent(g::GPUGrid) -> NSEBase.AbstractGrid

Return the underlying grid wrapped by `g`.
"""
Base.parent(g::GPUGrid) = g.parent


Base.size(g::GPUGrid) = size(parent(g))
NSEBase.weights(g::GPUGrid) = NSEBase.weights(parent(g))
NSEBase.wavenumber_scale(g::GPUGrid, dim::Int) = NSEBase.wavenumber_scale(parent(g), dim)

# ! is it a good idea to define constant aliases to fields that live on the device for dispatch?
