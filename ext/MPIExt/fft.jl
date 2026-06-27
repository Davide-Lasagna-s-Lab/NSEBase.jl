# FFT hooks for HaloArray-backed fields.
#
# FFTs never act along halo-bearing dimensions. Plans must nevertheless use the
# full parent shape because halo rows are independent batches along the
# homogeneous dimensions, and FFTW's planned strides must match execution.
# We therefore pass the entire halo-inclusive dense parent array to FFTW, not a
# view of the local interior. An interior view would have different strides and
# would not match the plan constructed for the actual storage buffer.

NSEBase._fft_size(g::DecomposedGrid) = size(g) .+ 2 .* nhalo(g)

# NSEBase's field-level FFT methods call `_fft_data(parent(field))`. For a
# HaloArray, the FFTW-compatible storage is the halo-inclusive dense parent.
NSEBase._fft_data(a::HaloArrays.HaloArray) = parent(a)


"""
    (f::FFTPlans)(û::DecomposedFTField,
                  u::DecomposedFTField;
                             kwargs...) -> DecomposedFTField

TODO: this docstring
"""
function (f::FFTPlans)(û::DecomposedFTField,
                       u::DecomposedFTField;
                       kwargs...)
    _ensure_complete!(NSEBase.grid(u))
    f(û, u; kwargs...)
end


"""
    _ensure_complete!(g::AbstractGrid)           -> nothing

TODO: this docstring
"""
function _ensure_complete!(g::AbstractGrid)
    for t in g.boundary_tasks
        istaskdone(t) || wait(t)
    end
end
