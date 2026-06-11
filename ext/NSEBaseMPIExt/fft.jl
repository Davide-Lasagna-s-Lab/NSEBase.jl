# ------------------------------------------------------------------ #
# FFT plans for HaloArray-backed fields                              #
# ------------------------------------------------------------------ #
# FFTs never act along halo-bearing dimensions. Plan against the full parent
# shape and execute on the full parent allocation: halo rows are transformed as
# independent batches along the homogeneous dimensions. This preserves FFTW's
# planned strides without copying through an interior view.

function NSEBase.FFTPlans(g::DecomposedGrid{T,D,AXES,FFT_DIMS_ORDER};
                          kwargs...) where {T,D,AXES,FFT_DIMS_ORDER}
    widths = nhalo(g)

    # Plan for the actual allocation strides, not merely the owned interior.
    # Halo rows are additional independent FFT batches along non-transformed
    # dimensions, so they must be present in the shape used to construct FFTW.
    parentsize = ntuple(dim -> size(g, dim) + 2 * widths[dim], Val(D))
    return NSEBase.FFTPlans(parentsize, FFT_DIMS_ORDER, T; kwargs...)
end

# NSEBase's field-level FFT methods call `_fft_data(parent(field))`. For a
# HaloArray, the FFTW-compatible storage is the halo-inclusive dense parent.
NSEBase._fft_data(a::HaloArrays.HaloArray) = parent(a)
