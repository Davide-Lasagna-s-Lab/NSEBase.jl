# Abstract interface to aid in the creation of fft plans using FFTW.jl, which
# are assumed to be quite useful for the definition and manipulation of a given
# field.


# ---------------- #
# useful constants #
# ---------------- #
const ESTIMATE = FFTW.ESTIMATE
const EXHAUSTIVE = FFTW.EXHAUSTIVE
const MEASURE = FFTW.MEASURE
const PATIENT = FFTW.PATIENT
const WISDOM_ONLY = FFTW.WISDOM_ONLY
const NO_TIMELIMIT = FFTW.NO_TIMELIMIT


# ---------------- #
# transform object #
# ---------------- #
struct FFTPlans{DIM, T, ORDER, DEALIAS, PLAN, IPLAN}
    plan::PLAN
    iplan::IPLAN
    cache::Array{Complex{T}, DIM}
    norm::T

    function FFTPlans(shape::Dims{DIM},
                      order::NTuple{H, Int},
                           ::Type{T}=Float64;
                    dealias::Bool   =true,
                      flags::UInt32 =EXHAUSTIVE,
                  timelimit::Real   =NO_TIMELIMIT) where {DIM, H, T}
        # create arrays
        shape = dealias ? _get_padded_shape(shape, order) : shape
        spectral_array = zeros(Complex{T}, _get_transform_shape(shape, order[1]))
        physical_array = zeros(T, shape)
        norm = T(1/prod(shape[collect(order)]))

        # construct plans
        plan  = FFTW.plan_rfft( physical_array,                  order, flags=flags, timelimit=timelimit)
        iplan = FFTW.plan_brfft(spectral_array, shape[order[1]], order, flags=flags, timelimit=timelimit)

        new{DIM, T, order, dealias, typeof(plan), typeof(iplan)}(plan, iplan, spectral_array, norm)
    end
end


# ------------------------ #
# in-place transformations #
# ------------------------ #
# physical -> spectral fields
function (f::FFTPlans{DIM, T})(û::VectorField{N, S},
                               u::VectorField{N, P},
                             add::Bool=false) where {DIM, T, N, S<:AbstractScalarField{DIM, Complex{T}}, P<:AbstractScalarField{DIM, T}}
    for n in 1:N
        f(û[n], u[n], add)
    end
    return û
end

(f::FFTPlans{DIM, T})(û::AbstractScalarField{DIM, Complex{T}},
                      u::AbstractScalarField{DIM,         T},
                    add::Bool=false,
              use_cache::Bool=false) where {DIM, T} = (f(parent(û), parent(u), add, use_cache); return û)

(f::FFTPlans{DIM, T, ORDER, DEALIAS})(û::AbstractArray{Complex{T}},
                                      u::AbstractArray{        T},
                                    add::Bool,
                              use_cache::Bool) where {DIM, T, ORDER, DEALIAS} = add ? _add_forward_transform!(û, u, f) : _forward_transform!(û, u, f, use_cache)

function _forward_transform!(û, u, f::FFTPlans{DIM, T, ORDER, DEALIAS}, use_cache::Bool) where {DIM, T, ORDER, DEALIAS}
    if DEALIAS
        FFTW.unsafe_execute!(f.plan, u, f.cache)
        _copy_from_padded!(û, f.cache, DIM, ORDER)
        û .*= f.norm
    elseif use_cache
        FFTW.unsafe_execute!(f.plan, u, f.cache)
        f.cache .*= f.norm
        û .= f.cache
    else
        FFTW.unsafe_execute!(f.plan, u, û)
        û .*= f.norm
    end
    return û
end

function _add_forward_transform!(û, u, f::FFTPlans{DIM, T, ORDER, DEALIAS}) where {DIM, T, ORDER, DEALIAS}
    FFTW.unsafe_execute!(f.plan, u, f.cache)
    f.cache .*= f.norm
    if DEALIAS
        _add_from_padded!(û, f.cache, DIM, ORDER)
    else
        û .+= f.cache
    end
    return û
end

# spectral -> physical fields
function (f::FFTPlans{DIM, T})(u::VectorField{N, P},
                               û::VectorField{N, S},
                            safe::Bool=true) where {DIM, T, N, S<:AbstractScalarField{DIM, Complex{T}}, P<:AbstractScalarField{DIM, T}}
    for n in 1:N
        f(u[n], û[n], safe)
    end
    return u
end

(f::FFTPlans{DIM, T})(u::AbstractScalarField{DIM,         T},
                      û::AbstractScalarField{DIM, Complex{T}},
                   safe::Bool=true,
              use_cache::Bool=false) where {DIM, T} = (f(parent(u), parent(û), safe, use_cache); return u)

(f::FFTPlans{DIM, T, ORDER, DEALIAS})(u::AbstractArray{        T},
                                      û::AbstractArray{Complex{T}},
                                   safe::Bool,
                              use_cache::Bool) where {DIM, T, ORDER, DEALIAS} = safe ? _backward_transform!(u, û, f) : _unsafe_backward_transform!(u, û, f, use_cache)

function _backward_transform!(u, û, f::FFTPlans{DIM, T, ORDER, DEALIAS}) where {DIM, T, ORDER, DEALIAS}
    if DEALIAS
        _copy_to_padded!(_apply_mask!(f.cache), û, DIM, ORDER)
        FFTW.unsafe_execute!(f.iplan, f.cache, u)
    else
        f.cache .= û
        FFTW.unsafe_execute!(f.iplan, f.cache, u)
    end
    return u
end

function _unsafe_backward_transform!(u, û, f::FFTPlans{DIM, T, ORDER, DEALIAS}, use_cache::Bool) where {DIM, T, ORDER, DEALIAS}
    if DEALIAS
        _backward_transform!(u, û, f)
    elseif use_cache
        f.cache .= û
        FFTW.unsafe_execute!(f.iplan, f.cache, u)
    else
        FFTW.unsafe_execute!(f.iplan, û, u)
    end
    return u
end


# ----------------- #
# utility functions #
# ----------------- #
function _get_padded_shape(shape, order)
    return ntuple(length(shape)) do i
        i ∈ order ? cld(3*shape[i], 2) | 1 : shape[i]
    end
end

function _get_transform_shape(shape, dim)
    return ntuple(length(shape)) do i
        i == dim ? (shape[i] >> 1) + 1 : shape[i]
    end
end

# copy function for dealiasing
for name in [:_copy_from_padded!, :_add_from_padded!, :_copy_to_padded!]
    # get useful variables for evaluation
    src, dst = name ∈ [:_copy_from_padded!, :_add_from_padded!] ? (:cache, :u) : (:u, :cache)
    blk_src, blk_dst = name ∈ [:_copy_from_padded!, :_add_from_padded!] ? (:blk_c, :blk_u) : (:blk_u, :blk_c)
    op = name ∈ [:_copy_from_padded!, :_copy_to_padded!] ? ((x, y)->x) : +

    # create functions
    @eval begin
        function $name($dst, $src, dim, ::NTuple{1})
            # copy non-negative frequencies
            blk = ntuple(i->1:size(u, i), dim)
            broadcast!($op, @view($dst[blk...]), @view($src[blk...]), @view($dst[blk...]))
            return $dst
        end

        function $name($dst, $src, dim, order::NTuple{2})
            # copy non-negative frequencies in both directions
            blk = ntuple(i->i == order[2] ? (1:(size(u, i) >> 1) + 1) : (1:size(u, i)), dim)
            broadcast!($op, @view($dst[blk...]), @view($src[blk...]), @view($dst[blk...]))

            # get positive and negative frequency lengths
            npos = (size(u, order[2]) >> 1) + 1
            nneg = size(u, order[2]) - npos
            nneg == 0 && return cache

            # copy negative frequencies in second direction
            blk_u = ntuple(i->i ∈ order[2] ? (npos+1:size(u, i))                    : (1:size(u, i)), dim)
            blk_c = ntuple(i->i ∈ order[2] ? (size(cache, i)-nneg+1:size(cache, i)) : (1:size(u, i)), dim)
            broadcast!($op, @view($dst[$blk_dst...]), @view($src[$blk_src...]), @view($dst[$blk_dst...]))

            return $dst
        end

        function $name($dst, $src, dim, order::NTuple{3})
            # copy non-negative frequencies in both directions
            blk = ntuple(i->i ∈ order[2:3] ? (1:(size(u, i) >> 1) + 1) : (1:size(u, i)), dim)
            broadcast!($op, @view($dst[blk...]), @view($src[blk...]), @view($dst[blk...]))

            for d in order[2:end]
                # get positive and negative frequency lengths
                npos = (size(u, d) >> 1) + 1
                nneg = size(u, d) - npos
                nneg == 0 && continue

                # get block ranges
                blk_u = ntuple(i->i ∈ order[2:end] ? (i == d ? (npos+1:size(u, d))                    : (1:(size(u, i) >> 1)+1)) : (1:size(u, i)), dim)
                blk_c = ntuple(i->i ∈ order[2:end] ? (i == d ? (size(cache, d)-nneg+1:size(cache, d)) : (1:(size(u, i) >> 1)+1)) : (1:size(u, i)), dim)

                # do copy
                broadcast!($op, @view($dst[$blk_dst...]), @view($src[$blk_src...]), @view($dst[$blk_dst...]))
            end

            # copy final block corner
            blk_u = ntuple(i->i ∈ order[2:end] ? ((size(u, i) >> 1)+2:size(u, i))               : (1:size(u, i)), dim)
            blk_c = ntuple(i->i ∈ order[2:end] ? (size(cache, i)-size(u, i)+(size(u, i) >> 1)+2:size(cache, i)) : (1:size(u, i)), dim)
            broadcast!($op, @view($dst[$blk_dst...]), @view($src[$blk_src...]), @view($dst[$blk_dst...]))

            return $dst
        end

        $name($dst, $src, dim, order::NTuple{N}) where {N} = throw(NotImplementedError(order))
    end
end

_apply_mask!(cache::Array{T}) where {T} = (cache .= zero(T); return cache)
