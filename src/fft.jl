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
struct FFTPlans{DIM, T, ORDER, PLAN, IPLAN}
    plan::PLAN
    iplan::IPLAN
    cache::Array{Complex{T}, DIM}
    norm::Int

    function FFTPlans(shape::Dims{DIM},
                      order::NTuple{H, Int},
                           ::Type{T}=Float64;
                      flags::UInt32 =EXHAUSTIVE,
                  timelimit::Real   =NO_TIMELIMIT) where {DIM, H, T}
        # create arrays
        shape = _get_padded_shape(shape, order)
        spectral_array = zeros(Complex{T}, _get_transform_shape(shape, order[1]))
        physical_array = zeros(T, shape)
        norm = 1/prod(shape[collect(order)])

        # construct plans
        plan  = FFTW.plan_rfft( physical_array,                  order, flags=flags, timelimit=timelimit)
        iplan = FFTW.plan_brfft(spectral_array, shape[order[1]], order, flags=flags, timelimit=timelimit)

        new{DIM, T, order, typeof(plan), typeof(iplan)}(plan, iplan, spectral_array, norm)
    end
end


# ------------------------ #
# in-place transformations #
# ------------------------ #
# physical -> spectral fields
function (f::FFTPlans{DIM, T})(û::VectorField{N, P},
                               u::VectorField{N, S},
                             add::Bool=true) where {DIM, T, N, S<:AbstractScalarField{DIM, Complex{T}}, P<:AbstractScalarField{DIM, T}}
    for n in 1:N
        f(û[n], u[n], add)
    end
    return û
end

(f::FFTPlans{DIM, T})(û::AbstractScalarField{DIM, Complex{T}},
                      u::AbstractScalarField{DIM,         T},
                    add::Bool) where {DIM, T} = add ? _add_forward_transform!(û, u, f) : _forward_transform!(û, u, f)

function _forward_transform(û, u, f::FFTPlans{DIM, T, ORDER}) where {DIM, T, ORDER}
    FFTW.unsafe_execute!(f.plan, parent(u), f.spectral_cache, DIM, ORDEr)
    _copy_from_padded!(parent(û), f.spectral_cache)
    parent(û) .*= f.norm
    return û
end

function _add_forward_transform!(û, u, f::FFTPlans{DIM, T, ORDER}) where {DIM, T, ORDER}
    FFTW.unsafe_execute!(f.plan, parent(u), f.spectral_cache)
    f.spectral_cache .*= f.norm
    _add_from_padded!(parent(û), f.spectral_cache, DIM, ORDER)
    return û
end

# spectral -> physical fields
function (f::FFTPlans{DIM, T})(u::VectorField{N, S},
                               û::VectorField{N, P},
                            safe::Bool=true) where {DIM, T, N, S<:AbstractScalarField{DIM, Complex{T}}, P<:AbstractScalarField{DIM, T}}
    for n in 1:N
        f(u[n], û[n], safe)
    end
    return u
end

(f::FFTPlans{DIM, T})(u::AbstractScalarField{DIM, T},
                      û::AbstractScalarField{DIM, Complex{T}}) where {DIM, T} = _backward_transform!(u, û, f)

function _backward_transform!(u, û, f::FFTPlans{DIM, T, ORDER}) where {DIM, T, ORDER}
    _copy_to_padded!(_apply_mask!(f.spectral_cache), parent(û), DIM, ORDER)
    FFTW.unsafe_execute!(f.iplan, f.spectral_cache, parent(u))
    return u
end


# ----------------- #
# utility functions #
# ----------------- #
function _get_padded_shape(shape, order)
    new_shape = zeros(Int, length(shape))
    for (i, s) in enumerate(shape)
        if i ∈ order
            new_shape[i] = (3*s)>>1 + 1 - ((3*s)>>1)&1
        else
            new_shape[i] = s
        end
    end
    return tuple(new_shape...)
end

function _get_transform_shape(shape, dim)
    new_shape = zeros(Int, length(shape))
    for (i, s) in enumerate(shape)
        if i == dim
            _newshape[i] = (s >> 1) + 1
        else
            new_shape[i] = s
        end
    end
    return tuple(new_shape...)
end

# copy function for dealiasing
for name in [:_copy_from_padded!, :_add_from_padded!, :_copy_to_padded!]
    # get useful variables for evaluation
    src, dst = name ∈ [:_copy_from_padded!, :_add_from_padded!] ? (:cache, :u) : (:u, :cache)
    blk_src, blk_dst = name ∈ [:_copy_from_padded!, :_add_from_padded!] ? (:blk_c, :blk_u) : (:blk_u, :blk_c)
    op = name ∈ [:_copy_from_padded!, :_copy_to_padded!] ? ((x, y)->x) : +

    @eval begin
        function $name($dst, $src, dim, ::NTuple{1})
            # copy non-negative frequencies
            blk = ntuple(i->1:size(u, i), dim)
            broadcast!($op, @view($dst[blk...]), @view($src[blk...]), @view($src[blk...]))
            return $dst
        end

        function $name($dst, $src, dim, order::NTuple{2})
            # copy non-negative frequencies in both directions
            blk = ntuple(i->i == order[2] ? (1:(size(u, i) >> 1) + 1) : (1:size(u, i)), dim)
            broadcast!($op, @view($dst[blk...]), @view($src[blk...]), @view($src[blk...]))

            # get positive and negative frequency lengths
            npos = (size(u, order[2]) >> 1) + 1
            nneg = size(u, order[2]) - npos
            nneg == 0 && return cache

            # copy negative frequencies in second direction
            blk_u = ntuple(i->i ∈ order[2] ? (npos+1:size(u, i))                    : (1:size(u, i)), dim)
            blk_c = ntuple(i->i ∈ order[2] ? (size(cache, i)-nneg+1:size(cache, i)) : (1:size(u, i)), dim)
            broadcast!($op, @view($dst[$blk_dst...]), @view($src[$blk_src...]), @view($src[$blk_src...]))

            return $dst
        end

        function $name($dst, $src, dim, order::NTuple{3})
            # copy non-negative frequencies in both directions
            blk = ntuple(i->i ∈ order[2:3] ? (1:(size(u, i) >> 1) + 1) : (1:size(u, i)), dim)
            broadcast!($op, @view($dst[blk...]), @view($src[blk...]), @view($src[blk...]))

            for d in order[2:end]
                # get positive and negative frequency lengths
                npos = (size(u, d) >> 1) + 1
                nneg = size(u, d) - npos
                nneg == 0 && continue

                # get block ranges
                blk_u = ntuple(i->i ∈ order[2:end] ? (i == d ? (npos+1:size(u, d))                    : (1:(size(u, i) >> 1)+1)) : (1:size(u, i)), dim)
                blk_c = ntuple(i->i ∈ order[2:end] ? (i == d ? (size(cache, d)-nneg+1:size(cache, d)) : (1:(size(u, i) >> 1)+1)) : (1:size(u, i)), dim)

                # do copy
                broadcast!($op, @view($dst[$blk_dst...]), @view($src[$blk_src...]), @view($src[$blk_src...]))
            end

            # copy final block corner
            blk_u = ntuple(i->i ∈ order[2:end] ? ((size(u, i) >> 1)+2:size(u, i))               : (1:size(u, i)), dim)
            blk_c = ntuple(i->i ∈ order[2:end] ? (size(cache, i)-size(u, i)+(size(u, i) >> 1)+2:size(cache, i)) : (1:size(u, i)), dim)
            broadcast!($op, @view($dst[$blk_dst...]), @view($src[$blk_src...]), @view($src[$blk_src...]))

            return $dst
        end

        $name($dst, $src, dim, order::NTuple{N}) where {N} = throw(NotImplementedError(order))
    end
end

_apply_mask!(cache::Array{T}) where {T} = (cache .= zero(T); return cache)
