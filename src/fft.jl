# Abstract interface to aid in the creation of fft plans using FFTW.jl, which
# are assumed to be quite useful for the definition and manipulation of a given
# field.

# TODO: add optional padded size for user when defining plan
# TODO: finish documentation

# ---------------- #
# transform object #
# ---------------- #
"""
    FFTPlans{DEALIAS}

Forward and inverse FFTW plans which can be used to efficiently go
between the Field and FTField representation of a scalar field.

# Fields
- `plan`: forward FFTW plan
- `iplan`: inverse FFTW plan
- `cache`: intermediate array useful for dealiasing
           and safety for forward transform
- `norm`: pre-computed normalisation factor for
          the forward transform
"""
struct FFTPlans{DEALIAS, D, T, ORDER, PLAN, IPLAN}
     plan::PLAN
    iplan::IPLAN
    cache::Array{Complex{T}, D}
     norm::T

    """
        FFTPlans(shape::Dims,
                 order::Dims,
                      ::Type{T}=Float64;
               dealias::Bool=true,
                 flags::UInt32=FFTW.EXHAUSTIVE,
             timelimit::Real=FFTW.NO_TIMELIMIT)

    Construct FFTW plans which work for physical arrays with size
    equal to `shape`, where `order` is a tuple of the dimensions of
    the arrays that are transformed by the plans in the order of
    transformation.

    Optionally, the `dealias` flag can be used to use the 3/2 padding
    rule for the physical field. The corresponding `FTField` and `Field`
    pair that are compatible with the dealiasing plans can be constructed
    by setting the `dealias` flag to `true` when constructing the `Field`,
    i.e. `û = FTField(g)` and `u = Field(g, dealias=true)`.
    """
    function FFTPlans(shape::Dims{D},
                      order::Dims,
                           ::Type{T}=Float64;
                    dealias::Bool   =true,
                      flags::UInt32 =FFTW.EXHAUSTIVE,
                  timelimit::Real   =FFTW.NO_TIMELIMIT) where {D, T}
        # create arrays
        shape = dealias ? padded_size(shape, order) : shape
        spectral_array = zeros(Complex{T}, _get_transform_shape(shape, order[1]))
        physical_array = zeros(T, shape)
        norm = T(1/prod(shape[collect(order)]))

        # construct plans
        plan  = FFTW.plan_rfft( physical_array,                  order, flags=flags, timelimit=timelimit)
        iplan = FFTW.plan_brfft(spectral_array, shape[order[1]], order, flags=flags, timelimit=timelimit)

        new{dealias, D, T, order, typeof(plan), typeof(iplan)}(plan, iplan, spectral_array, norm)
    end
end

FFTPlans(g::AbstractGrid{T, D, Axes, Hs, Ht}; kwargs...) where {T, D, Axes, Hs, Ht} = FFTPlans(size(g), (Hs..., Ht), T; kwargs...)
FFTPlans(u::FTField;               kwargs...)                 = FFTPlans(grid(u);       kwargs...)
FFTPlans(u::Field;                 kwargs...)                 = FFTPlans(grid(u);       kwargs...)

growto(u::FTField, size) = throw(NotImplementedError(u, size))
growto(u::VectorField, size) = throw(NotImplementedError(u, size))


# ------------------------ #
# in-place transformations #
# ------------------------ #
# physical -> spectral
function (f::FFTPlans)(û::VectorField{N, <:FTField},
                       u::VectorField{N, <:Field};
                     add::Bool=false,
               use_cache::Bool=false) where {N}
    for n in 1:N
        f(û[n], u[n]; add=add, use_cache=use_cache)
    end
    return û
end

(f::FFTPlans{true})(û::FTField{G},
                    u::Field{G};
                  add::Bool=false,
            use_cache::Bool=false) where {G} =
    add ? _add_forward_transform_dealias!(û, u, f) : _forward_transform_dealias!(û, u, f)

(f::FFTPlans{false})(û::FTField{G},
                     u::Field{G};
                   add::Bool=false,
             use_cache::Bool=false) where {G} =
    add ? _add_forward_transform!(û, u, f) : (use_cache ? _forward_transform_with_cache!(û, u, f) : _forward_transform_without_cache!(û, u, f))

function _forward_transform_dealias!(û, u, f::FFTPlans{DEALIAS, D, T, ORDER}) where {DEALIAS, D, T, ORDER}
    FFTW.unsafe_execute!(f.plan, parent(u), f.cache)
    _copy_from_padded!(û, f.cache, D, ORDER)
    û .*= f.norm
    return û
end

function _add_forward_transform_dealias!(û, u, f::FFTPlans{DEALIAS, D, T, ORDER}) where {DEALIAS, D, T, ORDER}
    FFTW.unsafe_execute!(f.plan, parent(u), f.cache)
    f.cache .*= f.norm
    _add_from_padded!(û, f.cache, D, ORDER)
    return û
end

function _forward_transform_without_cache!(û, u, f::FFTPlans{DEALIAS, D, T, ORDER}) where {DEALIAS, D, T, ORDER}
    FFTW.unsafe_execute!(f.plan, parent(u), parent(û))
    û .*= f.norm
    return û
end

function _forward_transform_with_cache!(û, u, f::FFTPlans{DEALIAS, D, T, ORDER}) where {DEALIAS, D, T, ORDER}
    FFTW.unsafe_execute!(f.plan, parent(u), f.cache)
    f.cache .*= f.norm
    û .= f.cache
    return û
end

function _add_forward_transform!(û, u, f::FFTPlans{DEALIAS, D, T, ORDER}) where {DEALIAS, D, T, ORDER}
    FFTW.unsafe_execute!(f.plan, parent(u), f.cache)
    f.cache .*= f.norm
    û .+= f.cache
    return û
end

# spectral -> physical
function (f::FFTPlans)(u::VectorField{N, <:Field},
                       û::VectorField{N, <:FTField};
                    safe::Bool=true) where {N}
    for n in 1:N
        f(u[n], û[n]; safe=safe)
    end
    return u
end

(f::FFTPlans{true})(u::Field{G},
                    û::FTField{G};
                 safe::Bool=true) where {G} = _backward_transform_dealias!(u, û, f)

(f::FFTPlans{false})(u::Field{G},
                     û::FTField{G};
                  safe::Bool=true) where {G} = safe ? _backward_transform!(u, û, f) : _unsafe_backward_transform!(u, û, f)

function _backward_transform_dealias!(u, û, f::FFTPlans{DEALIAS, D, T, ORDER}) where {DEALIAS, D, T, ORDER}
    _copy_to_padded!(_apply_mask!(f.cache), û, D, ORDER)
    FFTW.unsafe_execute!(f.iplan, f.cache, parent(u))
    return u
end

function _backward_transform!(u, û, f::FFTPlans{DEALIAS, D, T, ORDER}) where {DEALIAS, D, T, ORDER}
    f.cache .= û
    FFTW.unsafe_execute!(f.iplan, f.cache, parent(u))
    return u
end

function _unsafe_backward_transform!(u, û, f::FFTPlans{DEALIAS, D, T, ORDER}) where {DEALIAS, D, T, ORDER}
    FFTW.unsafe_execute!(f.iplan, parent(û), parent(u))
    return u
end

# --------------------- #
# allocating transforms #
# --------------------- #
function FFT(u::Field{G}) where {T, D, Axes, Hs, Ht, G<:AbstractGrid{T, D, Axes, Hs, Ht}}
    H = (Hs..., Ht)
    û = FTField(grid(u))
    parent(û) .= rfft(parent(u), H)
    û .*= 1/prod(fft_norm(grid(u)))
    return û
end
FFT(u::VectorField{N, <:Field}) where {N} = VectorField([FFT(u[n]) for n in 1:N]...)

"""
    FFT(u::Field, target_size)

Transform a physical field to spectral space, then grow the result to a target
homogeneous resolution using `growto`.  Concrete grid packages that support
resolution changes should extend `growto(grid, target_size)` and
`growto(û::FTField, target_size)`.
"""
FFT(u::Field, target_size) = growto(FFT(u), target_size)
FFT(u::VectorField{N, <:Field}, target_size) where {N} =
    VectorField([FFT(u[n], target_size) for n in 1:N]...)

function IFFT(û::FTField{G}) where {T, D, Axes, Hs, Ht, G<:AbstractGrid{T, D, Axes, Hs, Ht}}
    H = (Hs..., Ht)
    u = Field(grid(û))
    parent(u) .= brfft(parent(û), size(grid(û), H[1]), H)
    return u
end
IFFT(u::VectorField{N, <:FTField}) where {N} = VectorField([IFFT(u[n]) for n in 1:N]...)

"""
    IFFT(û::FTField, target_size)

Grow a spectral field to `target_size`, then transform it back to physical
space on the grown grid.
"""
function IFFT(û::FTField{G}, target_size) where {T, D, Axes, Hs, Ht, G<:AbstractGrid{T, D, Axes, Hs, Ht}}
    H = (Hs..., Ht)
    grown = growto(û, target_size)
    u = Field(grid(grown))
    parent(u) .= brfft(parent(grown), size(grid(grown), H[1]), H)
    return u
end
IFFT(u::VectorField{N, <:FTField}, target_size) where {N} =
    VectorField([IFFT(u[n], target_size) for n in 1:N]...)


# ----------------- #
# utility functions #
# ----------------- #
"""
    padded_size(shape::Dims, dims) -> Dims

Return `shape` with the dimensions in `dims` padded by the 3/2 dealiasing
rule used by the FFT plans.  The padded size is adjusted to be odd, matching
the existing channel-flow convention.

```julia
padded_size((Ny, Nx, Nz, Nt), (2, 3, 4))
```
"""
function padded_size(shape::Dims, dims)
    new_shape = zeros(Int, length(shape))
    for (i, s) in enumerate(shape)
        if i ∈ dims
            new_shape[i] = (3*s)>>1 + 1 - ((3*s)>>1)&1
        else
            new_shape[i] = s
        end
    end
    return tuple(new_shape...)
end

padded_size(shape::Dims, ::Val{true}) = padded_size(shape, 1:length(shape))
padded_size(shape::Dims, ::Val{false}) = shape

function _get_transform_shape(shape, dim)
    new_shape = zeros(Int, length(shape))
    for (i, s) in enumerate(shape)
        if i == dim
            new_shape[i] = (s >> 1) + 1
        else
            new_shape[i] = s
        end
    end
    return tuple(new_shape...)
end

# copy function for dealiasing
# TODO: properly benchmark and check allocation of these functions
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
            # ! bug???
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
