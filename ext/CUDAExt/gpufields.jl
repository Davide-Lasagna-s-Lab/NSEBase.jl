# Specialised constructors and methods for fields defined on a GPUGrid.

# ------------------------- #
# special CUDA constructors #
# ------------------------- #
"""
    FTField(g::GPUGrid)

Construct a `FTField` such that the parent data is stored
on the device.
"""
NSEBase.FTField(g::GPUGrid{T}) where {T} =
    NSEBase.FTField(g, CUDA.zeros(Complex{T}, NSEBase.transform_size(g)))

"""
    Field(g::GPUGrid)

Construct a `Field` such that the parent data is stored
on the device.
"""
NSEBase.Field(g::GPUGrid{T}; dealias=true) where {T} = dealias ?  NSEBase.Field(g, CUDA.zeros(T, NSEBase.get_padded_size(size(g), NSEBase.fft_storage_dims(g)))) :
                                                                  NSEBase.Field(g, CUDA.zeros(T, size(g)))

"""
    ProjectedField(g::GPUGrid)

Construct a `ProjectedField` such that the parent data and
modes are stored on the device.
"""
function NSEBase.ProjectedField(grid::GPUGrid{T}, modes) where {T}
    Nm = size(modes[1], 1)
    return NSEBase.ProjectedField(grid,
                                  CUDA.zeros(Complex{T}, Nm,
                                        NSEBase.transform_size(grid)[collect(NSEBase.fft_storage_dims(grid))]...),
                                  modes)
end


# ------------------------------ #
# adapt methods for CUDA kernels #
# ------------------------------ #
adapt_structure(to, u::NSEBase.FTField) =
    NSEBase.FTField(adapt_structure(to, NSEBase.grid(u)), adapt_structure(to, parent(u)))
adapt_structure(to, u::NSEBase.VectorField{N}) where {N} =
    NSEBase.VectorField(ntuple(n -> adapt_structure(to, u[n]), Val(N))...)
function adapt_structure(to, a::NSEBase.ProjectedField)
    g = adapt_structure(to, NSEBase.grid(a))
    data = adapt_structure(to, parent(a))
    mds = adapt_structure(to, NSEBase.modes(a))
    return NSEBase.ProjectedField(g, data, mds)
end


# --------------------------------- #
# opinionated CUDA field converters #
# --------------------------------- #
"""
    CUDA.cu(u::FTField)
    CUDA.cu(u::Field)
    CUDA.cu(u::VectorField)
    CUDA.cu(a::ProjectedField)

Opinionated converter for field types that moves all their
respective data to the device.

Automatically converts all numerics type to `Float32`.
"""
CUDA.cu(u::NSEBase.FTField)                  =
    NSEBase.FTField(CUDA.cu(NSEBase.grid(u)), CUDA.cu(parent(u)))
CUDA.cu(u::NSEBase.Field)                    =
    NSEBase.Field(CUDA.cu(NSEBase.grid(u)), CUDA.cu(parent(u)))
CUDA.cu(u::NSEBase.VectorField{N}) where {N} =
    NSEBase.VectorField(ntuple(n -> CUDA.cu(u[n]), Val(N))...)
CUDA.cu(a::NSEBase.ProjectedField)           =
    NSEBase.ProjectedField(CUDA.cu(NSEBase.grid(a)), CUDA.cu(parent(a)), CUDA.cu(NSEBase.modes(a)))


# ! could define some default methods that just don't work for GPUGrids (such as getindex, etc.)?
