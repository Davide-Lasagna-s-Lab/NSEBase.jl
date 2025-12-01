using Test

using NSEBase

# --------------------- #
# fake type for testing #
# --------------------- #
struct MyField <: AbstractScalarField{2, Float64}
    data::Matrix{Float64}
end
Base.parent(u::MyField) = u.date
Base.similar(u::MyField, ::Type{T}=Float64) = MyField(similar(parent(u)))
NSEBase.hsize(u::MyField) = size(parent(u), 2)



include("test_notimplementederror.jl")
