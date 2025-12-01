# Abstract equation interface that the user has to implement.

# nonlinear operator
abstract type NSE{S<:AbstractScalarField} end
(f::NSE{S})(t::Real, u::F, out::F) where {S<:AbstractScalarField, N, F<:Union{S, VectorField{N, S}}} = throw(NotImplementedError(t, u, out))
ndim(op::NSE) = throw(NotImplementedError(op))


# linearised operator
abstract type LNSE{S<:AbstractScalarField} end
(f::LNSE{S})(t::Real, u::F, v::F, out::F) where {S<:AbstractScalarField, N, F<:Union{S, VectorField{N, S}}} = throw(NotImplementedError(t, u, v, out))
(f::LNSE{S})(t::Real, v::F, out::F) where {S<:AbstractScalarField, N, F<:Union{S, VectorField{N, S}}} = throw(NotImplementedError(t, v, out))
ndim(op::LNSE) = throw(NotImplementedError(op))
