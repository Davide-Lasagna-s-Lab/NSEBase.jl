# Wrapper for the Navier-Stokes equation that includes the project-expand steps
# required for wall-bounded variational optimisation.

struct ProjectedNSE{EQ, LEQ, B, N, S1, S2}
        nl::EQ
        ln::LEQ
      base::B
    cache1::VectorField{N, S1}
    cache2::VectorField{N, S2}

    ProjectedNSE{N}(nl::EQ, ln::LEQ, base::B, caches) where {EQ, LEQ, B, N} =
        new{EQ, LEQ, B, N, eltype(caches[1]), eltype(caches[2])}(nl, ln, base, caches...)
end

ProjectedNSE(grid::AbstractGrid, N::Int, nl, ln, base) =
    ProjectedNSE{N}(nl, ln, base, ntuple(_->VectorField(grid, FTField, N=N), 2))

function (eq::ProjectedNSE)(out::ProjectedField,
                              a::ProjectedField)
    # aliases
    u   = eq.cache1
    N_u = eq.cache2

    # expand coefficients into spectral field
    expand!(u, a)
    add_base_flow!(u, eq.base)

    # operator action
    eq.nl(0, u, N_u)

    # project result back onto basis
    project!(out, N_u)

    return out
end

function (eq::ProjectedNSE)(out::ProjectedField,
                               ::ProjectedField,
                              b::ProjectedField)
    # aliases
    v    = eq.cache1
    M_uv = eq.cache2

    # expand coefficients into spectral fields
    expand!(v, b)

    # operator action
    eq.ln(0, v, M_uv)

    # project result back onto basis
    project!(out, M_uv)

    return out
end




# TODO: remove the stuff below



# nonlinear operator
# abstract type NSE{S<:AbstractScalarField} end
# (f::NSE{S})(t::Real, u::F, out::F) where {S<:AbstractScalarField, N, F<:Union{S, VectorField{N, S}}} = throw(NotImplementedError(t, u, out))
# ndim(op::NSE) = throw(NotImplementedError(op))

# linearised operator
# abstract type LNSE{S<:AbstractScalarField} end
# (f::LNSE{S})(t::Real, u::F, v::F, out::F) where {S<:AbstractScalarField, N, F<:Union{S, VectorField{N, S}}} = throw(NotImplementedError(t, u, v, out))
# (f::LNSE{S})(t::Real, v::F, out::F) where {S<:AbstractScalarField, N, F<:Union{S, VectorField{N, S}}} = throw(NotImplementedError(t, v, out))
# ndim(op::LNSE) = throw(NotImplementedError(op))
