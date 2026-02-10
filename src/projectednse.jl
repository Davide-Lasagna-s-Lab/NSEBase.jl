# Wrapper for the Navier-Stokes equation that includes the project-expand steps
# required for wall-bounded variational optimisation.

struct ProjectedNSE{EQ, LEQ, B, N, S1, S2}
        nl::EQ
        ln::LEQ
      base::B
    cache1::VectorField{N, S1}
    cache2::VectorField{N, S2}

    ProjectedNSE(nl::EQ, ln::LEQ, base::B, caches...) where {EQ, LEQ, B} =
        new{EQ, LEQ, B, length(caches[1]), eltype(caches[1]), eltype(caches[2])}(nl, ln, base, caches...)
end

function ProjectedNSE(u::S, nl::NSE{S}, ln::LNSE{S}, base::B) where {S<:AbstractScalarField, B}
    # construct cache
    caches = ntuple(_->VectorField(u, N=ndim(nl)), 2)

    ProjectedNSE(nl, ln, base, caches...)
end

function (eq::ProjectedNSE)(out::ProjectedField,
                              a::ProjectedField)
    # aliases
    u   = eq.cache1
    N_u = eq.cache2

    # expand coefficients into spectral field
    expand!(u, a)
    add_base!(u, eq.base)

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
