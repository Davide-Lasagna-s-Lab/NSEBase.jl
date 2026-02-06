# Wrapper for the Navier-Stokes equation that includes the project-expand steps
# required for wall-bounded variational optimisation.

struct ProjectedNSE{EQ, LEQ, B, N, S}
       nl::EQ
       ln::LEQ
     base::B
    cache::NTuple{2, VectorField{N, S}}

    function ProjectedNSE(u::S, nl::NSE{S}, ln::LNSE{S}, base::B) where {S<:AbstractScalarField, B}
        # construct cache
        cache = ntuple(_->VectorField(u, N=ndim(nl)), 2)

        new{typeof(nl), typeof(ln), B, ndim(nl), eltype(cache[1])}(nl, ln, base, cache)
    end
end

function (eq::ProjectedNSE)(out::ProjectedField,
                              a::ProjectedField)
    # aliases
    u   = eq.cache[1]
    N_u = eq.cache[2]

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
    v    = eq.cache[1]
    M_uv = eq.cache[2]

    # expand coefficients into spectral fields
    expand!(v, b)

    # operator action
    eq.ln(0, v, M_uv)

    # project result back onto basis
    project!(out, M_uv)

    return out
end
