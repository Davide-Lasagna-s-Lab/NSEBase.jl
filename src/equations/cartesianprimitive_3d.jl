# Concrete Navier-Stokes operators for three-component (u, v, w) Cartesian flows.
#
# Each operator is a callable struct whose `(::Real, u, out)` method evaluates
# one of the following PDEs in spectral space, writing the result into `out`:
#
#   CartesianPrimitive3DNSE  — nonlinear NSE: out = Δu/Re − (u·∇)u + force
#   CartesianPrimitive3DLNSE{Forward}          — forward linearised operator
#   CartesianPrimitive3DLNSE{AdjointContinuous}— continuous adjoint
#   CartesianPrimitive3DLNSE{AdjointDiscrete}  — discrete adjoint
#
# All variants share the same cache layout: `scache` holds spectral-space
# scratch VectorFields (for derivative intermediates) and `pcache` holds
# physical-space scratch VectorFields (for de-aliased nonlinear products).
#
# Body forces are injected via a callable `force(out, u, mode::Mode)`.
# Pass `NoForce()` when no body force is needed.
#
# Construct via [`construct_equations`](@ref) rather than directly to ensure
# cache and plan sizes are consistent.
#
# Halo-exchange pattern
# ---------------------
# All call methods follow the init/interior/overlap/wait/boundary split:
#
#   requests = init_requests!(u)
#   interior_*(out, u); ...          ← interior rows, independent of halo
#   <overlap work: FFT plans, etc.>  ← runs while MPI halos are in flight
#   wait_requests!(requests)
#   boundary_*(out, u); ...          ← boundary rows that need halo data
#
# For serial grids: init_requests! returns nothing, wait_requests!(nothing)
# is a no-op, interior_* does the full computation, boundary_* is a no-op —
# zero extra allocations. MPIExt overrides all four for DecomposedGrid
# fields, achieving communication–computation overlap without any Task objects.
#
# NOTE: on a single node this overlap measures no faster than a blocking swap
# (benchmarks/mpi_overlap.jl); it is kept pending a multi-node benchmark. See
# https://github.com/Davide-Lasagna-s-Lab/NSEBase.jl/issues/21 before changing it.


# ----------------------- #
# concrete 3D NSE struct  #
# ----------------------- #
"""
    CartesianPrimitive3DNSE{T, FFT, S, P, BF}

Nonlinear Navier-Stokes operator for three-component Cartesian flows.

Evaluates `out = Δu/Re − (u·∇)u + force(out, u, Forward())` in spectral space,
using dealiased physical-space products for the nonlinear term.

# Fields
- `Re`: Reynolds number
- `plans`: `FFTPlans` for physical↔spectral transforms
- `scache`: spectral-space scratch `VectorField`s (3 entries of 3 components)
- `pcache`: physical-space scratch `VectorField`s (4 entries of 3 components)
- `force`: body-force callable with signature `(out, u, mode)`
"""
mutable struct CartesianPrimitive3DNSE{T, FFT, S, P, BF}
              Re::T
     const plans::FFT
    const scache::Vector{VectorField{3, S}}
    const pcache::Vector{VectorField{3, P}}
    const  force::BF
end

function CartesianPrimitive3DNSE(g::G, Re;
                             force::BF=NoForce(),
                             flags    =FFTW.EXHAUSTIVE) where {T, G<:AbstractGrid{T}, BF}
    plans  = FFTPlans(g, flags=flags)
    scache = [VectorField([FTField(g)               for _ in 1:3]...) for _ in 1:3]
    pcache = [VectorField([  Field(g, dealias=true) for _ in 1:3]...) for _ in 1:4]
    return CartesianPrimitive3DNSE(T(Re), plans, scache, pcache, force)
end

# ----------------------- #
# concrete 3D LNSE struct #
# ----------------------- #
"""
    CartesianPrimitive3DLNSE{MODE, T, FFT, S, P, BF}

Linearised Navier-Stokes operator for three-component Cartesian flows,
parameterised on `MODE <: Mode` to select between the forward linearisation,
continuous adjoint, and discrete adjoint.

The three-argument call `eq(t, u, v, out)` first caches the physical base-flow
gradients from `u`, then delegates to the two-argument form `eq(t, v, out)`.
The two-argument form applies the chosen linearised operator to `v`.

# Fields
- `Re`: Reynolds number
- `plans`, `scache`, `pcache`, `force`: same as [`CartesianPrimitive3DNSE`](@ref)
"""
mutable struct CartesianPrimitive3DLNSE{MODE, T, FFT, S, P, BF}
              Re::T
     const plans::FFT
    const scache::Vector{VectorField{3, S}}
    const pcache::Vector{VectorField{3, P}}
    const  force::BF

    CartesianPrimitive3DLNSE{MODE}(Re::T,
                                plans::FFT,
                               scache::Vector{VectorField{3, S}},
                               pcache::Vector{VectorField{3, P}},
                                force::BF) where {MODE, T, FFT, S, P, BF} =
        new{MODE, T, FFT, S, P, BF}(Re, plans, scache, pcache, force)
end

function CartesianPrimitive3DLNSE(g::G, Re;
                               mode::Mode=AdjointDiscrete(),
                              force::BF  =NoForce(),
                              flags      =FFTW.EXHAUSTIVE) where {T, G<:AbstractGrid{T}, BF}
    plans  = FFTPlans(g, flags=flags)
    scache = [VectorField([FTField(g)               for _ in 1:3]...) for _ in 1:4]
    pcache = [VectorField([  Field(g, dealias=true) for _ in 1:3]...) for _ in 1:8]
    return CartesianPrimitive3DLNSE{typeof(mode)}(T(Re), plans, scache, pcache, force)
end


# ------------- #
# nonlinear NSE #
# ------------- #
function (eq::CartesianPrimitive3DNSE)(::Real,
                                      u::VectorField{3, F},
                                    out::VectorField{3, F}) where {F<:FTField}
    dudx = eq.scache[1]; dudy = eq.scache[2]; dudz = eq.scache[3]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]

    requests = init_requests!(u)
    init_laplacian!(out, u)
    init_ddx!(dudx, u)
    init_ddy!(dudy, u)
    init_ddz!(dudz, u)
    eq.plans(U, u)
    wait_requests!(requests)
    complete_laplacian!(out, u)
    complete_ddx!(dudx, u)
    complete_ddy!(dudy, u)
    complete_ddz!(dudz, u)
    out .*= 1/eq.Re

    eq.plans(dUdx, dudx); eq.plans(dUdy, dudy); eq.plans(dUdz, dudz)
    for n in 1:3
        @. dUdx[n] = -U[1]*dUdx[n] - U[2]*dUdy[n] - U[3]*dUdz[n]
    end
    eq.plans(out, dUdx, add=true)

    eq.force(out, u, Forward())
    return out
end


# -------------- #
# linearised NSE #
# -------------- #
# 3-arg: set up base-flow cache then delegate to 2-arg
function (eq::CartesianPrimitive3DLNSE)(::Real,
                                       u::VectorField{3, F},
                                       v::VectorField{3, F},
                                     out::VectorField{3, F}) where {F<:FTField}
    dudx = eq.scache[1]; dudy = eq.scache[2]; dudz = eq.scache[3]
    U    = eq.pcache[1]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]

    requests = init_requests!(u)
    init_ddx!(dudx, u)
    init_ddy!(dudy, u)
    init_ddz!(dudz, u)
    eq.plans(U, u)
    wait_requests!(requests)
    complete_ddx!(dudx, u)
    complete_ddy!(dudy, u)
    complete_ddz!(dudz, u)
    eq.plans(dUdy, dudy); eq.plans(dUdz, dudz)

    eq(0, v, out)
    return out
end

# forward LNSE
function (eq::CartesianPrimitive3DLNSE{Forward})(::Real,
                                                 v::VectorField{3, F},
                                               out::VectorField{3, F}) where {F<:FTField}
    dudx = eq.scache[1]; dvdx = eq.scache[2]; dvdy = eq.scache[3]; dvdz = eq.scache[4]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]
    V    = eq.pcache[5]; dVdx = eq.pcache[6]; dVdy = eq.pcache[7]; dVdz = eq.pcache[8]

    requests = init_requests!(v)
    init_laplacian!(out, v)
    init_ddx!(dvdx, v)
    init_ddy!(dvdy, v)
    init_ddz!(dvdz, v)
    eq.plans(V, v)
    wait_requests!(requests)
    complete_laplacian!(out, v)
    complete_ddx!(dvdx, v)
    complete_ddy!(dvdy, v)
    complete_ddz!(dvdz, v)
    out .*= 1/eq.Re

    eq.plans(dUdx, dudx)
    eq.plans(dVdx, dvdx); eq.plans(dVdy, dvdy); eq.plans(dVdz, dvdz)
    for n in 1:3
        @. dVdx[n]  = -U[1]*dVdx[n] - U[2]*dVdy[n] - U[3]*dVdz[n]
        @. dVdx[n] -=  V[1]*dUdx[n] + V[2]*dUdy[n] + V[3]*dUdz[n]
    end
    eq.plans(out, dVdx, add=true)

    eq.force(out, v, Forward())
    return out
end

# continuous adjoint LNSE
function (eq::CartesianPrimitive3DLNSE{AdjointContinuous})(::Real,
                                                           v::VectorField{3, F},
                                                         out::VectorField{3, F}) where {F<:FTField}
    dudx = eq.scache[1]; dvdx = eq.scache[2]; dvdy = eq.scache[3]; dvdz = eq.scache[4]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]
    V    = eq.pcache[5]; dVdx = eq.pcache[6]; dVdy = eq.pcache[7]; dVdz = eq.pcache[8]

    requests = init_requests!(v)
    init_laplacian!(out, v)
    init_ddx!(dvdx, v)
    init_ddy!(dvdy, v)
    init_ddz!(dvdz, v)
    eq.plans(V, v)
    wait_requests!(requests)
    complete_laplacian!(out, v)
    complete_ddx!(dvdx, v)
    complete_ddy!(dvdy, v)
    complete_ddz!(dvdz, v)
    out .*= 1/eq.Re
    eq.plans(dUdx, dudx)
    eq.plans(dVdx, dvdx); eq.plans(dVdy, dvdy); eq.plans(dVdz, dvdz)
    for n in 1:3
        @. dVdx[n] = U[1]*dVdx[n] + U[2]*dVdy[n] + U[3]*dVdz[n]
    end
    dVdz .= 0
    for i in 1:3
        @. dVdz[1] -= V[i]*dUdx[i]
        @. dVdz[2] -= V[i]*dUdy[i]
        @. dVdz[3] -= V[i]*dUdz[i]
    end
    eq.plans(out, dVdx, add=true); eq.plans(out, dVdz, add=true)

    eq.force(out, v, AdjointContinuous())
    return out
end

# discrete adjoint LNSE
function (eq::CartesianPrimitive3DLNSE{AdjointDiscrete})(::Real,
                                                         v::VectorField{3, F},
                                                       out::VectorField{3, F}) where {F<:FTField}
    dudx = eq.scache[1]; u1v  = eq.scache[2]; u2v  = eq.scache[3]; u3v  = eq.scache[4]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]
    V    = eq.pcache[5]; U1V  = eq.pcache[6]; U2V  = eq.pcache[7]; U3V  = eq.pcache[8]

    requests = init_requests!(v)
    init_laplacian!(out, v; adjoint=true)
    eq.plans(V, v)
    wait_requests!(requests)
    complete_laplacian!(out, v; adjoint=true)
    out .*= 1/eq.Re

    for n in 1:3
        @. U1V[n] = U[1]*V[n]
        @. U2V[n] = U[2]*V[n]
        @. U3V[n] = U[3]*V[n]
    end
    eq.plans(u1v, U1V); eq.plans(u2v, U2V); eq.plans(u3v, U3V)

    eq.plans(dUdx, dudx)
    for n in 1:3
        r1 = init_requests!(u1v[n])
        r2 = init_requests!(u2v[n])
        r3 = init_requests!(u3v[n])
        init_ddx!(dudx[1], u1v[n]; adjoint=true)
        init_ddy!(dudx[2], u2v[n]; adjoint=true)
        init_ddz!(dudx[3], u3v[n]; adjoint=true)
        wait_requests!(r1); wait_requests!(r2); wait_requests!(r3)
        complete_ddx!(dudx[1], u1v[n]; adjoint=true)
        complete_ddy!(dudx[2], u2v[n]; adjoint=true)
        complete_ddz!(dudx[3], u3v[n]; adjoint=true)
        out[n] .-= dudx[1] .+ dudx[2] .+ dudx[3]
    end
    U1V .= 0
    for n in 1:3
        @. U1V[1] -= V[n]*dUdx[n]
        @. U1V[2] -= V[n]*dUdy[n]
        @. U1V[3] -= V[n]*dUdz[n]
    end
    eq.plans(out, U1V, add=true)

    eq.force(out, v, AdjointDiscrete())
    return out
end
