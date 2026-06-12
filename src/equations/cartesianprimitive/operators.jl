# Velocity-only Cartesian primitive Navier-Stokes operator.
#
# `NDIM` = number of spatial derivative directions, `NCOMP` = number of advected
# components, `MODE` = which equation (nonlinear or a linearised variant), `FORM`
# = advection form. A single struct covers every bundled case; the per-dimension
# public names are *bare* type aliases, which forward both construction and
# dispatch with no hand-written forwarding methods:
#
#   {2,2} planar 2D    {2,3} 2D-3C    {3,3} 3D    {3,4} 3D + scalar (Boussinesq)
#
# The call skeletons and the AbstractNSE supertype live in abstract_nse.jl; the
# form-dispatched advection helpers live in advection_*.jl.
#
# `visc` is the coefficient that multiplies the Laplacian (ν = 1/Re): a scalar
# applies uniformly; an NCOMP-tuple gives a per-component diffusivity (e.g. 1/Re
# for velocity, 1/(Re·Pr) for the Boussinesq temperature).


"""
    CartesianPrimitiveNSE{NDIM, NCOMP, MODE, FORM, V, FFT, S, P, BF}

Velocity-only Cartesian primitive Navier-Stokes operator. `NDIM`/`NCOMP` fix the
geometry, `MODE` selects the equation (`NonLinear`, `Forward`,
`AdjointContinuous`, `AdjointDiscrete`), `FORM` selects the advection form, and
`visc` is the per-component (or scalar) Laplacian coefficient ν = 1/Re.

Build with `CartesianPrimitiveNSE{NDIM,NCOMP}(g, Re; mode, form, …)` or via
[`construct_equations`](@ref).
"""
mutable struct CartesianPrimitiveNSE{NDIM, NCOMP, MODE<:Mode, FORM<:AdvectionForm, V, FFT, S, P, BF} <: AbstractNSE{MODE, FORM}
            visc::V
     const plans::FFT
    const scache::Vector{VectorField{NCOMP, S}}
    const pcache::Vector{VectorField{NCOMP, P}}
    const  force::BF

    CartesianPrimitiveNSE{NDIM, NCOMP, MODE, FORM}(visc::V, plans::FFT,
                                                   scache::Vector{VectorField{NCOMP, S}},
                                                   pcache::Vector{VectorField{NCOMP, P}},
                                                   force::BF) where {NDIM, NCOMP, MODE, FORM, V, FFT, S, P, BF} =
        new{NDIM, NCOMP, MODE, FORM, V, FFT, S, P, BF}(visc, plans, scache, pcache, force)
end

function CartesianPrimitiveNSE{NDIM, NCOMP}(g::AbstractGrid{T}, Re;
                                            mode::Mode          =NonLinear(),
                                            visc                =inv(convert(T, Re)),
                                            form::AdvectionForm =Advective(),
                                            force               =NoForce(),
                                            flags               =FFTW.EXHAUSTIVE,
                                            dealias             =true) where {NDIM, NCOMP, T}
    ((NDIM == 2 && (NCOMP == 2 || NCOMP == 3)) || (NDIM == 3 && (NCOMP == 3 || NCOMP == 4))) ||
        throw(ArgumentError("unsupported Cartesian primitive dimensions: NDIM=$NDIM, NCOMP=$NCOMP"))
    # The nonlinear operator needs fewer scratch fields than a linearised one
    # (which also caches base-flow quantities).
    ns, np = mode isa NonLinear ? (NDIM, NDIM + 1) : (NDIM + 1, 2 * (NDIM + 1))
    plans  = FFTPlans(g; flags=flags)
    scache = [VectorField(ntuple(_ -> FTField(g),                NCOMP)...) for _ in 1:ns]
    pcache = [VectorField(ntuple(_ -> Field(g; dealias=dealias), NCOMP)...) for _ in 1:np]
    return CartesianPrimitiveNSE{NDIM, NCOMP, typeof(mode), typeof(form)}(visc, plans, scache, pcache, force)
end


# --------------------- public aliases -------------------- #
const CartesianPrimitive2DNSE   = CartesianPrimitiveNSE{2, 2}
const CartesianPrimitive2D3CNSE = CartesianPrimitiveNSE{2, 3}
const CartesianPrimitive3DNSE   = CartesianPrimitiveNSE{3, 3}
