# Boussinesq Navier-Stokes operators for 3D thermally-stratified flows.
#
# The state vector is (u, v, w, θ) ∈ VectorField{4}, where (u, v, w) are the
# three velocity components and θ is the temperature perturbation.
#
# Governing equations (non-dimensionalised with Re, Pr, Ri):
#
#   ∂u/∂t + (u·∇)u = -∇p + (1/Re)∇²u + Ri·θ·ê_grav + force
#   ∂θ/∂t + (u·∇)θ = (1/(Re·Pr))∇²θ
#
# where:
#   Ri = Ra/(Re²·Pr)   Richardson number (buoyancy coupling strength)
#   Pr                 Prandtl number
#   ê_grav             unit vector in the gravity direction (component `grav`)
#
# For Rayleigh-Bénard convection with no mean flow, set Re=1 and Ri=Ra/Pr.
#
# Cache layout (4 spectral, 8 physical VectorField{4}s shared between NSE+LNSE):
#
#   scache[1]  dudx          spectral ∂u/∂x (all 4 components)
#   scache[2]  dudy / u1v    spectral ∂u/∂y  /  U₁·v (physical→spectral)
#   scache[3]  dudz / u2v    spectral ∂u/∂z  /  U₂·v
#   scache[4]  dvdx / u3v    spectral ∂v/∂x  /  U₃·v
#   pcache[1]  U             physical base-flow state
#   pcache[2]  dUdx          physical ∂U/∂x
#   pcache[3]  dUdy          physical ∂U/∂y
#   pcache[4]  dUdz          physical ∂U/∂z
#   pcache[5]  V  / U1V      physical perturbation state  /  U₁·q (for adjoint)
#   pcache[6]  dVdx / U1V    physical ∂v/∂x
#   pcache[7]  dVdy / U2V    physical ∂v/∂y
#   pcache[8]  dVdz / U3V    physical ∂v/∂z


# ----------------------------------- #
# formulation tag                     #
# ----------------------------------- #
"""
    CartesianPrimitive3DBoussinesq

Tag selecting the four-component `(u, v, w, θ)` Boussinesq (thermally
stratified) NSE formulation.  Pass `CartesianPrimitive3DBoussinesq(Pr, Ri)`
to [`construct_equations`](@ref) to build
[`CartesianPrimitive3DBoussinesqNSE`](@ref) and
[`CartesianPrimitive3DBoussinesqLNSE`](@ref) operators.

The temperature field occupies component 4 of the state `VectorField`; velocity
components 1–3 follow the standard `CartesianPrimitive3D` convention.

# Fields
- `Pr`: Prandtl number.
- `Ri`: Richardson number `Ra/(Re²·Pr)`.  Set to zero for passive-scalar transport.
- `grav`: velocity component index for the gravity direction. Default `2` (wall-normal).

# Example
```julia
f  = CartesianPrimitive3DBoussinesq(0.71, 100.0)   # air, Ri = Ra/(Re²·Pr)
eq = construct_equations(grid, Re, (nothing, nothing, nothing, Θ), f)
```
"""
struct CartesianPrimitive3DBoussinesq
    Pr   :: Float64
    Ri   :: Float64
    grav :: Int
end
CartesianPrimitive3DBoussinesq(Pr, Ri; grav::Int=2) =
    CartesianPrimitive3DBoussinesq(Float64(Pr), Float64(Ri), grav)


# ----------------------------------- #
# concrete 3D Boussinesq NSE struct   #
# ----------------------------------- #
"""
    CartesianPrimitive3DBoussinesqNSE{T, FFT, S, P, BF}

Nonlinear Navier-Stokes operator for 3D Boussinesq flows.

State vector `(u, v, w, θ)`: velocity components 1–3 and temperature (component 4).

Evaluates:
- `out[1:3] = Δu/Re − (u·∇)u + Ri·θ·ê_grav + force`
- `out[4]   = Δθ/(Re·Pr) − (u·∇)θ`

# Fields
- `Re`, `Pr`, `Ri`: Reynolds number, Prandtl number, Richardson number (`Ra/(Re²·Pr)`)
- `grav`: velocity component index for the gravity direction (default 2 = wall-normal)
- `plans`, `scache`, `pcache`, `force`: as for [`CartesianPrimitive3DNSE`](@ref)
"""
mutable struct CartesianPrimitive3DBoussinesqNSE{T, FFT, S, P, BF}
              Re :: T
              Pr :: T
              Ri :: T
            grav :: Int
    const plans  :: FFT
    const scache :: Vector{VectorField{4, S}}
    const pcache :: Vector{VectorField{4, P}}
    const  force :: BF
end

# ----------------------------------- #
# concrete 3D Boussinesq LNSE struct  #
# ----------------------------------- #
"""
    CartesianPrimitive3DBoussinesqLNSE{MODE, T, FFT, S, P, BF}

Linearised Navier-Stokes operator for 3D Boussinesq flows, parameterised on
`MODE <: Mode`.

The three-argument call `eq(t, u, v, out)` caches physical base-flow gradients
from `u` (VectorField{4}) and then delegates to the two-argument form which
applies the chosen linearised operator to the perturbation `v` (VectorField{4}).

# Fields
- Same as [`CartesianPrimitive3DBoussinesqNSE`](@ref)
"""
mutable struct CartesianPrimitive3DBoussinesqLNSE{MODE, T, FFT, S, P, BF}
              Re :: T
              Pr :: T
              Ri :: T
            grav :: Int
    const plans  :: FFT
    const scache :: Vector{VectorField{4, S}}
    const pcache :: Vector{VectorField{4, P}}
    const  force :: BF

    CartesianPrimitive3DBoussinesqLNSE{MODE}(Re::T,
                                             Pr::T,
                                             Ri::T,
                                           grav::Int,
                                          plans::FFT,
                                         scache::Vector{VectorField{4, S}},
                                         pcache::Vector{VectorField{4, P}},
                                          force::BF) where {MODE, T, FFT, S, P, BF} =
        new{MODE, T, FFT, S, P, BF}(Re, Pr, Ri, grav, plans, scache, pcache, force)
end


# ------------- #
# nonlinear NSE #
# ------------- #
function (eq::CartesianPrimitive3DBoussinesqNSE)(::Real,
                                                  u::VectorField{4, F},
                                                out::VectorField{4, F}) where {F<:FTField}
    dudx = eq.scache[1]; dudy = eq.scache[2]; dudz = eq.scache[3]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]

    laplacian!(out, u)
    for n in 1:3; out[n] .*= 1/eq.Re; end
    out[4] .*= 1/(eq.Re * eq.Pr)

    ddx!(dudx, u)
    ddy!(dudy, u)
    ddz!(dudz, u)

    eq.plans(U, u); eq.plans(dUdx, dudx); eq.plans(dUdy, dudy); eq.plans(dUdz, dudz)
    for n in 1:4
        @. dUdx[n] = -U[1]*dUdx[n] - U[2]*dUdy[n] - U[3]*dUdz[n]
    end
    eq.plans(out, dUdx, add=true)

    out[eq.grav] .+= eq.Ri .* u[4]

    eq.force(out, u, Forward())
    return out
end


# -------------- #
# linearised NSE #
# -------------- #
# 3-arg: cache base-flow physical gradients then delegate to 2-arg
function (eq::CartesianPrimitive3DBoussinesqLNSE)(::Real,
                                                   u::VectorField{4, F},
                                                   v::VectorField{4, F},
                                                 out::VectorField{4, F}) where {F<:FTField}
    dudx = eq.scache[1]; dudy = eq.scache[2]; dudz = eq.scache[3]
    U    = eq.pcache[1]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]

    ddx!(dudx, u); ddy!(dudy, u); ddz!(dudz, u)
    eq.plans(U, u); eq.plans(dUdy, dudy); eq.plans(dUdz, dudz)

    eq(0, v, out)
    return out
end

# forward linearised: L·v
function (eq::CartesianPrimitive3DBoussinesqLNSE{Forward})(::Real,
                                                            v::VectorField{4, F},
                                                          out::VectorField{4, F}) where {F<:FTField}
    dudx = eq.scache[1]; dvdx = eq.scache[2]; dvdy = eq.scache[3]; dvdz = eq.scache[4]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]
    V    = eq.pcache[5]; dVdx = eq.pcache[6]; dVdy = eq.pcache[7]; dVdz = eq.pcache[8]

    laplacian!(out, v)
    for n in 1:3; out[n] .*= 1/eq.Re; end
    out[4] .*= 1/(eq.Re * eq.Pr)

    ddx!(dvdx, v); ddy!(dvdy, v); ddz!(dvdz, v)

    eq.plans(V, v); eq.plans(dUdx, dudx)
    eq.plans(dVdx, dvdx); eq.plans(dVdy, dvdy); eq.plans(dVdz, dvdz)

    # base-flow advection of all 4 perturbation components
    for n in 1:4
        @. dVdx[n]  = -U[1]*dVdx[n] - U[2]*dVdy[n] - U[3]*dVdz[n]
    end
    # velocity perturbation V[1:3] advects base-flow velocity and temperature
    for n in 1:4
        @. dVdx[n] -=  V[1]*dUdx[n] + V[2]*dUdy[n] + V[3]*dUdz[n]
    end
    eq.plans(out, dVdx, add=true)

    out[eq.grav] .+= eq.Ri .* v[4]

    eq.force(out, v, Forward())
    return out
end

# continuous adjoint: L†_cont · v
#
# Adjoint of the forward Boussinesq LNSE under the L² (continuous) inner product.
# Key differences from the velocity-only 3D adjoint:
#   - Cross-gradient sum runs over all 4 components (including Θ coupling to velocity)
#   - Buoyancy adjoint: out[4] += Ri * v[grav]
function (eq::CartesianPrimitive3DBoussinesqLNSE{AdjointContinuous})(::Real,
                                                                       v::VectorField{4, F},
                                                                     out::VectorField{4, F}) where {F<:FTField}
    dudx = eq.scache[1]; dvdx = eq.scache[2]; dvdy = eq.scache[3]; dvdz = eq.scache[4]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]
    V    = eq.pcache[5]; dVdx = eq.pcache[6]; dVdy = eq.pcache[7]; dVdz = eq.pcache[8]

    laplacian!(out, v)
    for n in 1:3; out[n] .*= 1/eq.Re; end
    out[4] .*= 1/(eq.Re * eq.Pr)

    ddx!(dvdx, v); ddy!(dvdy, v); ddz!(dvdz, v)

    eq.plans(V, v); eq.plans(dUdx, dudx)
    eq.plans(dVdx, dvdx); eq.plans(dVdy, dvdy); eq.plans(dVdz, dvdz)

    # +(U·∇)q for all 4 adjoint components (+ sign for continuous adjoint)
    for n in 1:4
        @. dVdx[n] = U[1]*dVdx[n] + U[2]*dVdy[n] + U[3]*dVdz[n]
    end
    eq.plans(out, dVdx, add=true)

    # Adjoint of base-flow gradient term: out[j] -= Σₙ V[n]*dU[n]/dxⱼ  (j=1,2,3)
    # Summation over all 4 components includes the Θ-coupling to velocity adjoint.
    dVdz .= 0
    for i in 1:4
        @. dVdz[1] -= V[i]*dUdx[i]
        @. dVdz[2] -= V[i]*dUdy[i]
        @. dVdz[3] -= V[i]*dUdz[i]
    end
    # dVdz[4] stays zero: no cross-gradient contribution to temperature adjoint.
    eq.plans(out, dVdz, add=true)

    # Buoyancy adjoint: temperature adjoint receives Ri * adjoint velocity at grav
    out[4] .+= eq.Ri .* v[eq.grav]

    eq.force(out, v, AdjointContinuous())
    return out
end

# discrete adjoint: L†_disc · v
#
# Exact transpose of the forward LNSE in the discrete inner product.
# Cross-gradient summation over n=1:4 ensures the Θ-coupling
# (temperature perturbation advecting base temperature) is correctly transposed.
function (eq::CartesianPrimitive3DBoussinesqLNSE{AdjointDiscrete})(::Real,
                                                                     v::VectorField{4, F},
                                                                   out::VectorField{4, F}) where {F<:FTField}
    dudx = eq.scache[1]; u1v  = eq.scache[2]; u2v  = eq.scache[3]; u3v  = eq.scache[4]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]
    V    = eq.pcache[5]; U1V  = eq.pcache[6]; U2V  = eq.pcache[7]; U3V  = eq.pcache[8]

    laplacian!(out, v, adjoint=true)
    for n in 1:3; out[n] .*= 1/eq.Re; end
    out[4] .*= 1/(eq.Re * eq.Pr)

    eq.plans(V, v)
    for n in 1:4
        @. U1V[n] = U[1]*V[n]
        @. U2V[n] = U[2]*V[n]
        @. U3V[n] = U[3]*V[n]
    end
    eq.plans(u1v, U1V); eq.plans(u2v, U2V); eq.plans(u3v, U3V)

    eq.plans(dUdx, dudx)
    for n in 1:4
        out[n] .-= ddx!(dudx[1], u1v[n], adjoint=true) .+
                   ddy!(dudx[2], u2v[n], adjoint=true) .+
                   ddz!(dudx[3], u3v[n], adjoint=true)
    end

    # Adjoint of velocity-perturbation-advects-base-state term.
    # Sum over all 4 base-state components; result only populates velocity adjoint [1:3].
    U1V .= 0
    for n in 1:4
        @. U1V[1] -= V[n]*dUdx[n]
        @. U1V[2] -= V[n]*dUdy[n]
        @. U1V[3] -= V[n]*dUdz[n]
    end
    # U1V[4] stays zero: no cross-gradient term in the temperature adjoint equation.
    eq.plans(out, U1V, add=true)

    # Buoyancy adjoint (exact transpose of `out[grav] += Ri * v[4]`)
    out[4] .+= eq.Ri .* v[eq.grav]

    eq.force(out, v, AdjointDiscrete())
    return out
end


# ------------------------------------------------------------------ #
# Specialised construct_equations for CartesianPrimitive3DBoussinesq #
# ------------------------------------------------------------------ #
"""
    construct_equations(grid, Re, base, f::CartesianPrimitive3DBoussinesq;
                        force=NoForce(), mode=AdjointDiscrete(),
                        flags=FFTW.EXHAUSTIVE, dealias=true)

Construct a [`ProjectedNSE`](@ref) for 3D Boussinesq (thermally stratified) flow.

The state vector is `(u, v, w, θ)` — three velocity components plus temperature.
Physical parameters (Pr, Ri, and the gravity direction) are carried by the
formulation tag `f`; see [`CartesianPrimitive3DBoussinesq`](@ref).

# Base flow
`base` should be a 4-tuple `(U₁, U₂, U₃, Θ)` where each entry is either a
vector of values at the inhomogeneous grid points or `nothing` for a zero
base-flow component.  For pure Rayleigh-Bénard (no mean velocity) pass
`(nothing, nothing, nothing, Θ_profile)`.

# Cache layout
Four spectral-space and eight physical-space `VectorField{4}` scratch arrays
are allocated and shared between the nonlinear and linearised operators.
"""
function construct_equations(grid   :: AbstractGrid{T},
                               Re,
                             base,
                                f  :: CartesianPrimitive3DBoussinesq;
                            force  = NoForce(),
                             mode  = AdjointDiscrete(),
                            flags  = FFTW.EXHAUSTIVE,
                          dealias  = true) where {T}
    mode isa Union{AdjointContinuous, AdjointDiscrete} ||
        throw(ArgumentError("linearised operator has to operate in adjoint mode"))
    plans  = FFTPlans(grid; flags=flags, dealias)
    scache = [VectorField([FTField(grid)                  for _ in 1:4]...) for _ in 1:4]
    pcache = [VectorField([  Field(grid; dealias=dealias) for _ in 1:4]...) for _ in 1:8]
    Pr, Ri, grav = T(f.Pr), T(f.Ri), f.grav
    nl = CartesianPrimitive3DBoussinesqNSE(T(Re), Pr, Ri, grav, plans, scache, pcache, force)
    ln = CartesianPrimitive3DBoussinesqLNSE{typeof(mode)}(T(Re), Pr, Ri, grav, plans, scache, pcache, force)
    return ProjectedNSE(grid, 4, nl, ln, base)
end
