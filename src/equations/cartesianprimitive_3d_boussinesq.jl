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
#   scache[4]  dvdz / u3v    spectral ∂v/∂z  /  U₃·v
#   pcache[1]  U             physical base-flow state
#   pcache[2]  dUdx          physical ∂U/∂x
#   pcache[3]  dUdy          physical ∂U/∂y
#   pcache[4]  dUdz          physical ∂U/∂z
#   pcache[5]  V             physical perturbation/adjoint state
#   pcache[6]  dVdx / U1V    physical ∂v/∂x
#   pcache[7]  dVdy / U2V    physical ∂v/∂y
#   pcache[8]  dVdz / U3V    physical ∂v/∂z


# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Formulation tag                                                                            // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

@doc raw"""
    CartesianPrimitive3DBoussinesq(Pr, Ri; grav=2)

Select the four-component three-dimensional Boussinesq formulation with state
`Q=(U₁,U₂,U₃,Θ)`, where `U=(U₁,U₂,U₃)` is velocity and `Θ` is temperature. Use this tag with
[`construct_equations`](@ref) to build the nonlinear and adjoint-linearised projected equations.

Before spatial discretisation, the dimensionless equations represented by this formulation are

```math
\begin{aligned}
\partial_t \boldsymbol U +(\boldsymbol U\!\cdot\!\nabla)\boldsymbol U
  &= -\nabla p + \frac{1}{Re}\nabla^2\boldsymbol U
     +Ri\,\Theta\,\boldsymbol e_{\mathrm{grav}}+\boldsymbol f_U,\\
\partial_t\Theta +(\boldsymbol U\!\cdot\!\nabla)\Theta
  &= \frac{1}{Re\,Pr}\nabla^2\Theta+f_\Theta,\\
\nabla\!\cdot\!\boldsymbol U&=0.
\end{aligned}
```

The primitive operators documented below evaluate the pressure-free right-hand sides. The
[`ProjectedNSE`](@ref) wrapper supplies the divergence-free expansion and projection through the
grid basis, so pressure is not stored as a fifth state component.

# Parameters

- `Pr` is the Prandtl number. Temperature diffuses with coefficient `1/(Re*Pr)`, while momentum
  diffuses with coefficient `1/Re`.
- `Ri` is the Richardson number multiplying buoyancy. Under the scaling used by the bundled
  Rayleigh–Bénard cases, `Ri=Ra/(Re^2*Pr)`; `Ri=0` leaves a passively advected scalar.
- `grav` selects the velocity component parallel to gravity and must be `1`, `2`, or `3`. It
  defaults to `2`, the usual wall-normal direction. This low-level tag stores the index without
  validating it, so callers constructing the tag directly are responsible for that constraint.

# State and base order

Spectral states and base tuples both follow `(u,v,w,θ)` / `(U₁,U₂,U₃,Θ)` order. A `nothing` entry
in a base tuple denotes a zero component; motionless thermal conduction can therefore be written
as `(nothing,nothing,nothing,Θ)`.

# Example

```julia
formulation = CartesianPrimitive3DBoussinesq(0.71, 100.0; grav=2)
equations = construct_equations(grid, Re, (nothing, nothing, nothing, Θ), formulation;
                                flags=FFTW.ESTIMATE)
```
"""
struct CartesianPrimitive3DBoussinesq
    Pr   :: Float64
    Ri   :: Float64
    grav :: Int
end
CartesianPrimitive3DBoussinesq(Pr, Ri; grav::Int=2) =
    CartesianPrimitive3DBoussinesq(Float64(Pr), Float64(Ri), grav)


# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Nonlinear operator                                                                         // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

@doc raw"""
    CartesianPrimitive3DBoussinesqNSE{T, FFT, S, P, BF}

Nonlinear primitive-variable Boussinesq operator for a three-dimensional Cartesian state
`Q=(U₁,U₂,U₃,Θ)`.

Let `D_j` and `Δ_h` denote the grid's discrete first derivatives and spatial Laplacian, and let
`M_h(a)b` denote multiplication in the physical workspace followed by the configured transform
back to spectral space, including padding and truncation when dealiasing is enabled. With

```math
c_n=\begin{cases}1/Re,&n=1,2,3,\\1/(Re\,Pr),&n=4,\end{cases}
```

the call `eq(t,Q,out)` evaluates

```math
\mathcal N_h(Q)_n
  =c_n\Delta_hQ_n-\sum_{j=1}^3M_h(Q_j)D_jQ_n
   +Ri\,\delta_{n,\mathrm{grav}}Q_4+\mathcal F^{\mathrm{Forward}}_n(Q),
\qquad n=1,\ldots,4,
```

where the buoyancy Kronecker delta is nonzero only for velocity components. Thus component four
is explicitly

```math
\mathcal N_h(Q)_4=\frac{1}{Re\,Pr}\Delta_h\Theta
                   -M_h(U_1)D_x\Theta-M_h(U_2)D_y\Theta-M_h(U_3)D_z\Theta
                   +\mathcal F^{\mathrm{Forward}}_4(Q).
```

No pressure gradient is evaluated here: projection in [`ProjectedNSE`](@ref) enforces the
velocity constraint. The leading `t` argument is accepted for the time-integrator interface and
is currently ignored. A transformed logical time coordinate, when present in a grid description,
is not included in `D_x,D_y,D_z` or `Δ_h`.

`force` is called after diffusion, advection, and buoyancy. Its `Forward()` action is therefore
part of the returned right-hand side, but its mathematical meaning is defined by the selected
force policy.

# Fields

- `Re`, `Pr`, and `Ri`: Reynolds, Prandtl, and Richardson numbers.
- `grav`: velocity component receiving `Ri*Θ`.
- `plans`: forward and inverse transforms used by `M_h`.
- `scache`, `pcache`: preallocated spectral and physical `VectorField{4}` workspaces.
- `force`: callable body-force policy.

The operator mutates its caches and `out`; one instance is neither reentrant nor safe for
concurrent calls.
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

# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Linearised operator                                                                        // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

@doc raw"""
    CartesianPrimitive3DBoussinesqLNSE{MODE, T, FFT, S, P, BF}

Forward, continuous-adjoint, or discrete-adjoint linearisation of
[`CartesianPrimitive3DBoussinesqNSE`](@ref) about a base state `Q=(U₁,U₂,U₃,Θ)`. `MODE` is
[`Forward`](@ref), [`AdjointContinuous`](@ref), or [`AdjointDiscrete`](@ref).

# Base-state cache contract

The call `eq(t,Q,q,out)` transforms and caches `Q` and `D_xQ,D_yQ,D_zQ`, then applies the selected
operator to `q`. Once primed, `eq(t,q,out)` may be called repeatedly for perturbations about the
same base; another four-argument call replaces the cached base. The leading `t` is accepted for
the time-integrator interface and ignored. These mutable caches make an instance non-reentrant
and unsafe for concurrent calls.

# Forward linearisation

Let `q=(u,v,w,θ)`, let `c₁=c₂=c₃=1/Re` and `c₄=1/(Re*Pr)`, and let `M_h` be the actual physical
multiplication/dealiasing map used by the FFT plans. The forward discrete action is

```math
(L_{h,Q}q)_n=c_n\Delta_hq_n
 -\sum_{j=1}^3\left[M_h(Q_j)D_jq_n+M_h(q_j)D_jQ_n\right]
 +Ri\,\delta_{n,\mathrm{grav}}q_4+\mathcal F^{\mathrm{Forward}}_n(q),
\qquad n=1,\ldots,4.
```

The equation with `n=4` includes `D_jQ₄=D_jΘ`: velocity perturbations therefore advect the
base temperature.

# Continuous adjoint

For adjoint state `p=(p₁,p₂,p₃,p₄)`, the implementation of the continuous `L²` adjoint is

```math
\begin{aligned}
(L_Q^{\dagger,c}p)_j
 &=\frac{1}{Re}\nabla^2p_j
   +\sum_{\ell=1}^3U_\ell\partial_\ell p_j
   -\sum_{n=1}^4p_n\partial_jQ_n
   +\mathcal F^{\dagger,c}_j(p), &&j=1,2,3,\\
(L_Q^{\dagger,c}p)_4
 &=\frac{1}{Re\,Pr}\nabla^2p_4
   +\sum_{\ell=1}^3U_\ell\partial_\ell p_4
   +Ri\,p_{\mathrm{grav}}+\mathcal F^{\dagger,c}_4(p).
\end{aligned}
```

The `n=4` term in the velocity sum is the transpose of perturbation advection of the base
temperature. The `Ri*p_grav` term is the transpose of buoyancy. This continuous formula assumes
`∇·U=0` and boundary conditions that remove the integration-by-parts boundary terms; otherwise
the adjoint of `-U·∇` also contains `(∇·U)p` and boundary contributions.

# Discrete weighted adjoint

For the discrete inner product represented by the grid quadrature, Fourier Hermitian
multiplicities, and component sum,

```math
\langle a,b\rangle_h=\sum_{n=1}^4\sum_{\boldsymbol k}\omega_{\boldsymbol k}
\operatorname{Re}\!\left(\overline{a_{n,\boldsymbol k}}b_{n,\boldsymbol k}\right),
\qquad L_{h,Q}^\dagger=W^{-1}L_{h,Q}^{H}W,
```

where `W` contains the weights `ω_k`. Writing `D_j^†` and `Δ_h^†` for the stored weighted
adjoints of the numerical derivative operators, the code evaluates

```math
\begin{aligned}
(L_{h,Q}^\dagger p)_j
 &=\frac{1}{Re}\Delta_h^\dagger p_j
   -\sum_{\ell=1}^3D_\ell^\dagger M_h(U_\ell)p_j
   -\sum_{n=1}^4M_h(D_jQ_n)p_n
   +\mathcal F^{\dagger,h}_j(p), &&j=1,2,3,\\
(L_{h,Q}^\dagger p)_4
 &=\frac{1}{Re\,Pr}\Delta_h^\dagger p_4
   -\sum_{\ell=1}^3D_\ell^\dagger M_h(U_\ell)p_4
   +Ri\,p_{\mathrm{grav}}+\mathcal F^{\dagger,h}_4(p).
\end{aligned}
```

This is computed by reversing the numerical graph of the forward action: physical products use
the same transform/dealiasing path, each derivative is replaced by its weighted adjoint, product
order is reversed, the base-gradient products are transposed component-by-component, and
buoyancy is transposed from velocity component `grav` into temperature component four. This is
why discrete base advection is `-D_j^†M_h(U_j)p`, not the pointwise continuous expression
`+M_h(U_j)D_jp`.

The equality `⟨p,Lq⟩_h=⟨L^†p,q⟩_h` includes forcing only when the force policy's
`AdjointDiscrete()` action is the weighted transpose of its `Forward()` linear action. An
input-independent body force is an affine nonlinear source, not a linear map, and should be
excluded from a homogeneous LNSE adjoint identity.

# Fields

The physical parameters, plans, caches, and force policy have the same meanings as in
[`CartesianPrimitive3DBoussinesqNSE`](@ref).
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


# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Nonlinear action                                                                           // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #
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


# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Base-state cache                                                                           // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #
# Four arguments: prepare the base-flow cache, then delegate to the cached three-argument action.
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

# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Forward linearisation                                                                      // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

# Forward linearised action: L·v.
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

# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Continuous adjoint                                                                         // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

# Continuous adjoint action: L†_cont·v.
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

# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Discrete adjoint                                                                           // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

# Discrete weighted-adjoint action: L†_disc·v.
#
# Exact transpose of the forward LNSE in the discrete inner product.
# Cross-gradient summation over n=1:4 ensures the Θ-coupling
# (velocity perturbations advecting base temperature) is correctly transposed.
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


# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Equation construction                                                                      // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

@doc raw"""
    construct_equations(grid, Re, base, f::CartesianPrimitive3DBoussinesq;
                        force=NoForce(), mode=AdjointDiscrete(),
                        flags=FFTW.EXHAUSTIVE, dealias=true) -> ProjectedNSE

Construct nonlinear and adjoint-linearised three-dimensional Boussinesq operators and wrap them
in a four-component [`ProjectedNSE`](@ref).

# Projected equations

Let `E` expand a reduced, divergence-free state into the primitive state, let `P=E^†` be its
weighted-adjoint projection, and let `B` be the primitive base state constructed from `base`. The
returned nonlinear residual is

```math
R_h(a)=P\,\mathcal N_h(Ea+B).
```

If `L_{h,Ea+B}` is the raw forward linearisation documented by
[`CartesianPrimitive3DBoussinesqLNSE`](@ref), its reduced forward and discrete-adjoint actions
are

```math
J_h(a)b=P\,L_{h,Ea+B}Eb,
\qquad
J_h(a)^\dagger c=P\,L_{h,Ea+B}^\dagger Ec.
```

For `mode=AdjointContinuous()`, the final expression instead uses the discretised continuous
adjoint `L_{Ea+B}^{†,c}`. Expansion and projection impose the velocity constraint and account for
pressure; temperature is carried as the fourth state component and is not divergence constrained.

# Arguments

- `grid`: a three-dimensional Cartesian grid whose physical spatial directions are `x`, `y`, and
  `z`. A transformed logical time coordinate is never included in the spatial derivatives.
- `Re`: Reynolds number used in both momentum and thermal diffusivities.
- `base`: four entries in `(U₁,U₂,U₃,Θ)` order. Each entry is an inhomogeneous-grid array or
  `nothing` for a zero component.
- `f`: [`CartesianPrimitive3DBoussinesq`](@ref), supplying `Pr`, `Ri`, and `grav`.

# Keywords

- `force`: callable force policy supporting `VectorField{4}` and the selected mode. Its adjoint
  action must be the weighted transpose of its forward linear action for the discrete-adjoint
  identity to include forcing.
- `mode`: [`AdjointDiscrete`](@ref) or [`AdjointContinuous`](@ref). This projected constructor is
  adjoint-oriented and rejects [`Forward`](@ref).
- `flags`: FFTW planning flags.
- `dealias`: whether physical products use padded transforms.

# Base flow

For motionless Rayleigh–Bénard conduction, pass `(nothing,nothing,nothing,Θ_profile)`. A velocity
base may be supplied in any or all of the first three entries; perturbation velocity then advects
both that velocity base and `Θ_profile`.

# Cache layout

Four spectral and eight physical `VectorField{4}` workspaces are allocated and shared by the
nonlinear and linearised operators. The returned object is stateful; see [`ProjectedNSE`](@ref)
for its nonlinear/linearised call ordering.

# Example

```julia
formulation = CartesianPrimitive3DBoussinesq(Pr, Ra / (Re^2 * Pr); grav=2)
equations = construct_equations(grid, Re, (nothing, nothing, nothing, Θ), formulation;
                                mode=AdjointDiscrete(), flags=FFTW.ESTIMATE)
```
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
    length(base) == 4 || throw(ArgumentError("a 3D Boussinesq base must contain (U, V, W, Θ)"))
    plans  = FFTPlans(grid; flags=flags, dealias)
    scache = [VectorField([FTField(grid)                  for _ in 1:4]...) for _ in 1:4]
    pcache = [VectorField([  Field(grid; dealias=dealias) for _ in 1:4]...) for _ in 1:8]
    Pr, Ri, grav = T(f.Pr), T(f.Ri), f.grav
    nl = CartesianPrimitive3DBoussinesqNSE(T(Re), Pr, Ri, grav, plans, scache, pcache, force)
    ln = CartesianPrimitive3DBoussinesqLNSE{typeof(mode)}(T(Re), Pr, Ri, grav, plans, scache, pcache, force)
    return ProjectedNSE(grid, 4, nl, ln, base)
end
