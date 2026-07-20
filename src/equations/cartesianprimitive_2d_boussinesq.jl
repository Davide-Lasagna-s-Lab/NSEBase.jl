# Boussinesq Navier–Stokes operators for two-dimensional thermally stratified flows.
#
# This file is the exact two-dimensional reduction of `cartesianprimitive_3d_boussinesq.jl`.
# Physical coordinates retain their Cartesian meaning: `x` is horizontal/streamwise, `y` is
# vertical/wall-normal, and the state is `(u,v,θ)`. The first two components are velocity and the
# third is temperature. No spanwise velocity or z derivative is present.
#
# With q=(u,v,θ), velocity U=(u,v), and gravity component `grav`, the nonlinear equations are
#
#   ∂U/∂t + (U·∇)U = -∇p + ∇²U/Re + Ri θ e_grav + force,
#   ∂θ/∂t + (U·∇)θ = ∇²θ/(Re Pr).
#
# The linearised operators include both couplings introduced by temperature: temperature
# perturbations generate buoyancy, and velocity perturbations advect the base temperature. The
# discrete adjoint below is the exact weighted transpose of the forward discretisation, including
# the transpose of both couplings.
#
# Three spectral and six physical `VectorField{3}` caches are shared by the nonlinear and
# linearised operators. Their aliases deliberately match `cartesianprimitive_2d.jl`:
#
#   scache[1]  dudx                          pcache[1]  U
#   scache[2]  dudy / dvdx / u1v             pcache[2]  dUdx
#   scache[3]         dvdy / u2v             pcache[3]  dUdy
#                                              pcache[4]  V
#                                              pcache[5]  dVdx / U1V
#                                              pcache[6]  dVdy / U2V

# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Physical-plane contract                                                                    // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

function _check_cartesian_2d_boussinesq_grid(::AbstractGrid{<:Any, <:Any, AXES}) where {AXES}
    x_dim, y_dim, z_dim = AXES[1], AXES[2], AXES[3]
    valid = !isnothing(x_dim) && !isnothing(y_dim) && isnothing(z_dim) && x_dim != y_dim
    valid || throw(ArgumentError("CartesianPrimitive2DBoussinesq requires a physical (x,y) " *
                                 "grid with z absent; got AXES=$AXES"))
    return nothing
end

# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Formulation tag                                                                            // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

@doc raw"""
    CartesianPrimitive2DBoussinesq(Pr, Ri; grav=2)

Select the three-component two-dimensional Boussinesq formulation with state `(u,v,θ)`, where
`u` and `v` are the physical `x`- and `y`-velocity components and `θ` is temperature. Use this tag
with [`construct_equations`](@ref), or use [`RayleighBenardFlow`](@ref) to obtain the standard
channel case with its conduction base state and Rayleigh-number scaling.

# Parameters

- `Pr` is the Prandtl number. Temperature diffuses with coefficient `1/(Re*Pr)`, while momentum
  diffuses with coefficient `1/Re`.
- `Ri` is the Richardson number multiplying buoyancy. [`RayleighBenardFlow`](@ref) computes
  `Ri=Ra/(Re^2*Pr)`; `Ri=0` leaves a passive temperature field.
- `grav` is the velocity component receiving buoyancy. It must be `1` or `2` and defaults to `2`,
  the physical wall-normal direction of a [`TwoDimensionalChannelGrid`](@ref).

# State and base order

Both spectral states and base tuples follow `(u,v,θ)` / `(U,V,Θ)` order. A `nothing` entry in the
base tuple denotes a zero component, so motionless conduction is represented by
`(nothing,nothing,Θ)`.

# Equations

Let ``Q=(Q_1,Q_2,Q_3)=(U,V,\Theta)``, ``\nu=Re^{-1}``, and
``\kappa=(Re\,Pr)^{-1}``. The formulation selects the pressure-free residual

```math
\begin{aligned}
\mathcal N_1(Q)&=\nu\nabla_h^2U-U\partial_xU-V\partial_yU
                  +Ri\,\delta_{1g}\Theta+\mathcal F_{N,1}(Q),\\
\mathcal N_2(Q)&=\nu\nabla_h^2V-U\partial_xV-V\partial_yV
                  +Ri\,\delta_{2g}\Theta+\mathcal F_{N,2}(Q),\\
\mathcal N_3(Q)&=\kappa\nabla_h^2\Theta-U\partial_x\Theta-V\partial_y\Theta
                  +\mathcal F_{N,3}(Q),
\end{aligned}
```

where ``g=\mathtt{grav}``. Pressure is handled only when these equations are
expanded and projected through an admissible [`ProjectedNSE`](@ref) basis.

# Example

```julia
grid = TwoDimensionalChannelGrid(63, 65; α=0.5, width=7)
Θ = rbc_base_temperature(grid)
formulation = CartesianPrimitive2DBoussinesq(0.71, 1000 / 0.71)
equations = construct_equations(grid, 1.0, (nothing, nothing, Θ), formulation;
                                flags=FFTW.ESTIMATE)
```
"""
struct CartesianPrimitive2DBoussinesq
    Pr::Float64
    Ri::Float64
    grav::Int

    function CartesianPrimitive2DBoussinesq(Pr::Float64, Ri::Float64, grav::Int)
        grav in 1:2 || throw(ArgumentError("the 2D gravity component must be 1 or 2"))
        return new(Pr, Ri, grav)
    end
end

CartesianPrimitive2DBoussinesq(Pr, Ri; grav::Int=2) =
    CartesianPrimitive2DBoussinesq(Float64(Pr), Float64(Ri), grav)

# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Nonlinear operator                                                                         // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

@doc raw"""
    CartesianPrimitive2DBoussinesqNSE{T, FFT, S, P, BF}
    CartesianPrimitive2DBoussinesqNSE(grid, Re, formulation;
                                      force=NoForce(), flags=FFTW.EXHAUSTIVE, dealias=true)

Nonlinear primitive-variable Boussinesq operator for a two-dimensional Cartesian state
`q=(u,v,θ)`.

Writing ``Q=(Q_1,Q_2,Q_3)=(U,V,\Theta)``, ``\nu=Re^{-1}``, and
``\kappa=(Re\,Pr)^{-1}``, the call `eq(t,Q,out)` evaluates

```math
\begin{aligned}
\mathcal N_n(Q)&=c_n\nabla_h^2Q_n-U\partial_xQ_n-V\partial_yQ_n
                 +Ri\,\delta_{n,g}\Theta+\mathcal F_{N,n}(Q),
                 \qquad n=1,2,3,\\
c_1&=c_2=\nu,\qquad c_3=\kappa,
\end{aligned}
```

where ``g=\mathtt{grav}\in\{1,2\}``; hence the buoyancy term vanishes from
the temperature equation. ``\mathcal F_N`` denotes the actual
`force(out,Q,Forward())` action, which may act on temperature as well as
momentum. Products are evaluated in physical space with optional dealiased
padding. The argument `t` is ignored, and a transformed logical `t` coordinate
is excluded from the spatial derivatives and Laplacian. This raw operator has
no pressure term; projection through a divergence-free basis handles it.

The object is stateful and reuses its `scache` and `pcache` workspaces. Construct it through
[`construct_equations`](@ref) or [`RayleighBenardFlow`](@ref), which allocate compatible caches
and FFT plans.

# Fields

- `Re`, `Pr`, and `Ri`: Reynolds, Prandtl, and Richardson numbers.
- `grav`: velocity component receiving `Ri*θ`.
- `plans`: forward and inverse transforms, shared with the linearised operator when the two are
  built together by [`construct_equations`](@ref).
- `scache`, `pcache`: preallocated spectral and physical `VectorField{3}` workspaces.
- `force`: callable body-force policy applied after diffusion, advection, and buoyancy.
"""
mutable struct CartesianPrimitive2DBoussinesqNSE{T, FFT, S, P, BF}
    Re::T
    Pr::T
    Ri::T
    grav::Int
    const plans::FFT
    const scache::Vector{VectorField{3, S}}
    const pcache::Vector{VectorField{3, P}}
    const force::BF

    function CartesianPrimitive2DBoussinesqNSE(Re::T, Pr::T, Ri::T, grav::Int, plans::FFT,
                                                scache::Vector{VectorField{3, S}},
                                                pcache::Vector{VectorField{3, P}},
                                                force::BF) where {T, FFT, S, P, BF}
        _check_cartesian_2d_boussinesq_grid(grid(scache[1][1]))
        grav in 1:2 || throw(ArgumentError("the 2D gravity component must be 1 or 2"))
        return new{T, FFT, S, P, BF}(Re, Pr, Ri, grav, plans, scache, pcache, force)
    end
end

function CartesianPrimitive2DBoussinesqNSE(g::G, Re, f::CartesianPrimitive2DBoussinesq;
                                            force::BF=NoForce(), flags=FFTW.EXHAUSTIVE,
                                            dealias=true) where {T, G<:AbstractGrid{T}, BF}
    _check_cartesian_2d_boussinesq_grid(g)
    plans = FFTPlans(g; flags, dealias)
    scache = [VectorField([FTField(g) for _ in 1:3]...) for _ in 1:2]
    pcache = [VectorField([Field(g; dealias) for _ in 1:3]...) for _ in 1:3]
    return CartesianPrimitive2DBoussinesqNSE(T(Re), T(f.Pr), T(f.Ri), f.grav,
                                             plans, scache, pcache, force)
end

# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Linearised operator                                                                        // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

@doc raw"""
    CartesianPrimitive2DBoussinesqLNSE{MODE, T, FFT, S, P, BF}
    CartesianPrimitive2DBoussinesqLNSE(grid, Re, formulation;
                                       mode=AdjointDiscrete(), force=NoForce(),
                                       flags=FFTW.EXHAUSTIVE, dealias=true)

Forward or adjoint linearisation of [`CartesianPrimitive2DBoussinesqNSE`](@ref) about a base state
`Q=(U,V,Θ)`. `MODE` is [`Forward`](@ref), [`AdjointContinuous`](@ref), or
[`AdjointDiscrete`](@ref).

The four-argument call `eq(t,Q,q,out)` transforms and caches `Q` and its derivatives before
applying the selected operator to `q`. Once the base cache is prepared, `eq(t,q,out)` may be used
repeatedly for perturbations about that same base. A later four-argument call replaces the cache.

Let ``Q=(Q_1,Q_2,Q_3)=(U,V,\Theta)``, ``q=(u,v,\theta)``,
``p=(a,b,\phi)``, ``(D_1,D_2)=(\partial_x,\partial_y)``,
``c_1=c_2=Re^{-1}``, and ``c_3=(Re\,Pr)^{-1}``. The forward action is

```math
(L_Qq)_n=c_n\nabla_h^2q_n-U D_1q_n-V D_2q_n-uD_1Q_n-vD_2Q_n
          +Ri\,\delta_{n,g}\theta+\mathcal F_{F,n}(q),\qquad n=1,2,3,
```

where ``g=\mathtt{grav}``. For `MODE == AdjointContinuous`, integration by
parts before discretisation gives

```math
\begin{aligned}
(L_Q^{\dagger,c}p)_1
 &=c_1\nabla^2a+U\partial_xa+V\partial_ya
   -\sum_{n=1}^3p_n\partial_xQ_n+\mathcal F_{C,1}(p),\\
(L_Q^{\dagger,c}p)_2
 &=c_2\nabla^2b+U\partial_xb+V\partial_yb
   -\sum_{n=1}^3p_n\partial_yQ_n+\mathcal F_{C,2}(p),\\
(L_Q^{\dagger,c}p)_3
 &=c_3\nabla^2\phi+U\partial_x\phi+V\partial_y\phi
   +Ri\,p_g+\mathcal F_{C,3}(p).
\end{aligned}
```

Thus the velocity equations contain the transpose of perturbation advection
of both base velocity and base temperature, while the temperature equation
receives ``Ri\,p_g``, the transpose of buoyancy. This continuous form assumes
``\partial_xU+\partial_yV=0`` and boundary conditions that remove the
integration-by-parts terms.

For `MODE == AdjointDiscrete`, let ``M_f`` denote the configured numerical
multiplication by real ``f``, including padding and truncation when dealiasing
is enabled, and let ``\dagger`` denote the weighted adjoint under [`dot`](@ref).
The code applies

```math
\begin{aligned}
(L_{h,Q}^{\dagger}p)_1
 &=c_1(\nabla_h^2)^\dagger a-D_1^\dagger M_Ua-D_2^\dagger M_Va
   -\sum_{n=1}^3M_{D_1Q_n}p_n+\mathcal F_{D,1}(p),\\
(L_{h,Q}^{\dagger}p)_2
 &=c_2(\nabla_h^2)^\dagger b-D_1^\dagger M_Ub-D_2^\dagger M_Vb
   -\sum_{n=1}^3M_{D_2Q_n}p_n+\mathcal F_{D,2}(p),\\
(L_{h,Q}^{\dagger}p)_3
 &=c_3(\nabla_h^2)^\dagger\phi-D_1^\dagger M_U\phi-D_2^\dagger M_V\phi
   +Ri\,p_g+\mathcal F_{D,3}(p).
\end{aligned}
```

The order ``-D_j^\dagger M_{Q_j}`` is the reverse of the forward
``-M_{Q_j}D_j`` path. It includes the stored finite-difference adjoints and the
adjoint FFT padding/truncation pipeline, so the hydrodynamic and buoyancy core
is the exact weighted transpose at finite resolution. See
[`AdjointDiscrete`](@ref) for the derivation. A force is part of this identity
only when its `AdjointDiscrete()` action transposes its `Forward()` action; an
affine source is not a linear operator and has no adjoint.

This object owns mutable scratch storage and is neither reentrant nor safe for concurrent calls.
Construct compatible instances through [`construct_equations`](@ref) or [`RayleighBenardFlow`](@ref).
"""
mutable struct CartesianPrimitive2DBoussinesqLNSE{MODE, T, FFT, S, P, BF}
    Re::T
    Pr::T
    Ri::T
    grav::Int
    const plans::FFT
    const scache::Vector{VectorField{3, S}}
    const pcache::Vector{VectorField{3, P}}
    const force::BF

    CartesianPrimitive2DBoussinesqLNSE{MODE}(Re::T, Pr::T, Ri::T, grav::Int, plans::FFT,
                                             scache::Vector{VectorField{3, S}},
                                             pcache::Vector{VectorField{3, P}},
                                             force::BF) where {MODE, T, FFT, S, P, BF} = begin
        _check_cartesian_2d_boussinesq_grid(grid(scache[1][1]))
        grav in 1:2 || throw(ArgumentError("the 2D gravity component must be 1 or 2"))
        new{MODE, T, FFT, S, P, BF}(Re, Pr, Ri, grav, plans, scache, pcache, force)
    end
end

function CartesianPrimitive2DBoussinesqLNSE(g::G, Re, f::CartesianPrimitive2DBoussinesq;
                                             mode::Mode=AdjointDiscrete(), force::BF=NoForce(),
                                             flags=FFTW.EXHAUSTIVE, dealias=true) where {T, G<:AbstractGrid{T}, BF}
    _check_cartesian_2d_boussinesq_grid(g)
    plans = FFTPlans(g; flags, dealias)
    scache = [VectorField([FTField(g) for _ in 1:3]...) for _ in 1:3]
    pcache = [VectorField([Field(g; dealias) for _ in 1:3]...) for _ in 1:6]
    return CartesianPrimitive2DBoussinesqLNSE{typeof(mode)}(
        T(Re), T(f.Pr), T(f.Ri), f.grav, plans, scache, pcache, force)
end

# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Nonlinear action                                                                           // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

function (eq::CartesianPrimitive2DBoussinesqNSE)(::Real, u::VectorField{3, F},
                                                  out::VectorField{3, F}) where {F<:FTField}
    dudx = eq.scache[1]; dudy = eq.scache[2]
    U = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]

    laplacian!(out, u)
    for n in 1:2; out[n] .*= 1 / eq.Re; end
    out[3] .*= 1 / (eq.Re * eq.Pr)

    ddx!(dudx, u); ddy!(dudy, u)
    eq.plans(U, u); eq.plans(dUdx, dudx); eq.plans(dUdy, dudy)
    for n in 1:3
        @. dUdx[n] = -U[1] * dUdx[n] - U[2] * dUdy[n]
    end
    eq.plans(out, dUdx, add=true)

    out[eq.grav] .+= eq.Ri .* u[3]
    eq.force(out, u, Forward())
    return out
end

# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Base-state cache                                                                           // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

function (eq::CartesianPrimitive2DBoussinesqLNSE)(::Real, u::VectorField{3, F},
                                                   v::VectorField{3, F},
                                                   out::VectorField{3, F}) where {F<:FTField}
    dudx = eq.scache[1]; dudy = eq.scache[2]
    U = eq.pcache[1]; dUdy = eq.pcache[3]

    ddx!(dudx, u); ddy!(dudy, u)
    eq.plans(U, u); eq.plans(dUdy, dudy)
    eq(0, v, out)
    return out
end

# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Forward linearisation                                                                      // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

function (eq::CartesianPrimitive2DBoussinesqLNSE{Forward})(::Real, v::VectorField{3, F},
                                                            out::VectorField{3, F}) where {F<:FTField}
    dudx = eq.scache[1]; dvdx = eq.scache[2]; dvdy = eq.scache[3]
    U = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]
    V = eq.pcache[4]; dVdx = eq.pcache[5]; dVdy = eq.pcache[6]

    laplacian!(out, v)
    for n in 1:2; out[n] .*= 1 / eq.Re; end
    out[3] .*= 1 / (eq.Re * eq.Pr)

    ddx!(dvdx, v); ddy!(dvdy, v)
    eq.plans(V, v); eq.plans(dUdx, dudx); eq.plans(dVdx, dvdx); eq.plans(dVdy, dvdy)
    for n in 1:3
        @. dVdx[n] = -U[1] * dVdx[n] - U[2] * dVdy[n]
        @. dVdx[n] -= V[1] * dUdx[n] + V[2] * dUdy[n]
    end
    eq.plans(out, dVdx, add=true)

    out[eq.grav] .+= eq.Ri .* v[3]
    eq.force(out, v, Forward())
    return out
end

# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Continuous adjoint                                                                         // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

function (eq::CartesianPrimitive2DBoussinesqLNSE{AdjointContinuous})(::Real,
                                                                      v::VectorField{3, F},
                                                                      out::VectorField{3, F}) where {F<:FTField}
    dudx = eq.scache[1]; dvdx = eq.scache[2]; dvdy = eq.scache[3]
    U = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]
    V = eq.pcache[4]; dVdx = eq.pcache[5]; dVdy = eq.pcache[6]

    laplacian!(out, v)
    for n in 1:2; out[n] .*= 1 / eq.Re; end
    out[3] .*= 1 / (eq.Re * eq.Pr)

    ddx!(dvdx, v); ddy!(dvdy, v)
    eq.plans(V, v); eq.plans(dUdx, dudx); eq.plans(dVdx, dvdx); eq.plans(dVdy, dvdy)
    for n in 1:3
        @. dVdx[n] = U[1] * dVdx[n] + U[2] * dVdy[n]
    end
    eq.plans(out, dVdx, add=true)

    # The transpose of perturbation advection populates velocity only. Summing over component
    # three is essential: it is the adjoint of velocity perturbations advecting base temperature.
    dVdy .= 0
    for n in 1:3
        @. dVdy[1] -= V[n] * dUdx[n]
        @. dVdy[2] -= V[n] * dUdy[n]
    end
    eq.plans(out, dVdy, add=true)

    out[3] .+= eq.Ri .* v[eq.grav]
    eq.force(out, v, AdjointContinuous())
    return out
end

# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Discrete adjoint                                                                           // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

function (eq::CartesianPrimitive2DBoussinesqLNSE{AdjointDiscrete})(::Real,
                                                                    v::VectorField{3, F},
                                                                    out::VectorField{3, F}) where {F<:FTField}
    dudx = eq.scache[1]; u1v = eq.scache[2]; u2v = eq.scache[3]
    U = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]
    V = eq.pcache[4]; U1V = eq.pcache[5]; U2V = eq.pcache[6]

    laplacian!(out, v; adjoint=true)
    for n in 1:2; out[n] .*= 1 / eq.Re; end
    out[3] .*= 1 / (eq.Re * eq.Pr)

    eq.plans(V, v)
    for n in 1:3
        @. U1V[n] = U[1] * V[n]
        @. U2V[n] = U[2] * V[n]
    end
    eq.plans(u1v, U1V); eq.plans(u2v, U2V); eq.plans(dUdx, dudx)
    for n in 1:3
        ddx!(dudx[1], u1v[n]; adjoint=true); ddy!(dudx[2], u2v[n]; adjoint=true)
        out[n] .-= dudx[1] .+ dudx[2]
    end

    # This physical multiplication is self-adjoint. Component three remains zero because the
    # forward cross-gradient term depends only on the two velocity components of a perturbation.
    U1V .= 0
    for n in 1:3
        @. U1V[1] -= V[n] * dUdx[n]
        @. U1V[2] -= V[n] * dUdy[n]
    end
    eq.plans(out, U1V, add=true)

    out[3] .+= eq.Ri .* v[eq.grav]
    eq.force(out, v, AdjointDiscrete())
    return out
end

# //////////////////////////////////////////////////////////////////////////////////////////////// #
# // Equation construction                                                                      // #
# //////////////////////////////////////////////////////////////////////////////////////////////// #

@doc raw"""
    construct_equations(grid, Re, base, formulation::CartesianPrimitive2DBoussinesq;
                        force=NoForce(), mode=AdjointDiscrete(), flags=FFTW.EXHAUSTIVE,
                        dealias=true) -> ProjectedNSE

Construct nonlinear and adjoint-linearised two-dimensional Boussinesq operators and wrap them in
a three-component [`ProjectedNSE`](@ref).

# Projected equations

Let ``E`` expand reduced coefficients into the primitive state, let
``P=E^\dagger`` be its weighted-adjoint projection, and let ``B`` be the base
state assembled from `base`. The returned nonlinear residual is

```math
R_h(a)=P\,\mathcal N_h(Ea+B).
```

If ``L_{h,Ea+B}`` is the forward operator documented by
[`CartesianPrimitive2DBoussinesqLNSE`](@ref), then the reduced forward and
discrete-adjoint actions are

```math
J_h(a)b=P\,L_{h,Ea+B}Eb,
\qquad
J_h(a)^\dagger c=P\,L_{h,Ea+B}^\dagger Ec.
```

For `mode=AdjointContinuous()`, the last expression instead uses the
discretised continuous-adjoint equations. Expansion and projection impose the
two-dimensional velocity constraint and remove pressure; temperature remains
the unconstrained third state component.

# Arguments

- `grid`: a physical `(x,y)` Cartesian grid with distinct `x` and `y` storage dimensions and `z`
  absent. The bundled Rayleigh–Bénard case uses [`TwoDimensionalChannelGrid`](@ref).
- `Re`: Reynolds number used by the momentum and temperature diffusivities.
- `base`: three entries in `(U,V,Θ)` order. Each entry is an inhomogeneous-grid array or `nothing`.
- `formulation`: [`CartesianPrimitive2DBoussinesq`](@ref), which supplies `Pr`, `Ri`, and `grav`.

# Keywords

- `force`: an optional callable body-force policy supporting `VectorField{3}`. Its
  `AdjointDiscrete()` action must be the weighted transpose of its linear `Forward()` action for
  the full reduced discrete-adjoint identity to hold.
- `mode`: [`AdjointDiscrete`](@ref) or [`AdjointContinuous`](@ref). The projected wrapper is an
  adjoint-oriented interface and rejects [`Forward`](@ref).
- `flags`: FFTW planning flags.
- `dealias`: whether nonlinear physical products use padded transforms.

The nonlinear and linearised operators share three spectral and six physical workspaces. The
returned object is stateful; see [`ProjectedNSE`](@ref) for the required nonlinear/linearised call
ordering.

# Example

```julia
grid = TwoDimensionalChannelGrid(31, 33; α=1, width=5)
Θ = rbc_base_temperature(grid)
formulation = CartesianPrimitive2DBoussinesq(0.71, 1e4 / 0.71)
equations = construct_equations(grid, 1, (nothing, nothing, Θ), formulation;
                                flags=FFTW.ESTIMATE, dealias=false)
```
"""
function construct_equations(grid::AbstractGrid{T}, Re, base,
                             f::CartesianPrimitive2DBoussinesq; force=NoForce(),
                             mode=AdjointDiscrete(), flags=FFTW.EXHAUSTIVE, dealias=true) where {T}
    _check_cartesian_2d_boussinesq_grid(grid)
    mode isa Union{AdjointContinuous, AdjointDiscrete} ||
        throw(ArgumentError("linearised operator has to operate in adjoint mode"))
    length(base) == 3 || throw(ArgumentError("a 2D Boussinesq base must contain (U, V, Θ)"))
    plans = FFTPlans(grid; flags, dealias)
    scache = [VectorField([FTField(grid) for _ in 1:3]...) for _ in 1:3]
    pcache = [VectorField([Field(grid; dealias) for _ in 1:3]...) for _ in 1:6]
    Pr, Ri, grav = T(f.Pr), T(f.Ri), f.grav
    nl = CartesianPrimitive2DBoussinesqNSE(T(Re), Pr, Ri, grav, plans, scache, pcache, force)
    ln = CartesianPrimitive2DBoussinesqLNSE{typeof(mode)}(T(Re), Pr, Ri, grav, plans, scache, pcache, force)
    return ProjectedNSE(grid, 3, nl, ln, base)
end
