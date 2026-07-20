# Equation-mode tags used to select concrete NSE operators.
#
# Mode tags control which flavour of the linearised Navier-Stokes operator is
# applied.  Each concrete NSE/LNSE operator is parameterised on a mode type so
# that the compiler generates specialised, zero-overhead method bodies for each
# adjoint variant rather than branching at runtime.

# ---------- #
# mode types #
# ---------- #
"""
    Mode

Abstract supertype for equation-mode tags.  Concrete subtypes select which
variant of the linearised Navier-Stokes operator is applied.
"""
abstract type Mode end

@doc raw"""
    Forward <: Mode

Select the forward LNSE action ``L_U q`` at the cached base state ``U``. It is
the Fréchet derivative of the nonlinear spatial residual when the selected
force policy supplies its own linearised action. For a Cartesian velocity
equation the advective part is

```math
(L_Uq)_i = -\sum_j U_jD_jq_i-\sum_jq_jD_jU_i.
```

Diffusion, formulation-specific couplings, and the mode-selected force action
are added by the concrete operator.
"""
struct Forward           <: Mode end

@doc raw"""
    AdjointDiscrete <: Mode

Select the exact adjoint of the *discretised* forward linearised operator. If
``L_h`` denotes the numerical map applied by `Forward()`, this mode applies the
unique map ``L_h^\dagger`` satisfying

```math
\langle L_hq,p\rangle_h=\langle q,L_h^\dagger p\rangle_h
```

up to floating-point roundoff, with the same discrete inner product as
[`dot`](@ref). For an `FTField`, that inner product is

```math
\langle a,b\rangle_h=
\sum_n\sum_{\boldsymbol k}c_{k_r}\sum_{\boldsymbol j}w_{\boldsymbol j}
\operatorname{Re}\!\left(\overline{a_{n,\boldsymbol j,\boldsymbol k}}
                                  b_{n,\boldsymbol j,\boldsymbol k}\right).
```

Here ``n`` is the state component, ``\boldsymbol j`` indexes the
finite-difference directions, ``w_{\boldsymbol j}`` is the product quadrature
weight, and ``c_{k_r}`` is one on the rFFT zero plane and two on stored positive
rFFT modes. The forward-transform normalisation supplies the average over the
periodic directions; the quadrature is an integral, not an average, over the
inhomogeneous directions. In matrix notation, if ``W_h`` represents these
weights, then

```math
L_h^\dagger=W_h^{-1}L_h^{\mathrm H}W_h.
```

The conjugate transpose is understood on the Hermitian spectral subspace that
represents real physical fields. NSEBase never assembles ``W_h`` or this full
matrix. Instead, the implementation reverses the forward computational graph
and applies the weighted adjoint of each primitive:

1. A Fourier derivative ``D_j=\mathrm i k_j`` becomes
   ``D_j^\dagger=-\mathrm i k_j``. A finite-difference derivative uses the
   grid's stored weighted matrix ``D_j^+=W_j^{-1}D_j^{\mathsf T}W_j``.
   [`laplacian!`](@ref) similarly uses the stored adjoints of the
   inhomogeneous second derivatives; its Fourier contribution is self-adjoint.
2. Let ``B`` denote inverse transformation to the optionally padded physical
   grid, and let ``F`` denote the normalised forward transformation followed by
   spectral truncation. Parseval's identity with the rFFT multiplicities gives
   ``B^\dagger=F``; zero-extension and truncation are also an adjoint pair.
   Consequently the numerical product
   ``M_f=F\,\operatorname{diag}(f)B`` satisfies ``M_f^\dagger=M_f`` for a real
   cached physical field ``f``, including when 3/2-rule dealiasing is enabled.
3. Consequently, the forward base-advection term
   ``-M_{U_j}D_jq_i`` is transposed as
   ``-D_j^\dagger(M_{U_j}p_i)``. The order is important: in general it is not
   ``M_{U_j}D_j^\dagger p_i``.
4. The term in which a perturbation advects the base,
   ``-q_jD_jU_i``, contributes
   ``-\sum_i p_iD_jU_i`` to adjoint component ``j``. Boussinesq operators sum
   over temperature as well as velocity, so velocity receives the transpose of
   perturbation advection of the base temperature. Buoyancy
   ``Ri\,\theta\,\boldsymbol e_g`` transposes to ``Ri\,p_g`` in the temperature
   equation.
5. In [`ProjectedNSE`](@ref), [`project!`](@ref) is the weighted adjoint of
   [`expand!`](@ref), so the reduced discrete adjoint is
   ``P L_h^\dagger E`` for expansion ``E`` and projection ``P=E^\dagger``.

These rules can be seen directly from one base-advection term:

```math
\left\langle-M_{U_j}D_jq_i,p_i\right\rangle_h
=\left\langle q_i,-D_j^\dagger M_{U_j}p_i\right\rangle_h,
```

and from one perturbation-advection term:

```math
\sum_i\left\langle-M_{D_jU_i}q_j,p_i\right\rangle_h
=\left\langle q_j,-\sum_iM_{D_jU_i}p_i\right\rangle_h.
```

More generally, suppose a state has ``s`` components, its first ``m``
components advect along derivatives ``D_1,\ldots,D_m``, and component ``n``
diffuses with coefficient ``c_n``. Excluding formulation-specific linear
couplings and forcing, the forward action and its implemented transpose are

```math
\begin{aligned}
(L_Qq)_n
 &=c_n\nabla_h^2q_n-
   \sum_{j=1}^m\left(M_{Q_j}D_jq_n+M_{D_jQ_n}q_j\right),\\
(L_{h,Q}^\dagger p)_n
 &=c_n(\nabla_h^2)^\dagger p_n-
   \sum_{j=1}^mD_j^\dagger M_{Q_j}p_n-
   \mathbf 1_{n\leq m}\sum_{r=1}^sM_{D_nQ_r}p_r.
\end{aligned}
```

The second cross-gradient sum runs over *all* state components. This matters
for Boussinesq flow: it transposes velocity perturbations advecting the base
temperature. Its separate buoyancy map
``q_\theta\mapsto Ri\,q_\theta\boldsymbol e_g`` transposes to
``p\mapsto Ri\,p_g`` in the temperature equation.

The hydrodynamic part therefore is the exact discrete transpose at finite
resolution and for nonuniform quadrature. A custom force is part of that claim
only if its `AdjointDiscrete()` method is itself the discrete adjoint of its
`Forward()` method. If a policy adds a state-independent body source during an
LNSE call, that call is affine rather than linear and no full-operator adjoint
identity exists; the identity above then applies only to the linear
hydrodynamic part. See the forcing interface accepted by
[`construct_equations`](@ref).
"""
struct AdjointDiscrete   <: Mode end

@doc raw"""
    AdjointContinuous <: Mode

Select the continuous adjoint PDE, discretised after integration by parts. For
a divergence-free Cartesian base velocity ``U`` and boundary conditions that
remove the boundary terms, the advective part is

```math
(L_U^{\dagger,c}p)_i=\sum_jU_j\partial_jp_i
                      -\sum_n p_n\partial_iU_n.
```

The implementation uses the ordinary derivative and Laplacian operators in
this formula. It therefore need not equal [`AdjointDiscrete`](@ref) at finite
resolution, near boundaries, or when the supplied quadrature and derivative
matrices do not obey the continuous integration-by-parts identities exactly.
A force policy's `AdjointContinuous()` action belongs to this formula only if
the policy implements the continuous adjoint of its linear `Forward()` action;
the mode tag cannot establish that contract for an arbitrary callable.
"""
struct AdjointContinuous <: Mode end
