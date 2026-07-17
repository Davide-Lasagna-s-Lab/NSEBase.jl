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

"""
    Forward <: Mode

Tag selecting the forward linearised operator L·v, i.e. the linearisation of
the NSE around the current base flow in the forward-time direction.
"""
struct Forward           <: Mode end

"""
    AdjointDiscrete <: Mode

Tag selecting the discrete adjoint of the forward linearised operator.
The discrete adjoint is derived by transposing the discrete operator exactly —
it satisfies ⟨L·v, w⟩ = ⟨v, L*·w⟩ with respect to the discrete inner product
used in [`dot`](@ref).
"""
struct AdjointDiscrete   <: Mode end

"""
    AdjointContinuous <: Mode

Tag selecting the continuous adjoint of the linearised operator.  The
continuous adjoint is derived by integration-by-parts before discretisation,
which gives a different operator from the discrete adjoint for finite
resolution.
"""
struct AdjointContinuous <: Mode end
