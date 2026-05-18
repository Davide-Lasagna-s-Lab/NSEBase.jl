# Shared equation-mode types and the default no-op body-force functor.
# These are generic across all NSE formulations (Cartesian, polar, …).

# ---------- #
# mode types #
# ---------- #
abstract type               Mode end
struct Forward           <: Mode end
struct AdjointDiscrete   <: Mode end
struct AdjointContinuous <: Mode end


# -------------------------- #
# default body force (no-op) #
# -------------------------- #
struct NoForce end
(::NoForce)(out, _, _) = out
