# Convenience aliases for field wrappers on decomposed grids.

# Scalar fields.
const DecomposedField            = NSEBase.Field{<:DecomposedGrid}
const DecomposedFTField          = NSEBase.FTField{<:DecomposedGrid}
const DecomposedProjectedField   = NSEBase.ProjectedField{<:DecomposedGrid}
const DecomposedScalarField      = Union{DecomposedField, DecomposedFTField}

# Vector fields.
const DecomposedVectorField{N}   = NSEBase.VectorField{N, <:DecomposedScalarField}
const DecomposedFTVectorField{N} = NSEBase.VectorField{N, <:DecomposedFTField}

# Operation groups.
const DecomposedSpectralField    = Union{DecomposedFTField, DecomposedFTVectorField}
