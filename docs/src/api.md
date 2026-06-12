# API Reference

## Grid interface

```@docs
AbstractGrid
GridDecomposition
Undecomposed
Decomposed
decomposition_dims
ndecomposed_dims
fft_dims
spatial_fft_dims
inhomogeneous_dims
spatial_inhomogeneous_dims
to_storage_order
transform_size
fft_norm
growto
points
wavenumber_scale
weights
```

## Wavenumber indexing

```@docs
WaveNumberVector
to_homogeneous_indices
to_wavenumber_vector
```

## Field types

```@docs
Field
FTField
VectorField
ProjectedField
grid
modes
```

## Transforms

```@docs
FFTPlans
FFT
IFFT
```

## Derivatives

```@docs
ddx!
ddx_1!
ddx_2!
ddx_3!
ddx_4!
inhomogeneous_laplacian!
add_homogeneous_laplacian!
laplacian!
```

## Shifts

```@docs
shift!
shift
```

## Inner products and norms

```@docs
NSEBase.dot
NSEBase.norm
normdiff
minnormdiff
```

## Galerkin projection

```@docs
LoopGalerkin
GemmGalerkin
project!
project
expand!
expand
```

## Weighting

```@docs
FarazmandWeight
```

## Base-flow utilities

```@docs
add_base_flow!
```

## NSE formulations

```@docs
CartesianPrimitive3D
CartesianPrimitive2D
CartesianPrimitive2D3C
PolarPrimitive
ncomp
cache_length
nonlinear_operator
linearised_operator
construct_equations
```

## Operators

```@docs
AbstractNSE
AbstractLNSE
CartesianPrimitiveNSE
CartesianPrimitiveLNSE
CartesianPrimitive3DNSE
CartesianPrimitive3DLNSE
CartesianPrimitive2DNSE
CartesianPrimitive2DLNSE
CartesianPrimitive2D3CNSE
CartesianPrimitive2D3CLNSE
ProjectedNSE
```

## Mode tags

```@docs
Mode
Forward
AdjointContinuous
AdjointDiscrete
NoForce
CompoundForcing
```

## IO

```@docs
save_grid
load_grid
save_field
load_field
```
