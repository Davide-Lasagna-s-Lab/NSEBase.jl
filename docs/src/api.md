# API Reference

## Grid interface

```@docs
AbstractGrid
storage_dim
physical_dim
physical_to_storage_dim
to_storage_order
rfft_storage_dim
rfft_physical_dim
fft_storage_dims
fft_physical_dims
spatial_fft_storage_dims
spatial_fft_physical_dims
inhomogeneous_storage_dims
inhomogeneous_physical_dims
spatial_inhomogeneous_storage_dims
spatial_inhomogeneous_physical_dims
transform_size
fft_norm
growto
points
wavenumber_scale
weights
NSEBase.derivative_matrix
```

Distributed-grid construction and GPU adaptation are supplied by NSEBase's MPI
and CUDA extensions, respectively; load the corresponding packages before using
those extension APIs.

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
dd!
ddx!
ddy!
ddz!
ddt!
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
CartesianPrimitive3DBoussinesq
PolarPrimitive
ncomp
cache_length
nonlinear_operator
linearised_operator
construct_equations
```

## Operators

```@docs
CartesianPrimitive3DNSE
CartesianPrimitive3DLNSE
CartesianPrimitive2DNSE
CartesianPrimitive2DLNSE
CartesianPrimitive2D3CNSE
CartesianPrimitive2D3CLNSE
CartesianPrimitive3DBoussinesqNSE
CartesianPrimitive3DBoussinesqLNSE
ProjectedNSE
```

## Mode tags

```@docs
Mode
OperatorMode
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
