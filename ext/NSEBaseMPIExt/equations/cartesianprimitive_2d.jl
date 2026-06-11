# CartesianPrimitive2D call methods for decomposed grids.
#
# Constructors are not overridden here: NSEBase's constructors call FTField(g)
# and Field(g), which dispatch to the NSEBaseMPIExt overrides and allocate halo
# storage when needed.
#
# The call methods below group each numerical operation into one native Julia
# task. MPI requests are posted before the @sync block and passed to each
# @async task so computation overlaps communication:
#
#   requests = haloswap!(u, false)
#   @sync begin
#       @async NSEBase.ddx!(dudx, u, requests)
#       @async eq.plans(U, u)
#   end
#
# Dispatch selects these methods whenever the velocity fields are
# VectorField{N, FTField{<:DecomposedGrid}}.

# ------------------------------------------------------------------ #
# 2D nonlinear operator                                               #
# ------------------------------------------------------------------ #

function (eq::NSEBase.CartesianPrimitive2DNSE)(   ::Real,
                                                 u::NSEBase.VectorField{2, F},
                                               out::NSEBase.VectorField{2, F}) where {F<:NSEBase.FTField{<:DecomposedGrid}}
    # Reuse the equation-owned caches: lowercase fields are spectral, while
    # uppercase fields hold the corresponding physical-space values.
    dudx = eq.scache[1]; dudy = eq.scache[2]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]

    # Post halo communication once, then overlap every operation that can make
    # progress independently: diffusion, directional derivatives, and IFFT(u).
    @sync begin
        requests = haloswap!(u, false)
        @async (NSEBase.laplacian!(out, u, requests); out .*= 1/eq.Re)
        @async NSEBase.ddx!(dudx, u, requests)
        @async NSEBase.ddy!(dudy, u, requests)
        # Owned inverse-transform results do not require freshly exchanged halo
        # values; halo rows are merely independent transform batches.
        @async eq.plans(U, u)
    end

    # Evaluate the nonlinear advection term in physical space, reusing dUdx as
    # its destination: dUdx = -(U dot grad) U.
    eq.plans(dUdx, dudx); eq.plans(dUdy, dudy)
    for n in 1:2
        @. dUdx[n] = -U[1]*dUdx[n] - U[2]*dUdy[n]
    end

    # Add the transformed nonlinear term to diffusion, then apply any
    # equation-specific spectral forcing.
    eq.plans(out, dUdx, add=true)
    eq.force(out, u, NSEBase.Forward())
    return out
end

# ------------------------------------------------------------------ #
# 2D linearised operators                                             #
# ------------------------------------------------------------------ #

# 3-arg: compute and cache base-flow derivatives, then delegate to 2-arg.
function (eq::NSEBase.CartesianPrimitive2DLNSE)(   ::Real,
                                                  u::NSEBase.VectorField{2, F},
                                                  v::NSEBase.VectorField{2, F},
                                                out::NSEBase.VectorField{2, F}) where {F<:NSEBase.FTField{<:DecomposedGrid}}
    # Prime the base-flow caches used by each two-argument linear operator.
    dudx = eq.scache[1]; dudy = eq.scache[2]
    U    = eq.pcache[1]; dUdy = eq.pcache[3]

    # Differentiate and transform the base flow while its halo exchange is in
    # flight. dudx stays spectral until a selected linear operator needs it.
    @sync begin
        requests = haloswap!(u, false)
        @async NSEBase.ddx!(dudx, u, requests)
        @async NSEBase.ddy!(dudy, u, requests)
        @async eq.plans(U, u)
    end
    eq.plans(dUdy, dudy)

    # Delegate the perturbation evaluation after the shared base-flow state has
    # been prepared.
    eq(0, v, out)
    return out
end

function (eq::NSEBase.CartesianPrimitive2DLNSE{NSEBase.Forward})(   ::Real,
                                                                   v::NSEBase.VectorField{2, F},
                                                                 out::NSEBase.VectorField{2, F}) where {F<:NSEBase.FTField{<:DecomposedGrid}}
    # Lowercase caches are spectral derivatives; uppercase caches are the
    # physical base flow, perturbation, and their derivatives.
    dudx = eq.scache[1]; dvdx = eq.scache[2]; dvdy = eq.scache[3]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]
    V    = eq.pcache[4]; dVdx = eq.pcache[5]; dVdy = eq.pcache[6]

    # Overlap perturbation diffusion, its directional derivatives, and IFFT(v)
    # behind one non-blocking halo exchange.
    @sync begin
        requests = haloswap!(v, false)
        @async (NSEBase.laplacian!(out, v, requests); out .*= 1/eq.Re)
        @async NSEBase.ddx!(dvdx, v, requests)
        @async NSEBase.ddy!(dvdy, v, requests)
        @async eq.plans(V, v)
    end

    # Transform the derivatives needed to evaluate the linearized advection in
    # physical space. The base y derivative was prepared by the three-argument
    # method; its x derivative is transformed here.
    eq.plans(dUdx, dudx)
    eq.plans(dVdx, dvdx); eq.plans(dVdy, dvdy)
    for n in 1:2
        # dVdx is reused as -(U dot grad)V - (V dot grad)U.
        @. dVdx[n]  = -U[1]*dVdx[n] - U[2]*dVdy[n]
        @. dVdx[n] -=  V[1]*dUdx[n] + V[2]*dUdy[n]
    end

    # Add the transformed advection term to diffusion and apply forward forcing.
    eq.plans(out, dVdx, add=true)
    eq.force(out, v, NSEBase.Forward())
    return out
end

function (eq::NSEBase.CartesianPrimitive2DLNSE{NSEBase.AdjointContinuous})(   ::Real,
                                                                             v::NSEBase.VectorField{2, F},
                                                                           out::NSEBase.VectorField{2, F}) where {F<:NSEBase.FTField{<:DecomposedGrid}}
    # Bind the same cache layout as the forward operator so all variants share
    # allocations owned by the NSEBase equation object.
    dudx = eq.scache[1]; dvdx = eq.scache[2]; dvdy = eq.scache[3]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]
    V    = eq.pcache[4]; dVdx = eq.pcache[5]; dVdy = eq.pcache[6]

    # Diffusion, perturbation derivatives, and IFFT(v) are mutually independent
    # until their halo-dependent rows are completed.
    @sync begin
        requests = haloswap!(v, false)
        @async (NSEBase.laplacian!(out, v, requests); out .*= 1/eq.Re)
        @async NSEBase.ddx!(dvdx, v, requests)
        @async NSEBase.ddy!(dvdy, v, requests)
        @async eq.plans(V, v)
    end
    eq.plans(dUdx, dudx)
    eq.plans(dVdx, dvdx); eq.plans(dVdy, dvdy)
    for n in 1:2
        # The continuous adjoint of advection contributes +(U dot grad)V.
        @. dVdx[n] = U[1]*dVdx[n] + U[2]*dVdy[n]
    end

    # Reuse dVdy for the pointwise contraction -(grad U)^T V.
    dVdy .= 0
    for i in 1:2
        @. dVdy[1] -= V[i]*dUdx[i]
        @. dVdy[2] -= V[i]*dUdy[i]
    end

    # Both physical-space contributions are transformed and added to diffusion.
    eq.plans(out, dVdx, add=true); eq.plans(out, dVdy, add=true)
    eq.force(out, v, NSEBase.AdjointContinuous())
    return out
end

function (eq::NSEBase.CartesianPrimitive2DLNSE{NSEBase.AdjointDiscrete})(   ::Real,
                                                                           v::NSEBase.VectorField{2, F},
                                                                         out::NSEBase.VectorField{2, F}) where {F<:NSEBase.FTField{<:DecomposedGrid}}
    # u1v and u2v hold spectral transforms of the physical products U_i * V.
    # Their physical counterparts are reused from equation-owned cache fields.
    dudx = eq.scache[1]; u1v  = eq.scache[2]; u2v  = eq.scache[3]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]
    V    = eq.pcache[4]; U1V  = eq.pcache[5]; U2V  = eq.pcache[6]

    # The discrete-adjoint diffusion uses adjoint FD matrices. Owned IFFT(v)
    # values do not depend on freshly exchanged halos, so it can run concurrently.
    @sync begin
        requests = haloswap!(v, false)
        @async (NSEBase.laplacian!(out, v, requests; adjoint=true); out .*= 1/eq.Re)
        @async eq.plans(V, v)
    end

    # Form U_i * V component-wise in physical space, then transform the products
    # so discrete adjoint derivatives act on the represented discrete operator.
    for n in 1:2
        @. U1V[n] = U[1]*V[n]
        @. U2V[n] = U[2]*V[n]
    end
    eq.plans(u1v, U1V); eq.plans(u2v, U2V)

    eq.plans(dUdx, dudx)
    for n in 1:2
        # Exchange the two product fields independently, overlap their
        # directional adjoint derivatives, and subtract div(U * V_n) from
        # output component n.
        @sync begin
            requests_1 = haloswap!(u1v[n], false)
            requests_2 = haloswap!(u2v[n], false)
            @async NSEBase.ddx!(dudx[1], u1v[n], requests_1; adjoint=true)
            @async NSEBase.ddy!(dudx[2], u2v[n], requests_2; adjoint=true)
        end
        out[n] .-= dudx[1] .+ dudx[2]
    end

    # Add the remaining -(grad U)^T V contraction in physical space.
    U1V .= 0
    for n in 1:2
        @. U1V[1] -= V[n]*dUdx[n]
        @. U1V[2] -= V[n]*dUdy[n]
    end

    # Transform the contraction, add it to diffusion and divergence terms, and
    # finally apply the discrete-adjoint forcing.
    eq.plans(out, U1V, add=true)
    eq.force(out, v, NSEBase.AdjointDiscrete())
    return out
end
