# CartesianPrimitive3D call methods for decomposed grids.
#
# Constructors are inherited from NSEBase; the methods below overlap halo
# exchange with interior finite differences and local FFT work.

# ------------------------------------------------------------------ #
# 3D nonlinear operator                                               #
# ------------------------------------------------------------------ #

function (eq::NSEBase.CartesianPrimitive3DNSE)(   ::Real,
                                                 u::NSEBase.VectorField{3, F},
                                               out::NSEBase.VectorField{3, F}) where {F<:NSEBase.FTField{<:DecomposedGrid}}
    # Reuse the equation-owned caches: lowercase fields are spectral, while
    # uppercase fields hold the corresponding physical-space values.
    dudx = eq.scache[1]; dudy = eq.scache[2]; dudz = eq.scache[3]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]

    # Post halo communication once, then overlap diffusion, three directional
    # derivatives, and IFFT(u).
    #
    # TODO (perf): the shared request tuple waits for every component's halo
    # before any boundary work can start. A per-component request schedule could
    # let component n's boundary work begin as soon as its own halo lands, at the
    # cost of fanning 5 async tasks out to ~12. Defer until a profile shows it
    # matters.
    @sync begin
        requests = haloswap!(u, false)
        @async (NSEBase.laplacian!(out, u, requests); out .*= 1/eq.Re)
        @async NSEBase.ddx!(dudx, u, requests)
        @async NSEBase.ddy!(dudy, u, requests)
        @async NSEBase.ddz!(dudz, u, requests)
        # Owned inverse-transform results do not require freshly exchanged halo
        # values; halo rows are merely independent transform batches.
        @async eq.plans(U, u)
    end

    # Evaluate nonlinear advection in physical space, reusing dUdx as its
    # destination: dUdx = -(U dot grad) U.
    eq.plans(dUdx, dudx); eq.plans(dUdy, dudy); eq.plans(dUdz, dudz)
    for n in 1:3
        @. dUdx[n] = -U[1]*dUdx[n] - U[2]*dUdy[n] - U[3]*dUdz[n]
    end

    # Add the transformed nonlinear term to diffusion, then apply forcing.
    eq.plans(out, dUdx, add=true)
    eq.force(out, u, NSEBase.Forward())
    return out
end

# ------------------------------------------------------------------ #
# 3D linearised operators                                             #
# ------------------------------------------------------------------ #

# 3-arg: compute and cache base-flow derivatives, then delegate to 2-arg.
function (eq::NSEBase.CartesianPrimitive3DLNSE)(   ::Real,
                                                  u::NSEBase.VectorField{3, F},
                                                  v::NSEBase.VectorField{3, F},
                                                out::NSEBase.VectorField{3, F}) where { F<:NSEBase.FTField{<:DecomposedGrid}}
    # Prime the base-flow caches shared by the three-dimensional linear
    # operator variants.
    dudx = eq.scache[1]; dudy = eq.scache[2]; dudz = eq.scache[3]
    U    = eq.pcache[1]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]

    # Differentiate and transform the base flow while communication proceeds.
    # dudx remains spectral until a selected operator variant consumes it.
    @sync begin
        requests = haloswap!(u, false)
        @async NSEBase.ddx!(dudx, u, requests)
        @async NSEBase.ddy!(dudy, u, requests)
        @async NSEBase.ddz!(dudz, u, requests)
        @async eq.plans(U, u)
    end
    eq.plans(dUdy, dudy); eq.plans(dUdz, dudz)

    # Evaluate the selected perturbation operator after preparing shared state.
    eq(0, v, out)
    return out
end

function (eq::NSEBase.CartesianPrimitive3DLNSE{NSEBase.Forward})(   ::Real,
                                                                   v::NSEBase.VectorField{3, F},
                                                                 out::NSEBase.VectorField{3, F}) where {F<:NSEBase.FTField{<:DecomposedGrid}}
    # Lowercase caches are spectral derivatives; uppercase caches are the base
    # flow, perturbation, and their physical-space derivatives.
    dudx = eq.scache[1]; dvdx = eq.scache[2]; dvdy = eq.scache[3]; dvdz = eq.scache[4]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]
    V    = eq.pcache[5]; dVdx = eq.pcache[6]; dVdy = eq.pcache[7]; dVdz = eq.pcache[8]

    # Overlap perturbation diffusion, all directional derivatives, and IFFT(v)
    # behind one non-blocking halo exchange.
    @sync begin
        requests = haloswap!(v, false)
        @async (NSEBase.laplacian!(out, v, requests); out .*= 1/eq.Re)
        @async NSEBase.ddx!(dvdx, v, requests)
        @async NSEBase.ddy!(dvdy, v, requests)
        @async NSEBase.ddz!(dvdz, v, requests)
        @async eq.plans(V, v)
    end

    # Transform the derivatives needed for pointwise physical-space products.
    eq.plans(dUdx, dudx)
    eq.plans(dVdx, dvdx); eq.plans(dVdy, dvdy); eq.plans(dVdz, dvdz)
    for n in 1:3
        # dVdx is reused as -(U dot grad)V - (V dot grad)U.
        @. dVdx[n]  = -U[1]*dVdx[n] - U[2]*dVdy[n] - U[3]*dVdz[n]
        @. dVdx[n] -=  V[1]*dUdx[n] + V[2]*dUdy[n] + V[3]*dUdz[n]
    end

    # Add the transformed advection term to diffusion and apply forward forcing.
    eq.plans(out, dVdx, add=true)
    eq.force(out, v, NSEBase.Forward())
    return out
end

function (eq::NSEBase.CartesianPrimitive3DLNSE{NSEBase.AdjointContinuous})(   ::Real,
                                                                             v::NSEBase.VectorField{3, F},
                                                                            out::NSEBase.VectorField{3, F}) where {F<:NSEBase.FTField{<:DecomposedGrid}}
    # Bind the shared cache layout used by every linear operator variant.
    dudx = eq.scache[1]; dvdx = eq.scache[2]; dvdy = eq.scache[3]; dvdz = eq.scache[4]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]
    V    = eq.pcache[5]; dVdx = eq.pcache[6]; dVdy = eq.pcache[7]; dVdz = eq.pcache[8]

    # Diffusion, perturbation derivatives, and IFFT(v) can all progress while
    # their halo-dependent rows are completed.
    @sync begin
        requests = haloswap!(v, false)
        @async (NSEBase.laplacian!(out, v, requests); out .*= 1/eq.Re)
        @async NSEBase.ddx!(dvdx, v, requests)
        @async NSEBase.ddy!(dvdy, v, requests)
        @async NSEBase.ddz!(dvdz, v, requests)
        @async eq.plans(V, v)
    end
    eq.plans(dUdx, dudx)
    eq.plans(dVdx, dvdx); eq.plans(dVdy, dvdy); eq.plans(dVdz, dvdz)
    for n in 1:3
        # The continuous adjoint of advection contributes +(U dot grad)V.
        @. dVdx[n] = U[1]*dVdx[n] + U[2]*dVdy[n] + U[3]*dVdz[n]
    end

    # Reuse dVdz for the pointwise contraction -(grad U)^T V.
    dVdz .= 0
    for i in 1:3
        @. dVdz[1] -= V[i]*dUdx[i]
        @. dVdz[2] -= V[i]*dUdy[i]
        @. dVdz[3] -= V[i]*dUdz[i]
    end

    # Transform and accumulate both physical-space adjoint contributions.
    eq.plans(out, dVdx, add=true); eq.plans(out, dVdz, add=true)
    eq.force(out, v, NSEBase.AdjointContinuous())
    return out
end

function (eq::NSEBase.CartesianPrimitive3DLNSE{NSEBase.AdjointDiscrete})(    ::Real,
                                                                            v::NSEBase.VectorField{3, F},
                                                                          out::NSEBase.VectorField{3, F}) where {F<:NSEBase.FTField{<:DecomposedGrid}}
    # u1v, u2v, and u3v hold spectral transforms of U_i * V. Their uppercase
    # counterparts are reusable physical-space caches.
    dudx = eq.scache[1]; u1v  = eq.scache[2]; u2v  = eq.scache[3]; u3v  = eq.scache[4]
    U    = eq.pcache[1]; dUdx = eq.pcache[2]; dUdy = eq.pcache[3]; dUdz = eq.pcache[4]
    V    = eq.pcache[5]; U1V  = eq.pcache[6]; U2V  = eq.pcache[7]; U3V  = eq.pcache[8]

    # The discrete-adjoint diffusion uses adjoint FD matrices. Owned IFFT(v)
    # values do not depend on freshly exchanged halos, so it can run concurrently.
    @sync begin
        requests = haloswap!(v, false)
        @async (NSEBase.laplacian!(out, v, requests; adjoint=true); out .*= 1/eq.Re)
        @async eq.plans(V, v)
    end

    # Build physical products U_i * V_n and transform them before applying the
    # adjoints of the represented discrete directional derivatives.
    for n in 1:3
        @. U1V[n] = U[1]*V[n]
        @. U2V[n] = U[2]*V[n]
        @. U3V[n] = U[3]*V[n]
    end
    eq.plans(u1v, U1V); eq.plans(u2v, U2V); eq.plans(u3v, U3V)

    eq.plans(dUdx, dudx)
    for n in 1:3
        # Exchange the product fields independently, overlap their three
        # directional adjoint derivatives, and subtract div(U * V_n) from
        # component n.
        @sync begin
            requests_1 = haloswap!(u1v[n], false)
            requests_2 = haloswap!(u2v[n], false)
            requests_3 = haloswap!(u3v[n], false)
            @async NSEBase.ddx!(dudx[1], u1v[n], requests_1; adjoint=true)
            @async NSEBase.ddy!(dudx[2], u2v[n], requests_2; adjoint=true)
            @async NSEBase.ddz!(dudx[3], u3v[n], requests_3; adjoint=true)
        end
        out[n] .-= dudx[1] .+ dudx[2] .+ dudx[3]
    end

    # Add the remaining -(grad U)^T V contraction in physical space.
    U1V .= 0
    for n in 1:3
        @. U1V[1] -= V[n]*dUdx[n]
        @. U1V[2] -= V[n]*dUdy[n]
        @. U1V[3] -= V[n]*dUdz[n]
    end

    # Transform the contraction, accumulate it into output, and apply forcing.
    eq.plans(out, U1V, add=true)
    eq.force(out, v, NSEBase.AdjointDiscrete())
    return out
end
