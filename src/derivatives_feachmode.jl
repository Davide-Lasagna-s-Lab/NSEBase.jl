# for_each_mode-based implementations of ddx! and add_homogeneous_laplacian!.
# Prototype on branch feature/for-each-mode-ddx.
# Not included in NSEBase.jl — standalone include from the test script.

"""
    ddx_fem!(out::FTField, u::FTField, ::Val{Dim}; adjoint=false)

`for_each_mode`-based prototype of `ddx!`.  Same semantics; exists only to
validate correctness against the `@generated` split-block implementation.
"""
@generated function ddx_fem!(out::FTField{G}, u::FTField{G}, ::Val{Dim};
                              adjoint::Bool=false) where {T, D, Axes, Hs, Ht,
                                                          G<:AbstractGrid{T, D, Axes, Hs, Ht}, Dim}
    H = (Hs..., Ht)
    Dim ∉ H && return :(throw(ArgumentError(lazy"ddx_fem!: dim $Dim not in H=$($(H))")))

    N      = length(H)
    h_syms = [Symbol("_ih", k) for k in 1:N]   # H-order closure arguments

    h_pos = findfirst(==(Dim), H)               # slot of Dim in H
    non_H = [d for d in 1:D if d ∉ H]

    # Map each array dim to the right variable: H-order symbol or non-spectral loop var.
    arr_idx = Vector{Any}(undef, D)
    for d in 1:D
        k = findfirst(==(d), H)
        arr_idx[d] = k !== nothing ? h_syms[k] : Symbol("_iy", d)
    end

    # Signed wavenumber for the differentiated dimension.
    # H[1] is rfft → always non-negative.  Others need a conditional (zero cost).
    hs   = h_syms[h_pos]
    wnum = Dim == H[1] ?
        :($hs - 1) :
        :($hs <= (Base.size(u, $Dim) >> 1) + 1 ? $hs - 1 : $hs - Base.size(u, $Dim) - 1)

    # Build: assignment, then wrap in non-spectral loops (lower dim = innermost).
    out_ref = Expr(:ref, :(parent(out)), arr_idx...)
    u_ref   = Expr(:ref, :(parent(u)),  arr_idx...)
    inner   = :(@inbounds $out_ref = _coeff * $u_ref)
    for d in sort(non_H)                   # ascending → lowest dim innermost
        s     = Symbol("_iy", d)
        inner = :(for $s in 1:Base.size(u, $d); $inner; end)
    end

    closure_body = Base.remove_linenums!(quote
        _n     = $wnum
        _coeff = _ddx_sign * _n * _ddx_scale
        $inner
    end)
    closure = Expr(:->, Expr(:tuple, h_syms...), closure_body)

    return Base.remove_linenums!(quote
        _ddx_scale = wavenumber_scale(grid(u), $Dim)
        _ddx_sign  = adjoint ? -1im : 1im
        for_each_mode($closure, grid(u))
        return out
    end)
end

"""
    add_homogeneous_laplacian_fem!(out::FTField, u::FTField)

`for_each_mode`-based prototype of `add_homogeneous_laplacian!`.
"""
@generated function add_homogeneous_laplacian_fem!(out::FTField{G}, u::FTField{G}) where {T, D, Axes, Hs, Ht, G<:AbstractGrid{T, D, Axes, Hs, Ht}}
    H      = (Hs..., Ht)
    N      = length(H)
    h_syms = [Symbol("_ih", k) for k in 1:N]  # H-order closure args
    n_syms = [Symbol("_n",  k) for k in 1:N]  # signed wavenumber per H dim
    s_syms = [Symbol("_ks", k) for k in 1:N]  # wavenumber scale per H dim

    non_H = [d for d in 1:D if d ∉ H]

    arr_idx = Vector{Any}(undef, D)
    for d in 1:D
        k = findfirst(==(d), H)
        arr_idx[d] = k !== nothing ? h_syms[k] : Symbol("_iy", d)
    end

    # Wavenumber assignments (evaluated inside the closure).
    wnum_assigns = Expr[]
    push!(wnum_assigns, :($(n_syms[1]) = $(h_syms[1]) - 1))   # rfft: always ≥ 0
    for k in 2:N
        d  = H[k]
        hs = h_syms[k]
        push!(wnum_assigns, :($(n_syms[k]) = $hs <= (Base.size(u, $d) >> 1) + 1 ?
                                              $hs - 1 : $hs - Base.size(u, $d) - 1))
    end

    # k² = ∑_k (scale_k · n_k)²
    k2_terms = [:(( $(s_syms[k]) * $(n_syms[k]) )^2) for k in 1:N]
    k2       = length(k2_terms) == 1 ? k2_terms[1] : Expr(:call, :+, k2_terms...)

    out_ref = Expr(:ref, :(parent(out)), arr_idx...)
    u_ref   = Expr(:ref, :(parent(u)),  arr_idx...)
    inner   = :(@inbounds $out_ref -= $k2 * $u_ref)
    for d in sort(non_H)
        s     = Symbol("_iy", d)
        inner = :(for $s in 1:Base.size(u, $d); $inner; end)
    end

    closure_body = Base.remove_linenums!(quote
        $(wnum_assigns...)
        $inner
    end)
    closure = Expr(:->, Expr(:tuple, h_syms...), closure_body)

    # Scales defined outside the closure; captured by reference.
    scale_assigns = [:($(s_syms[k]) = wavenumber_scale(grid(u), $(H[k]))) for k in 1:N]

    return Base.remove_linenums!(quote
        $(scale_assigns...)
        for_each_mode($closure, grid(u))
        return out
    end)
end
