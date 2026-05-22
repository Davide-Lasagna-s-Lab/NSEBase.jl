# JLD2-based persistence for grids and spectral fields.

"""
    save_grid(g::AbstractGrid; path="./grid.jld2")

Save the grid `g` to `path` in JLD2 format.
"""
function save_grid(g::AbstractGrid; path="./grid.jld2")
    jldopen(path, "w") do f
        f["grid"] = g
    end
    return nothing
end

"""
    load_grid(path) -> AbstractGrid

Load a grid from the JLD2 file at `path`.
"""
function load_grid(path)
    jldopen(path, "r") do f
        return f["grid"]
    end
end

"""
    save_field(a::ProjectedField; path="./a.jld2")

Save the coefficient array of `a` to `path` in JLD2 format.
"""
function save_field(a::ProjectedField; path="./a.jld2")
    jldopen(path, "w") do f
        f["data"] = parent(a)
    end
    return nothing
end

"""
    load_field(g::AbstractGrid, modes, path) -> ProjectedField

Load a `ProjectedField` from the JLD2 file at `path`, using grid `g` and `modes`.
"""
function load_field(g::AbstractGrid, modes, path)
    data = jldopen(path, "r") do f
        f["data"]
    end
    return ProjectedField(g, data, modes)
end
