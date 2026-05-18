# JLD2-based persistence for spectral fields.

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
    return ProjectedField(typeof(FTField(g)), data, modes)
end
