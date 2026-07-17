using Documenter, NSEBase

makedocs(
    sitename = "NSEBase.jl",
    modules  = [NSEBase],
    authors  = "Davide Lasagna",
    format   = Documenter.HTML(
        prettyurls  = get(ENV, "CI", nothing) == "true",
        canonical   = "https://Davide-Lasagna-s-Lab.github.io/NSEBase.jl/stable",
    ),
    pages = [
        "Home" => "index.md",
        "Concepts and conventions" => "guide.md",
        "Rectangular grids" => "rectangular_grids.md",
        "Cases" => [
            "Overview" => "cases/index.md",
            "Channel flow" => "cases/channel.md",
            "Square-duct flow" => "cases/duct.md",
            "Lid-driven cavity" => "cases/lid_driven_cavity.md",
            "Rotating plane Couette flow" => "cases/rpcf.md",
            "Rayleigh–Bénard convection" => "cases/rayleigh_benard.md",
            "Migrating from case packages" => "cases/migration.md",
        ],
        "API reference" => [
            "Overview" => "api.md",
            "Grids and layouts" => "api/grids.md",
            "Fields and transforms" => "api/fields.md",
            "Numerical operations" => "api/operations.md",
            "Equation formulations" => "api/equations.md",
            "Cases and forcing" => "api/cases.md",
        ],
    ],
    checkdocs = :exports,
    warnonly  = false,
)

deploydocs(
    repo   = "github.com/Davide-Lasagna-s-Lab/NSEBase.jl.git",
    target = "build",
)
