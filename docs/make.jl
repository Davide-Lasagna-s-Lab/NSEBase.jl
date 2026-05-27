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
        "Home"                    => "index.md",
        "Concepts & Conventions"  => "guide.md",
        "API Reference"           => "api.md",
    ],
    checkdocs = :exports,
    warnonly  = false,
)

deploydocs(
    repo   = "github.com/Davide-Lasagna-s-Lab/NSEBase.jl.git",
    target = "build",
)
