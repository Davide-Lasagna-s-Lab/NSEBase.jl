@testset verbose=true "Channel case                                                " begin
    include("channel/helpers.jl")
    include("channel/grid.jl")
    include("channel/derivatives.jl")
    include("channel/norms.jl")
    include("channel/shifts.jl")
    include("channel/weighting.jl")
    include("channel/equations.jl")
end
