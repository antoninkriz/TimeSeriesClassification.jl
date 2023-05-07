module KNNDTW

include("utils.jl")
include("dtw.jl")
include("lb.jl")
include("knn.jl")

export dtw!, DTW, DTWSakoeChiba, DTWItakura, lower_bound!, LBNone, LBKeogh, KNNDTWModel
using ._DTW: DTWType, dtw!, DTW, DTWSakoeChiba, DTWItakura
using ._LB: LBType, lower_bound!, LBNone, LBKeogh
using ._KNN: KNNDTWModel

end
