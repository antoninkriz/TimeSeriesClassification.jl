module KNNDTW

include("utils.jl")
include("dtw.jl")
include("knn.jl")

export dtw, DTW, DTWSakoeChiba, DTWItakura, KNNDTWModel
using ._DTW: dtw, DTW, DTWSakoeChiba, DTWItakura
using ._KNN: KNNDTWModel

end
