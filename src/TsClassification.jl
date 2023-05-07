module TsClassification

import MLJModelInterface

include("MiniRocket/MiniRocket.jl")
include("KNNDTW/KNNDTW.jl")
include("DataSets/DataSets.jl")

export MiniRocketModel
using .MiniRocket: MiniRocketModel

export DTWType, dtw!, DTW, DTWSakoeChiba, DTWItakura, LBType, lower_bound!, LBNone, LBKeogh, KNNDTWModel
using .KNNDTW: DTWType, dtw!, DTW, DTWSakoeChiba, DTWItakura, LBType, lower_bound!, LBNone, LBKeogh, KNNDTWModel

export DataSets
import .DataSets

MLJModelInterface.metadata_pkg.(
    (MiniRocketModel, KNNDTWModel),
    name = "TsClassification",
    uuid = "4869f98a-92d2-4a27-bbf6-5599fe134177",
    url = "https://github.com/antoninkriz/julia-ts-classification",
    julia = true,
    license = "TODO",
    is_wrapper = false,
)

MLJModelInterface.metadata_model(
    MiniRocketModel,
    input_scitype = Tuple{AbstractMatrix{MLJModelInterface.Continuous}, MLJModelInterface.Unknown},
    output_scitype = AbstractMatrix{<:MLJModelInterface.Continuous},
    descr = "MiniRocket model",
    load_path = "TsClassification.MiniRocketModel",
)

MLJModelInterface.metadata_model(
    KNNDTWModel,
    input_scitype = Tuple{AbstractMatrix{MLJModelInterface.Continuous}, MLJModelInterface.Unknown},
    output_scitype = AbstractMatrix{<:MLJModelInterface.Finite},
    descr = "KNN+DTW model",
    load_path = "TsClassification.KNNDTWModel",
)

"""
$(MLJModelInterface.doc_header(MiniRocketModel))

### MiniRocketModel

A model type for constructing MiniRocket transformer.
"""
MiniRocketModel

end
