module TsClassification

import MLJModelInterface
import ScientificTypesBase

include("MiniRocket/MiniRocket.jl")
include("KNNDTW/KNNDTW.jl")
include("DataSets/DataSets.jl")

export MiniRocketModel
using .MiniRocket: MiniRocketModel

export dtw, DTW, DTWSakoeChiba, DTWItakura, KNNDTWModel
using .KNNDTW: dtw, DTW, DTWSakoeChiba, DTWItakura, KNNDTWModel

export MiniRocketModel
using .MiniRocket: MiniRocketModel

export DataSets
import .DataSets


MLJModelInterface.metadata_pkg.(
    (MiniRocketModel),
    name = "TsClassification",
    uuid = "4869f98a-92d2-4a27-bbf6-5599fe134177",
    url = "https://github.com/antoninkriz/julia-ts-classification",
    julia = true,
    license = "TODO",
    is_wrapper = false,
)

MLJModelInterface.metadata_model(
    MiniRocketModel,
    input_scitype = Tuple{Tuple{AbstractMatrix{ScientificTypesBase.Continuous}, ScientificTypesBase.Unknown}},
    output_scitype = AbstractMatrix{<:MLJModelInterface.Continuous},
    descr = "MiniRocket model",
    load_path = "TsClassification.MiniRocketModel",
)

"""
$(MLJModelInterface.doc_header(MiniRocketModel))

### MiniRocketModel

A model type for constructing MiniRocket transformer.
"""
MiniRocketModel

end
