module TsClassification

import MLJModelInterface

include("MiniRocket/MiniRocket.jl")
include("KNNDTW/KNNDTW.jl")

export MiniRocketModel
using .MiniRocket: MiniRocketModel



MLJModelInterface.metadata_pkg.(
    (MiniRocketModel),
    name = "TsClassification",
    uuid = "4869f98a-92d2-4a27-bbf6-5599fe134177",
    url = "TODO",
    julia = true,
    license = "TODO",
    is_wrapper = false,
)

MLJModelInterface.metadata_model(
    MiniRocketModel,
    input_scitype = AbstractMatrix{<:MLJModelInterface.Continuous},
    output_scitype = Tuple{AbstractVector{Unsigned}, AbstractVector{Unsigned}, AbstractVector{MLJModelInterface.Continuous}},
    descr = "MiniRocket model",
    load_path = "TsClassification.MiniRocketModel",
)

end
