module _KNN

import MLJModelInterface
using .._DTW: DTWType, DTW, dtw
using .._Utils: FastMaxHeap

export KNNDTWModel

MLJModelInterface.@mlj_model mutable struct KNNDTWModel <: MLJModelInterface.Supervised
    k::Unsigned = Unsigned(1)::(0 < _)
    weights::Symbol = :uniform::(_ in (:uniform, :distance))
    distance::DTWType = DTW()
end

# TODO: Version with "y", reformat "y" to categorical
function MLJModelInterface.reformat(::KNNDTWModel, (X, type::Symbol))
    @assert type in (:row_based, :column_based)

    (MLJModelInterface.matrix(X, transpose=s == :row_based),)
end
# TODO: Version with "y"
MLJModelInterface.selectrows(::KNNDTWModel, I, Xmatrix) = (view(Xmatrix, :, I),)

function MLJModelInterface.fit(::KNNDTWModel, _, X::Matrix{T}, y) where {T <: AbstractFloat}
    return ((X, y), nothing, nothing)
end

function MLJModelInterface.predict(model::KNNDTWModel, (X::Matrix{T}, y), Xnew::Matrix{T}) where {T <: AbstractFloat}
    heap = FastMaxHeap(model.k)

    for x_query in Xnew
        for i in axes(X, 2)
            distance = dtw(model.distance, x_query, @views X[:, i])

            if isempty(heap) || distance < max(heap)
                push!(heap, (distance, y[i]))
            end
        end
    end

    # TODO: Implement classification
end

function MLJModelInterface.fitted_params(::KNNDTWModel, fitresults)
    fitresults
end

end