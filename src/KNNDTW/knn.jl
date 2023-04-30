module _KNN

import MLJModelInterface
using CategoricalArrays: CategoricalArray
using .._DTW: DTWType, DTW, dtw
using .._Utils: FastMaxHeap

export KNNDTWModel

MLJModelInterface.@mlj_model mutable struct KNNDTWModel <: MLJModelInterface.Supervised
    K::Unsigned = Unsigned(1)::(0 < _)
    weights::Symbol = :uniform::(_ in (:uniform, :distance))
    distance::DTWType = DTW()
end

function MLJModelInterface.reformat(::KNNDTWModel, (X, type))
    @assert type in (:row_based, :column_based)

    (MLJModelInterface.matrix(X, transpose=s === :row_based))
end

function MLJModelInterface.reformat(::KNNDTWModel, (X, type), y)
    @assert type in (:row_based, :column_based)

    (MLJModelInterface.matrix(X, transpose=s === :row_based), MLJModelInterface.categorical(y))
end

function MLJModelInterface.reformat(::KNNDTWModel, (X, type), y, w)
    @assert type in (:row_based, :column_based)

    (MLJModelInterface.matrix(X, transpose=s === :row_based), MLJModelInterface.categorical(y), w)
end

MLJModelInterface.selectrows(::KNNDTWModel, I, Xmatrix) = (view(Xmatrix, :, I),)
MLJModelInterface.selectrows(::KNNDTWModel, I, Xmatrix, y) = (view(Xmatrix, :, I), view(y, I))
MLJModelInterface.selectrows(::KNNDTWModel, I, Xmatrix, y, w) = (view(Xmatrix, :, I), view(y, I), view(w, I))

function MLJModelInterface.fit(::KNNDTWModel, ::Any, X::Matrix{T}, y, w = nothing) where {T <: AbstractFloat}
    return ((X, y, w), nothing, nothing)
end

function MLJModelInterface.predict(model::KNNDTWModel, (X, y, w)::Tuple{Matrix{T}, CategoricalArray, Vector{T2}}, Xnew::Matrix{T}) where {T <: AbstractFloat, T2 <: AbstractFloat}
    heap = FastMaxHeap{T, (eltype(y), typeof(w) == Nothing ? Nothing : eltype(w))}(model.K)
    classes = MLJModelInterface.classes(y)
    probas = zeros(T, length(classes), size(Xnew, 2))

    for q in 1:size(Xnew, 2)
        for i in axes(X, 2)
            dtw_distance = dtw(model.distance, @views Xnew[:, q], @views X[:, i])

            if isempty(heap) || distance < max(heap)
                push!(heap, (dtw_distance, y[i]))
            end
        end
        
        if model.distance == :uniform
            for (dist, (label, weight)) in heap.data
                probas[classes .== label, q] = one(T) / model.K * (w === nothing ? 1 : weight);
            end
        elseif model.distance == :distance
            for (dist, (label, weight)) in heap.data
                probas[classes .== label, q] = one(T) / dist * (w === nothing ? 1 : weight);
            end
        end

        probas[:, q] ./= sum(probas[:, q])
        empty!(heap)
    end

    return MLJModelInterface.UnivariateFinite(classes, transpose(probas), pool=y)
end

function MLJModelInterface.fitted_params(::KNNDTWModel, fitresults)
    fitresults
end

end