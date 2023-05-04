module _KNN

import MLJModelInterface
using CategoricalArrays: AbstractCategoricalArray, CatArrOrSub
using VectorizedStatistics: vsum
using LoopVectorization: @turbo
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

    (MLJModelInterface.matrix(X, transpose = type == :row_based),)
end

function MLJModelInterface.reformat(::KNNDTWModel, (X, type), y)
    @assert type in (:row_based, :column_based)

    (MLJModelInterface.matrix(X, transpose = type == :row_based), MLJModelInterface.categorical(y))
end

function MLJModelInterface.reformat(::KNNDTWModel, (X, type), y, w)
    @assert type in (:row_based, :column_based)

    (MLJModelInterface.matrix(X, transpose = type == :row_based), MLJModelInterface.categorical(y), w)
end

MLJModelInterface.selectrows(::KNNDTWModel, I, Xmatrix) = (view(Xmatrix, :, I),)
MLJModelInterface.selectrows(::KNNDTWModel, I, Xmatrix, y) = (view(Xmatrix, :, I), view(y, I))
MLJModelInterface.selectrows(::KNNDTWModel, I, Xmatrix, y, w) = (view(Xmatrix, :, I), view(y, I), view(w, I))

function MLJModelInterface.fit(::KNNDTWModel, ::Any, X::AbstractMatrix{T}, y::Union{AbstractCategoricalArray, SubArray{<:Any, <:Any, <:AbstractCategoricalArray}}, w = nothing) where {T <: AbstractFloat}
    return ((X, y, w), nothing, nothing)
end

function MLJModelInterface.predict(model::KNNDTWModel, (X, y, w), Xnew::Matrix{T}) where {T <: AbstractFloat}
    heap = FastMaxHeap{T, Tuple{eltype(y), typeof(w) == Nothing ? Nothing : eltype(w)}}(model.K)
    classes = MLJModelInterface.classes(y)
    probas = zeros(T, length(classes), size(Xnew, 2))

    for q in 1:size(Xnew, 2)
        for i in axes(X, 2)
            dtw_distance = @views dtw(model.distance, Xnew[:, q], X[:, i])

            if isempty(heap) || dtw_distance < max(heap)[1]
                push!(heap, (dtw_distance, (y[i], w === nothing ? nothing : w[i])))
            end
        end

        if model.weights == :uniform
            for (_, (label, weight)) in @views heap.data[begin:length(heap)]
                ww = (w === nothing ? 1 : weight)
                probas[findfirst(==(label), classes), q] = one(T) / model.K * ww
            end
        elseif model.weights == :distance
            for (dist, (label, weight)) in @views heap.data[begin:length(heap)]
                ww = (w === nothing ? 1 : weight)
                probas[findfirst(==(label), classes), q] = one(T) / (dist + sqrt(nextfloat(zero(Float64)))) * ww
            end
        end

        @turbo probas[:, q] ./= vsum(@views probas[:, q])
        empty!(heap)
    end

    return MLJModelInterface.UnivariateFinite(classes, transpose(probas))
end

function MLJModelInterface.fitted_params(::KNNDTWModel, fitresults)
    fitresults
end

end
