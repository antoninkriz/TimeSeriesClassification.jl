module _KNN

import MLJModelInterface
using CategoricalArrays: AbstractCategoricalArray, CatArrOrSub
using CategoricalDistributions: mode
using VectorizedStatistics: vsum
using Logging: @info
using LoopVectorization: @turbo
using .._DTW: DTWType, DTW, dtw!
using .._LB: LBType, LBNone, lower_bound!
using .._Utils: FastHeap

export KNNDTWModel

MLJModelInterface.@mlj_model mutable struct KNNDTWModel <: MLJModelInterface.Probabilistic
    K::Int64 = 1::(0 < _)
    weights::Symbol = :uniform::(_ in (:uniform, :distance))
    distance::DTWType = DTW{AbstractFloat}()
    bounding::LBType = LBNone()
end

function MLJModelInterface.reformat(::KNNDTWModel, X::AbstractMatrix{<:AbstractFloat})
    Xt = transpose(X)
    return ([view(Xt, :, col) for col in axes(Xt, 2)],)
end
MLJModelInterface.reformat(m::KNNDTWModel, X::AbstractMatrix{<:AbstractFloat}, y) = (MLJModelInterface.reformat(m, X)..., MLJModelInterface.categorical(y))

MLJModelInterface.reformat(::KNNDTWModel, X::AbstractVector{<:AbstractVector{<:AbstractFloat}}) = (X,)
MLJModelInterface.reformat(::KNNDTWModel, X::AbstractVector{<:AbstractVector{<:AbstractFloat}}, y) = (X, MLJModelInterface.categorical(y))

MLJModelInterface.selectrows(::KNNDTWModel, I, Xvec) = (view(Xvec, I),)
MLJModelInterface.selectrows(::KNNDTWModel, I, Xvec, y) = (view(Xvec, I), view(y, I))

function MLJModelInterface.fit(
    ::KNNDTWModel,
    ::Any,
    X::AbstractVector{<:AbstractVector{T}},
    y::Union{AbstractCategoricalArray, SubArray{<:Any, <:Any, <:AbstractCategoricalArray}}
) where {T <: AbstractFloat}
    ((X, y), nothing, nothing)
end

macro conditional_threads(cond::Union{Bool, Symbol, Expr}, ex::Expr)
    q = quote
        Threads.@threads $ex
    end

    quote
        if $(esc(cond))
            $(esc(q))
        else
            $(esc(ex))
        end 
    end
end

function MLJModelInterface.predict_mode(
    model::KNNDTWModel,
    fitparams::Tuple{
        AbstractVector{<:AbstractVector{T}},
        Union{AbstractCategoricalArray, SubArray{<:Any, <:Any, <:AbstractCategoricalArray}}
    },
    Xnew::AbstractVector{<:AbstractVector{T}}
) where {T <: AbstractFloat}
    return mode.(MLJModelInterface.predict(model, fitparams, Xnew))
end

function MLJModelInterface.predict(
    model::KNNDTWModel,
    (X, y)::Tuple{
        AbstractVector{<:AbstractVector{T}},
        Union{AbstractCategoricalArray, SubArray{<:Any, <:Any, <:AbstractCategoricalArray}}
    },
    Xnew::AbstractVector{<:AbstractVector{T}}
) where {T <: AbstractFloat}
    # How should we do threading?
    parallel_on_Xnew = length(Xnew) >= length(X)
    n_chunks = min(Threads.nthreads(), parallel_on_Xnew ? length(Xnew) : length(X))

    # Init stuff for threads
    heaps = [
        FastHeap{T, eltype(y)}(model.K, :max)
        for _ in 1:n_chunks
    ]
    distances = [
        deepcopy(model.distance)
        for _ in 1:n_chunks
    ]
    boundings = [
        deepcopy(model.bounding)
        for _ in 1:n_chunks
    ]

    # Prepare stuff for aggregating results
    classes = MLJModelInterface.classes(y)
    probas = zeros(T, length(classes), length(Xnew))
    chunksXnew = collect(parallel_on_Xnew ? (((round(Int64, i * (length(Xnew) / n_chunks)))+1:round(Int64,(i+1)*(length(Xnew) / n_chunks)), i+1) for i in 0:n_chunks-1) : ((1:length(Xnew), -1),))
    chunksX = collect(!parallel_on_Xnew ? (((round(Int64, i * (length(X) / n_chunks)))+1:round(Int64,(i+1)*(length(X) / n_chunks)), i+1) for i in 0:n_chunks-1) : ((1:length(X), -1),))

    @inbounds @conditional_threads parallel_on_Xnew for (rangeXnew, chunk_id_xnew) in chunksXnew
        for q in rangeXnew
            @conditional_threads !parallel_on_Xnew for (rangeX, chunk_id_x) in chunksX
                chunk_id = parallel_on_Xnew ? chunk_id_xnew : chunk_id_x
                heap = heaps[chunk_id]

                for i in rangeX
                    if !isempty(heap) && lower_bound!(boundings[chunk_id], Xnew[q], X[i], update=i == 1) > first(heap)[1]
                        continue
                    end

                    dtw_distance = dtw!(distances[chunk_id], Xnew[q], X[i])

                    if length(heap) < model.K
                        push!(heap, (dtw_distance, y[i]))
                    elseif dtw_distance < first(heap)[1]
                        pushfirst!(heap, (dtw_distance, y[i]))
                    end
                end
            end

            # Merge heaps from threads into one when using multiple for one sample form Xnew
            final_heap = if parallel_on_Xnew
                heaps[chunk_id_xnew]
            else
                for h in @views heaps[2:end]
                    for el in @views h.data[1:length(h)]
                        if length(heaps[1]) != model.K
                            push!(heaps[1], el)
                        elseif el[1] < first(heaps[1])[1]
                            pushfirst!(heaps[1], el)
                        end
                    end
                end
                heaps[1]
            end

            # Calculate probabilities
            if model.weights == :uniform
                for (_, label) in @views final_heap.data[begin:length(final_heap)]
                    probas[findfirst(==(label), classes), q] += one(T) / model.K
                end
            elseif model.weights == :distance
                for (dist, label) in @views final_heap.data[begin:length(final_heap)]
                    probas[findfirst(==(label), classes), q] += one(T) / dist
                end
            end

            has_inf = findfirst(isinf, @views probas[:, q]) !== nothing
            if has_inf
                @turbo (@views probas[:, q]) .= isinf.(@views probas[:, q])
            end
            @turbo (@views probas[:, q]) ./= vsum(@views probas[:, q])

            if parallel_on_Xnew
                empty!(final_heap)
            else
                empty!.(heaps)
            end
        end
    end

    return MLJModelInterface.UnivariateFinite(classes, transpose(probas))
end

function MLJModelInterface.fitted_params(::KNNDTWModel, fitresults)
    fitresults
end

end
