module _KNN

import MLJModelInterface
using CategoricalArrays: AbstractCategoricalArray, CatArrOrSub
using CategoricalDistributions: mode
using ChunkSplitters: chunks
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

function MLJModelInterface.reformat(::KNNDTWModel, X::AbstractVector{<:AbstractVector{<:AbstractFloat}})
    return (X,)
end

function MLJModelInterface.reformat(::KNNDTWModel, (X, type)::Tuple{<:AbstractMatrix{<:AbstractFloat}, Symbol})
    @assert type in (:row_based, :column_based) "Unsupported matrix format"
    matrix = MLJModelInterface.matrix(X, transpose = type == :row_based)
    return if type == :row_based
        @info "Copying data from the row based matrix"
        ([matrix[row, :] for row in axes(matrix, 1)],)
    else
        ([view(matrix, :, col) for col in axes(X, 2)],)
    end
end

MLJModelInterface.reformat(m::KNNDTWModel, (X, type), y) = (MLJModelInterface.reformat(m, (X, type))..., MLJModelInterface.categorical(y))

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
    
    @conditional_threads parallel_on_Xnew for (rangeXnew, chunk_id_xnew) in chunks(Xnew, parallel_on_Xnew ? n_chunks : 1, :batch)
        for q in rangeXnew
            @conditional_threads !parallel_on_Xnew for (rangeX, chunk_id_x) in chunks(X, parallel_on_Xnew ? 1 : n_chunks, :batch)
                chunk_id = parallel_on_Xnew ? chunk_id_xnew : chunk_id_x

                for i in rangeX
                    if !isempty(heaps[chunk_id]) && (@views lower_bound!(boundings[chunk_id], Xnew[q], X[i], update_envelope=i == 1)) > first(heaps[chunk_id])[1]
                        continue
                    end
        
                    dtw_distance = @views dtw!(distances[chunk_id], Xnew[q], X[i])

                    if length(heaps[chunk_id]) != model.K
                        push!(heaps[chunk_id], (dtw_distance, y[i]))
                    elseif dtw_distance < first(heaps[chunk_id])[1]
                        pushfirst!(heaps[chunk_id], (dtw_distance, y[i]))
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
                (@views probas[:, q]) .= isinf.(@views probas[:, q])
            end
            @turbo probas[:, q] ./= vsum(@views probas[:, q])

            empty!.(heaps)
            empty!(final_heap)
        end
    end

    return MLJModelInterface.UnivariateFinite(classes, transpose(probas))
end

function MLJModelInterface.fitted_params(::KNNDTWModel, fitresults)
    fitresults
end

end
