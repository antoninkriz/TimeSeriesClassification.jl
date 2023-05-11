module _MiniRocket

using Random: AbstractRNG, GLOBAL_RNG, rand
using StaticArrays: SMatrix
using Statistics: quantile, quantile!
using VectorizedStatistics: vsum
using LoopVectorization: @turbo
import MLJModelInterface

using .._Utils: sorted_unique_counts, logspace, RangeAsArray

export MiniRocketModel

const NUM_KERNELS::Int64 = 84

const INDICES::SMatrix{3, NUM_KERNELS, Int64} = [
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 5 5 5 5 5 5 6 6 6 7
    2 2 2 2 2 2 2 3 3 3 3 3 3 4 4 4 4 4 5 5 5 5 6 6 6 7 7 8 3 3 3 3 3 3 4 4 4 4 4 5 5 5 5 6 6 6 7 7 8 4 4 4 4 4 5 5 5 5 6 6 6 7 7 8 5 5 5 5 6 6 6 7 7 8 6 6 6 7 7 8 7 7 8 8
    3 4 5 6 7 8 9 4 5 6 7 8 9 5 6 7 8 9 6 7 8 9 7 8 9 8 9 9 4 5 6 7 8 9 5 6 7 8 9 6 7 8 9 7 8 9 8 9 9 5 6 7 8 9 6 7 8 9 7 8 9 8 9 9 6 7 8 9 7 8 9 8 9 9 7 8 9 8 9 9 8 9 9 9
]

function fit_biases(
    X::AbstractMatrix{T},
    dilations::Vector{Int64},
    num_features_per_dilation::Vector{Int64},
    quantiles::Vector{T};
    shuffled::Bool = false,
    rng::AbstractRNG,
)::AbstractVector{T} where {T <: AbstractFloat}
    input_length, num_examples = size(X)

    num_features = NUM_KERNELS * vsum(num_features_per_dilation)
    biases = zeros(T, num_features)

    feature_index_start = 0

    C = zeros(T, input_length)
    A = zeros(T, input_length)
    G = zeros(T, input_length)

    @fastmath @inbounds for dilation_index in eachindex(dilations)
        dilation = dilations[dilation_index]
        padding = ((9 - 1) * dilation) ÷ 2
        num_features_this_dilation = num_features_per_dilation[dilation_index]

        for kernel_index in 1:NUM_KERNELS
            feature_index_end = feature_index_start + num_features_this_dilation

            _X = @views X[:, shuffled ? (((kernel_index + ((dilation_index - 1) * NUM_KERNELS) - 1) % num_examples) + 1) : (abs(rand(rng, Int64) % num_examples) + 1)]

            @turbo @. A .= _X .* T(-1)
            @turbo @. G .= _X .* T(3)

            copyto!(C, A)

            s = dilation + 1
            e = input_length - padding

            for idx in 1:(9÷2)
                d = e + dilation * (idx - 1)
                @turbo (@views C[end-d+1:end]) .+= (@views A[begin:d])
            end

            for idx in (9÷2)+2:9
                d = s + dilation * (idx - ((9 ÷ 2) + 2))
                @turbo (@views C[begin:end-d+1]) .+= (@views A[d:end])
            end

            for idx in @views INDICES[:, kernel_index]
                if idx < 5
                    d = e + dilation * (idx - 1)
                    @turbo (@views C[end-d+1:end]) .+= (@views G[begin:d])
                elseif idx > 5
                    d = s + dilation * (idx - ((9 ÷ 2) + 2))
                    @turbo (@views C[begin:end-d+1]) .+= (@views G[d:end])
                else
                    @turbo @views C .+= G
                end
            end

            @views quantile!(
                biases[feature_index_start+1:feature_index_end],
                C,
                quantiles[feature_index_start+1:feature_index_end],
            )

            feature_index_start = feature_index_end
        end
    end

    return biases
end

function fit_dilations(
    input_length::Int64,
    num_features::Unsigned,
    max_dilations_per_kernel::Unsigned,
)::Tuple{Vector{Int64}, Vector{Int64}}
    num_features_per_kernel = Int64(num_features) ÷ NUM_KERNELS
    true_max_dilations_per_kernel = min(num_features_per_kernel, Int64(max_dilations_per_kernel))
    multiplier = num_features_per_kernel / true_max_dilations_per_kernel

    max_exponent = log2((input_length - 1) / (9 - 1))

    dilations, num_features_per_dilation =
        sorted_unique_counts(floor.(Int64, logspace(0, max_exponent, true_max_dilations_per_kernel, base = 2.0)))

    num_features_per_dilation = floor.(Int64, (num_features_per_dilation * multiplier))

    remainder = num_features_per_kernel - vsum(num_features_per_dilation)
    i = 1
    while remainder > 0
        num_features_per_dilation[i] += 1
        i = (i % length(num_features_per_dilation)) + 1
        remainder -= 1
    end

    return dilations, num_features_per_dilation
end

function fit(
    X::AbstractMatrix{T};
    num_features::Unsigned = Unsigned(10_000),
    max_dilations_per_kernel::Unsigned = Unsigned(32),
    shuffled::Bool = false,
    rng::AbstractRNG = GLOBAL_RNG,
)::Tuple{Vector{Int64}, Vector{Int64}, Vector{T}} where {T <: AbstractFloat}
    input_length = size(X, 1)

    dilations, num_features_per_dilation = fit_dilations(input_length, num_features, max_dilations_per_kernel)

    num_features_per_kernel = vsum(num_features_per_dilation)

    quantiles = [(x * ((sqrt(T(5)) + 1) / 2)) % 1 for x in 1:(NUM_KERNELS*num_features_per_kernel)]

    biases = fit_biases(X, dilations, num_features_per_dilation, quantiles, shuffled = shuffled, rng = rng)

    return (dilations, num_features_per_dilation, biases)
end

@inline function fast_ppv(arr::AbstractVector{T}, bias::T) where T
    s = 0
    @turbo for i in eachindex(arr)
        s += arr[i] > bias
    end
    s / length(arr)
end

function transform(
    X::AbstractMatrix{T};
    dilations::Vector{Int64},
    num_features_per_dilation::Vector{Int64},
    biases::Vector{T},
)::Matrix{T} where {T <: AbstractFloat}
    input_length, num_examples = size(X)
    n_chunks = min(Threads.nthreads(), num_examples)

    features = zeros(T, (NUM_KERNELS * vsum(num_features_per_dilation), num_examples))

    # Small allocations might be faster than this thing. This needs tes`ting.
    C_alpha_threads = zeros(T, input_length, n_chunks)
    C_threads = zeros(T, input_length, n_chunks)
    A_threads = zeros(T, input_length, n_chunks)
    G_threads = zeros(T, input_length, n_chunks)

    chunks = collect(((round(Int64, i * (num_examples / n_chunks)))+1:round(Int64,(i+1)*(num_examples / n_chunks)), i+1) for i in 0:n_chunks-1)
    @fastmath @inbounds Threads.@threads for (xrange, chunk_id) in chunks
        for example_index in xrange
            _X = @views X[:, example_index]

            C_alpha = @views C_alpha_threads[:, chunk_id]
            C = @views C_threads[:, chunk_id]
            A = @views A_threads[:, chunk_id]
            G = @views G_threads[:, chunk_id]

            fill!(C_alpha, zero(T))
            fill!(C, zero(T))

            @turbo @. A .= _X .* T(-1)
            @turbo @. G .= _X .* T(3)

            feature_index_start = 0

            for dilation_index in 1:length(dilations)
                dilation = dilations[dilation_index]
                padding = ((9 - 1) * dilation) ÷ 2
                num_features_this_dilation = num_features_per_dilation[dilation_index]

                copyto!(C_alpha, A)

                s = dilation + 1
                e = input_length - padding

                for idx in 1:(9÷2)
                    d = e + dilation * (idx - 1)
                    @turbo (@views C_alpha[end-d+1:end]) .+= (@views A[begin:d])
                end

                for idx in (9÷2)+2:9
                    d = s + dilation * (idx - ((9 ÷ 2) + 2))
                    @turbo (@views C_alpha[begin:end-d+1]) .+= (@views A[d:end])
                end

                _padding0 = (dilation_index - 1) % 2
                for kernel_index in 1:NUM_KERNELS
                    feature_index_end = feature_index_start + num_features_this_dilation

                    copyto!(C, C_alpha)

                    for idx in @views INDICES[:, kernel_index]
                        if idx < 5
                            d = e + dilation * (idx - 1)
                            @turbo (@views C[end-d+1:end]) .+= (@views G[begin:d])
                        elseif idx > 5
                            d = s + dilation * (idx - ((9 ÷ 2) + 2))
                            @turbo (@views C[begin:end-d+1]) .+= (@views G[d:end])
                        else
                            @turbo (C .+= G)
                        end
                    end

                    _padding1 = (_padding0 + (kernel_index - 1)) % 2
                    if _padding1 == 0
                        for feature_count in 1:num_features_this_dilation
                            features[feature_index_start+feature_count, example_index] =
                                fast_ppv(C, biases[feature_index_start+feature_count])
                        end
                    else
                        for feature_count in 1:num_features_this_dilation
                            features[feature_index_start+feature_count, example_index] =
                                fast_ppv((@views C[padding+1:end-padding]), biases[feature_index_start+feature_count])
                        end
                    end

                    feature_index_start = feature_index_end
                end
            end
        end
    end

    return features
end

"Structure defining the MiniRocket transformer."
MLJModelInterface.@mlj_model mutable struct MiniRocketModel <: MLJModelInterface.Unsupervised
    num_features::Unsigned = Unsigned(10_000)::(84 <= _)
    max_dilations_per_kernel::Unsigned = Unsigned(32)::(0 < _)
    rng::AbstractRNG = GLOBAL_RNG
    shuffled::Bool = false
end

function MLJModelInterface.reformat(::MiniRocketModel, (X, type))
    @assert type in (:row_based, :column_based)

    (MLJModelInterface.matrix(X, transpose = type == :row_based),)
end
MLJModelInterface.selectrows(::MiniRocketModel, I, Xmatrix) = (view(Xmatrix, :, I),)

"Function to train MiniRocket transformer."
function MLJModelInterface.fit(
    model::MiniRocketModel,
    ::Any,
    X::AbstractMatrix{T},
)::Tuple{
    Tuple{Vector{Int64}, Vector{Int64}, Vector{T}},
    Nothing,
    Nothing,
} where {T <: AbstractFloat}
    return fit(
        X,
        num_features = model.num_features,
        max_dilations_per_kernel = model.max_dilations_per_kernel,
        shuffled = model.shuffled,
        rng = model.rng,
    ), nothing, nothing
end

"Function to transform a datased using trained values of the MiniRocket transformer."
function MLJModelInterface.transform(
    model::MiniRocketModel,
    fitresult::Tuple{Vector{Int64}, Vector{Int64}, Vector{T}},
    Xnew::AbstractMatrix{T},
)::AbstractMatrix{T} where {T <: AbstractFloat}
    return transform(
        Xnew,
        dilations = fitresult[1],
        num_features_per_dilation = fitresult[2],
        biases = fitresult[3],
    )
end

"Loads fit paramters of the MiniRocket transformer."
function MLJModelInterface.fitted_params(::MiniRocketModel, fitresult)
    return (dilations=fitresult[1], num_features_per_dilation=fitresult[2], biases=fitresult[3])
end

end
