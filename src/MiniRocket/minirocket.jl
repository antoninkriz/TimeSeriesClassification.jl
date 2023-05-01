module _MiniRocket

using Random: AbstractRNG, GLOBAL_RNG, rand
using StaticArrays: SMatrix
using Statistics: quantile, quantile!
using LoopVectorization: @turbo
import MLJModelInterface
import ScientificTypesBase

using .._Utils: sorted_unique_counts, logspace

export MiniRocketModel


const NUM_KERNELS = 84

const INDICES::SMatrix{3,NUM_KERNELS,Int64} = [
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 5 5 5 5 5 5 6 6 6 7
    2 2 2 2 2 2 2 3 3 3 3 3 3 4 4 4 4 4 5 5 5 5 6 6 6 7 7 8 3 3 3 3 3 3 4 4 4 4 4 5 5 5 5 6 6 6 7 7 8 4 4 4 4 4 5 5 5 5 6 6 6 7 7 8 5 5 5 5 6 6 6 7 7 8 6 6 6 7 7 8 7 7 8 8
    3 4 5 6 7 8 9 4 5 6 7 8 9 5 6 7 8 9 6 7 8 9 7 8 9 8 9 9 4 5 6 7 8 9 5 6 7 8 9 6 7 8 9 7 8 9 8 9 9 5 6 7 8 9 6 7 8 9 7 8 9 8 9 9 6 7 8 9 7 8 9 8 9 9 7 8 9 8 9 9 8 9 9 9
]


function fit_biases(X::AbstractMatrix{T}, dilations::Vector{Int64}, num_features_per_dilation::Vector{Int64}, quantiles::Vector{T}; shuffled::Bool=false, rng::AbstractRNG)::AbstractVector{T} where {T <: AbstractFloat}
    input_length, num_examples = size(X)

    num_features = NUM_KERNELS * sum(num_features_per_dilation)
    biases = zeros(T, num_features)

    feature_index_start = 0

    # TODO: Make this work for unshuffled datasets
    # This will work ONLY with the assumption that the dataset is already shuffled!
    # Also, this index starts from zero because it's used within modulo operation.
    idx_nonrand = 0

    _quantiles = zeros(T, maximum(num_features_per_dilation))
    C = zeros(T, input_length)
    A = zeros(T, input_length)
    G = zeros(T, input_length)

    @inbounds for dilation_index = eachindex(dilations)
        dilation = dilations[dilation_index]
        padding = ((9 - 1) * dilation) ÷ 2
        num_features_this_dilation = num_features_per_dilation[dilation_index]

        for kernel_index = 1:NUM_KERNELS
            feature_index_end = feature_index_start + num_features_this_dilation

            _X = @views X[:, shuffled ? ((idx_nonrand % num_examples) + 1) : (abs(rand(rng, Int64) % num_examples) + 1)]
            idx_nonrand += 1

            @turbo @. A = _X * -1
            @turbo @. G = _X * 3

            copy!(C, A)

            s = dilation + 1
            e = input_length - padding

            for idx in 1:(9 ÷ 2)
                d = e + dilation * (idx - 1)
                @views C[end - d + 1:end] += A[begin:d]
            end

            for idx in (9 ÷ 2) + 2:9
                d = s + dilation * (idx - ((9 ÷ 2) + 2))
                @views C[begin:end - d + 1] += A[d:end]
            end

            for idx in @views INDICES[:, kernel_index]
                if idx < 5
                    d = e + dilation * (idx - 1)
                    @views C[end - d + 1:end] += G[begin:d]
                elseif idx > 5
                    d = s + dilation * (idx - ((9 ÷ 2) + 2))
                    @views C[begin:end - d + 1] += G[d:end]
                else
                    C += G
                end
            end

            biases[feature_index_start + 1:feature_index_end] = @views quantile!(_quantiles[1:num_features_this_dilation], C, quantiles[feature_index_start + 1:feature_index_end])

            feature_index_start = feature_index_end
        end
    end

    return biases
end


function fit_dilations(input_length::Int64, num_features::Unsigned, max_dilations_per_kernel::Unsigned)::Tuple{Vector{Int64},Vector{Int64}}
    num_features_per_kernel = num_features ÷ NUM_KERNELS
    true_max_dilations_per_kernel = min(num_features_per_kernel, max_dilations_per_kernel)
    multiplier = num_features_per_kernel / true_max_dilations_per_kernel

    max_exponent = log2((input_length - 1) / (9 - 1))

    dilations, num_features_per_dilation = sorted_unique_counts(floor.(Int64, logspace(0, max_exponent, true_max_dilations_per_kernel, base = 2)))

    num_features_per_dilation = floor.(Int64, (num_features_per_dilation * multiplier))

    remainder = num_features_per_kernel - sum(num_features_per_dilation)
    i = 1
    while remainder > 0
        num_features_per_dilation[i] += 1
        i = (i % length(num_features_per_dilation)) + 1
        remainder -= 1
    end

    return dilations, num_features_per_dilation
end


function fit(X::AbstractMatrix{T}; num_features::Unsigned=Unsigned(10_000), max_dilations_per_kernel::Unsigned=Unsigned(32), shuffled::Bool=false, rng::AbstractRNG=GLOBAL_RNG)::NamedTuple{(:dilations, :num_features_per_dilation, :biases), Tuple{Vector{Int64}, Vector{Int64}, Vector{T}}} where {T <: AbstractFloat}
    input_length = size(X, 1)

    dilations, num_features_per_dilation = fit_dilations(input_length, num_features, max_dilations_per_kernel)

    num_features_per_kernel = sum(num_features_per_dilation)

    quantiles = [(x * ((sqrt(T(5)) + 1) / 2)) % 1 for x in 1:(NUM_KERNELS * num_features_per_kernel)]

    biases = fit_biases(X, dilations, num_features_per_dilation, quantiles, shuffled = shuffled, rng = rng)

    return (dilations=dilations, num_features_per_dilation=num_features_per_dilation, biases=biases)
end


function transform(X::AbstractMatrix{T}; dilations::Vector{Int64}, num_features_per_dilation::Vector{Int64}, biases::Vector{T})::Matrix{T} where {T <: AbstractFloat}
    input_length, num_examples = size(X)

    features = zeros(T, (NUM_KERNELS * sum(num_features_per_dilation), num_examples))

    # Small allocations might be faster than this thing. This needs testing.
    C_alpha_theads = zeros(T, input_length, Threads.nthreads())
    C_theads = zeros(T, input_length, Threads.nthreads())
    A_threads = zeros(T, input_length, Threads.nthreads())
    G_threads = zeros(T, input_length, Threads.nthreads())

    @inbounds Threads.@threads for example_index in 1:num_examples
        _X = @views X[:, example_index]

        C_alpha = @views C_alpha_theads[:, Threads.threadid()]
        C = @views C_theads[:, Threads.threadid()]
        A = @views A_threads[:, Threads.threadid()]
        G = @views G_threads[:, Threads.threadid()]

        fill!(C_alpha, 0)
        fill!(C, 0)

        @turbo @. A = _X * -1
        @turbo @. G = _X * 3

        feature_index_start = 0

        for dilation_index in eachindex(dilations)
            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) ÷ 2
            num_features_this_dilation = num_features_per_dilation[dilation_index]

            copy!(C_alpha, A)

            s = dilation + 1
            e = input_length - padding

            for idx in 1:(9 ÷ 2)
                d = e + dilation * (idx - 1)
                @views C_alpha[end - d + 1:end] += A[begin:d]
            end

            for idx in (9 ÷ 2) + 2:9
                d = s + dilation * (idx - ((9 ÷ 2) + 2))
                @views C_alpha[begin:end - d + 1] += A[d:end]
            end

            _padding0 = (dilation_index - 1) % 2
            for kernel_index in 1:NUM_KERNELS
                feature_index_end = feature_index_start + num_features_this_dilation

                copy!(C, C_alpha)

                for idx in @views INDICES[:, kernel_index]
                    if idx < 5
                        d = e + dilation * (idx - 1)
                        @views C[end - d + 1:end] += G[begin:d]
                    elseif idx > 5
                        d = s + dilation * (idx - ((9 ÷ 2) + 2))
                        @views C[begin:end - d + 1] += G[d:end]
                    else
                        C += G
                    end
                end

                _padding1 = (_padding0 + (kernel_index - 1)) % 2
                if _padding1 === 0
                    for feature_count in 1:num_features_this_dilation
                        features[feature_index_start + feature_count, example_index] = sum(C .> biases[feature_index_start + feature_count]) / length(C)
                    end
                else
                    for feature_count in 1:num_features_this_dilation
                        features[feature_index_start + feature_count, example_index] = sum(@views C[padding + 1:end - padding] .> biases[feature_index_start + feature_count]) / ((length(C) - padding) - (padding + 1) + 1)
                    end
                end

                feature_index_start = feature_index_end
            end
        end
    end

    return features
end


MLJModelInterface.@mlj_model mutable struct MiniRocketModel <: MLJModelInterface.Unsupervised
    num_features::Unsigned = Unsigned(10_000)::(84 <= _)
    max_dilations_per_kernel::Unsigned = Unsigned(32)::(0 < _)
    rng::AbstractRNG = GLOBAL_RNG
    shuffled::Bool = false
end

function MLJModelInterface.reformat(::MiniRocketModel, (X, type))
    @assert type in (:row_based, :column_based)

    (MLJModelInterface.matrix(X, transpose=type == :row_based),)
end
MLJModelInterface.selectrows(::MiniRocketModel, I, Xmatrix) = (view(Xmatrix, :, I),)

function MLJModelInterface.fit(model::MiniRocketModel, verbosity, X::AbstractMatrix{T})::Tuple{NamedTuple{(:dilations, :num_features_per_dilation, :biases), Tuple{Vector{Int64}, Vector{Int64}, Vector{T}}}, Nothing, Nothing} where {T <: AbstractFloat}
    f_res = fit(X, num_features = model.num_features, max_dilations_per_kernel = model.max_dilations_per_kernel, shuffled = model.shuffled, rng = model.rng)
    return f_res, nothing, nothing
end

function MLJModelInterface.transform(model::MiniRocketModel, fitresult::NamedTuple{(:dilations, :num_features_per_dilation, :biases), Tuple{Vector{Int64}, Vector{Int64}, Vector{T}}}, Xnew::AbstractMatrix{T})::AbstractMatrix{T} where {T <: AbstractFloat}
    return transform(Xnew, dilations = fitresult.dilations, num_features_per_dilation = fitresult.num_features_per_dilation, biases = fitresult.biases)
end

function MLJModelInterface.fitted_params(::MiniRocketModel, fitresult::NamedTuple{(:dilations, :num_features_per_dilation, :biases), Tuple{Vector{Int64}, Vector{Int64}, Vector{T}}})::NamedTuple{(:dilations, :num_features_per_dilation, :biases), Tuple{Vector{Int64}, Vector{Int64}, Vector{T}}} where {T <: AbstractFloat}
    return fitresult
end

end
