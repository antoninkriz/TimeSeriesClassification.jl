using StaticArrays: SMatrix
using Statistics: quantile, mean
using Base.Threads: @threads



const NUM_KERNELS = 84


const INDICES::SMatrix{3,NUM_KERNELS,Integer} = [
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 5 5 5 6
    1 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 4 4 4 4 5 5 5 6 6 7 2 2 2 2 2 2 3 3 3 3 3 4 4 4 4 5 5 5 6 6 7 3 3 3 3 3 4 4 4 4 5 5 5 6 6 7 4 4 4 4 5 5 5 6 6 7 5 5 5 6 6 7 6 6 7 7
    2 3 4 5 6 7 8 3 4 5 6 7 8 4 5 6 7 8 5 6 7 8 6 7 8 7 8 8 3 4 5 6 7 8 4 5 6 7 8 5 6 7 8 6 7 8 7 8 8 4 5 6 7 8 5 6 7 8 6 7 8 7 8 8 5 6 7 8 6 7 8 7 8 8 6 7 8 7 8 8 7 8 8 8
] .+ 1


function fit_biases(X::AbstractMatrix{T}, dilations::AbstractVector{Unsigned}, num_features_per_dilation::AbstractVector{Unsigned}, quantiles::AbstractVector{T})::AbstractVector{T} where {T <: AbstractFloat}
    input_length, num_examples = size(X)

    num_features = NUM_KERNELS * sum(num_features_per_dilation)
    biases = zeros(T, num_features)

    feature_index_start = 1

    # This will work ONLY with the assumption that the dataset is already shuffled!
    # Also, this index starts from zero because it's used with modulo operation.
    idx_nonrand = 0

    for dilation_index = eachindex(dilations)
        dilation = Signed(dilations[dilation_index])
        padding = ((9 - 1) * dilation) ÷ 2
        num_features_this_dilation = @views num_features_per_dilation[dilation_index]

        for kernel_index = 1:NUM_KERNELS
            feature_index_end = feature_index_start + num_features_this_dilation - 1

            # Single feature (column)
            _X = @views X[:, (idx_nonrand % num_examples) + 1]
            idx_nonrand += 1

            A = -_X
            G = _X * 3

            C_alpha = copy(A)

            C_gamma = zeros(T, input_length, 9)
            C_gamma[:, (9 ÷ 2) + 1] = G

            s = dilation + 1
            e = input_length - padding

            for gamma_index in 1:(9 ÷ 2)
                C_alpha[end - e + 1:end] += @views A[begin:e]
                C_gamma[end - e + 1:end, gamma_index] = @views G[begin:e]
                e += dilation
            end

            for gamma_index in (9 ÷ 2) + 1:9
                C_alpha[begin:end - s + 1] += @views A[s:end]
                C_gamma[begin:end - s + 1, gamma_index] = @views G[s:end]
                s += dilation
            end

            i0, i1, i2 = @views INDICES[:, kernel_index]
            C = C_alpha + @views C_gamma[:, i0] + @views C_gamma[:, i1] + @views C_gamma[:, i2]

            q = quantiles[feature_index_start:feature_index_end]
            biases[feature_index_start:feature_index_end] = quantile(C, quantiles[feature_index_start:feature_index_end])

            feature_index_start = feature_index_end + 1
        end
    end

    return biases
end


function fit_dilations(input_length::Unsigned, num_features::Unsigned, max_dilations_per_kernel::Unsigned)::Tuple{AbstractVector{Unsigned},AbstractVector{Unsigned}}
    num_features_per_kernel = num_features ÷ NUM_KERNELS
    true_max_dilations_per_kernel = min(num_features_per_kernel, max_dilations_per_kernel)
    multiplier = num_features_per_kernel / true_max_dilations_per_kernel

    max_exponent = log2((input_length - 1) / (9 - 1))

    dilations, num_features_per_dilation = sorted_unique_counts(floor.(Unsigned, logspace(0, max_exponent, true_max_dilations_per_kernel, base = 2)))

    num_features_per_dilation = floor.(Unsigned, (num_features_per_dilation * multiplier))

    remainder = num_features_per_kernel - sum(num_features_per_dilation)
    i = 0
    while remainder > 0
        @inbounds num_features_per_dilation[i + 1] += 1
        i = (i + 1) % length(num_features_per_dilation)
        remainder -= 1
    end

    return dilations, num_features_per_dilation
end


function fit(X::AbstractMatrix{T}; num_features::Unsigned=Unsigned(10_000), max_dilations_per_kernel::Unsigned=Unsigned(32))::Tuple{AbstractVector{Unsigned}, AbstractVector{Unsigned}, AbstractVector{T}} where {T <: AbstractFloat}
    X = convert(Matrix{T}, X)

    # Number of samples, sample length
    input_length::Unsigned = size(X, 1)

    dilations, num_features_per_dilation = fit_dilations(input_length, num_features, max_dilations_per_kernel)

    num_features_per_kernel = sum(num_features_per_dilation)

    quantiles = map(x -> (x * ((sqrt(T(5)) + 1) / 2)) % 1, 1:(NUM_KERNELS * num_features_per_kernel))

    biases = fit_biases(X, dilations, num_features_per_dilation, quantiles)

    return dilations, num_features_per_dilation, biases
end


function transform(X::AbstractMatrix{T}; dilations::AbstractVector{Unsigned}, num_features_per_dilation::AbstractVector{Unsigned}, biases::AbstractVector{T}) where {T <: AbstractFloat}
    input_length, num_examples = size(X)

    features = zeros(T, (num_examples, NUM_KERNELS * sum(num_features_per_dilation)))

    for example_index in 1:num_examples
        _X = @views X[:, example_index]

        A = -_X
        G = _X * 3

        feature_index_start = 0

        for dilation_index in eachindex(dilations)
            dilation = Signed(dilations[dilation_index])
            padding = ((9 - 1) * dilation) ÷ 2
            num_features_this_dilation = @views num_features_per_dilation[dilation_index]

            C_alpha = copy(A)

            C_gamma = zeros(T, input_length, 9)
            C_gamma[:, (9 ÷ 2) + 1] = G

            s = dilation + 1
            e = input_length - padding

            for gamma_index in 1:(9 ÷ 2)
                C_alpha[end - e + 1:end] += @views A[begin:e]
                C_gamma[end - e + 1:end, gamma_index] = @views G[begin:e]
                e += dilation
            end

            for gamma_index in (9 ÷ 2) + 1: 9
                C_alpha[begin:end - s + 1] += @views A[s:end]
                C_gamma[begin:end - s + 1, gamma_index] = @views G[s:end]
                s += dilation
            end

            for kernel_index in 1:NUM_KERNELS
                feature_index_end = feature_index_start + num_features_this_dilation

                i0, i1, i2 = @views INDICES[:, kernel_index]
                C = C_alpha + @views C_gamma[:, i0] + @views C_gamma[:, i1] + @views C_gamma[:, i2]

                if (((dilation_index - 1) % 2) + (kernel_index - 1)) % 2 == 1
                    for feature_count in 1:num_features_this_dilation - 1
                        features[example_index, feature_index_start + feature_count] = mean(T, C .> biases[feature_index_start + feature_count])
                    end
                else
                    for feature_count in 1:num_features_this_dilation - 1
                        features[example_index, feature_index_start + feature_count] = mean(T, C[padding + 1:end - padding] .> biases[feature_index_start + feature_count])
                    end
                end

                feature_index_start = feature_index_end + 1
            end
        end
    end

    return biases
end
