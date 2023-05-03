module _Loader

using LoopVectorization: @turbo

using .._Reader: read_ts_file

export load_dataset, dataset_flatten_to_matrix, DEFAULT_DIR, AbstractLoader

const DEFAULT_DIR = ".julia_ts_classification"

abstract type AbstractLoader end

function list_available_datasets(::AbstractLoader)::Vector{Symbol}
    @assert false "Specify a dataset Loader type to list it's available datasets"
end

function load_dataset(
    path::AbstractString,
    ::Type{T} = Float64,
    replace_missing_by::T = NaN64,
    missing_symbol::AbstractString = "?",
)::Tuple{Vector{Vector{Vector{T}}}, Vector{String}} where {T}
    read_ts_file(path, T, replace_missing_by, missing_symbol)
end

function dataset_flatten_to_matrix(dataset::Vector{Vector{Vector{T}}})::Matrix{T} where {T}
    # Is the whole dataset empty?
    isempty(dataset) && return T[;;]

    @assert length(dataset[begin]) == 1 "Dataset has more than one dimension"
    @assert all(x -> length(x) == length(dataset[begin]), dataset) "Dataset contains time series with different number of dimensions"

    # All time series have 0 dimensions?
    isempty(dataset[begin]) && return zeros(T, 0, length(dataset))

    @assert all(x -> all(y -> length(y) == length(dataset[begin][begin]), x), dataset) "Dataset contains a dimension with series of unequal lengths, consider interpolate=true"
    reduce(hcat, [ts[1] for ts in dataset])
end

function ts_linear_interpolate_missing!(arr::Vector{T}, is_missing::Function = isnan) where {T <: Number}
    s = findfirst(x -> !is_missing(x), arr)
    t = arr[s]
    @inbounds arr[1:s - 1] .= t

    c = 0
    @inbounds for i in s+1:length(arr)
        if !is_missing(arr[i])
            n = (arr[i]-t) / (c+1)
            for j in 1:c
                arr[s+j] = t + n * j
            end
            s = i
            t = arr[s]
            c = 0
        else
            c += 1
        end
    end

    @inbounds arr[s+1:end] .= t
    arr
end

end
