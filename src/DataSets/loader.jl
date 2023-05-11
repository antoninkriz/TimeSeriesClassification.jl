module _Loader

using LoopVectorization: @turbo

using .._Reader: read_ts_file, read_ts_file_metadata

export load_dataset, load_dataset_metadata, dataset_flatten_to_matrix, DEFAULT_DIR, AbstractLoader

const DEFAULT_DIR = ".julia_ts_classification"

abstract type AbstractLoader end

"Function to list all available dataset by the loader."
function list_available_datasets(::AbstractLoader)::Vector{Symbol}
    @assert false "Specify a dataset Loader type to list it's available datasets"
end

"
Function to load a dataset in the TS file format form a file.

Read file from an absolute `path`. The values will be of the type `T`.
Missing values, defined by `missing_symbol` will be replaced by `replace_missing_by`.
"
function load_dataset(
    path::AbstractString,
    ::Type{T} = Float64;
    replace_missing_by::T = T(NaN),
    missing_symbol::AbstractString = "?",
)::Tuple{Vector{Vector{Vector{T}}}, Union{Vector{String}, Vector{Int64}}} where {T}
    read_ts_file(path, T, replace_missing_by, missing_symbol)
end

"
Function to load only metadata form a TS file.

Read TS metadata from a file located in follwoing absolute `path`.
"
function load_dataset_metadata(
    path::AbstractString,
)::NamedTuple{(
    :problem_name,
    :dimension,
    :series_length,
    :has_timestamps,
    :has_missing,
    :is_classification,
    :class_labels
), Tuple{String, Int64, Int64, Bool, Bool, Bool, Union{Set{String}, Set{Int64}}}}
    metadata, _, _ = read_ts_file_metadata(path)
    return metadata
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
