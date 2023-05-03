module _Loader

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

function dataset_flatten_to_matrix(dataset::Vector{Vector{Vector{T}}}; interpolate::Bool = false)::Matrix{T} where {T}
    # Is the whole dataset empty?
    isempty(dataset) && return T[;;]

    @assert length(dataset[begin]) == 1 "Dataset has more than one dimension"
    @assert all(x -> length(x) == length(dataset[begin]), dataset) "Dataset contains time series with different number of dimensions"

    # All time series have 0 dimensions?
    isempty(dataset[begin]) && return zeros(T, 0, length(dataset))

    return if interpolate
        @assert false "Interpolating series of different lengths or with NaNs is not implemented yet"
    else
        @assert all(x -> all(y -> length(y) == length(dataset[begin][begin]), x), dataset) "Dataset contains a dimension with series of unequal lengths, consider interpolate=true"
        reduce(hcat, [ts[1] for ts in dataset])
    end
end

end
