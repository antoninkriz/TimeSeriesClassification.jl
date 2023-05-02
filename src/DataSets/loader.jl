module _Loader

using .._Reader: read_ts_file

export load_dataset, DEFAULT_DIR


const DEFAULT_DIR = ".julia_ts_classification"


function load_dataset(path::AbstractString, ::Type{T} = Float64, replace_nan::T = NaN64, missing_symbol::AbstractString="?")::Tuple{Vector{Vector{Vector{T}}}, Vector{String}} where {T}
    read_ts_file(path, T, replace_nan, missing_symbol)
end

end