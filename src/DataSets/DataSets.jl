module DataSets

include("utils.jl")
include("reader.jl")
include("loader.jl")

export read_ts_file
using ._Reader: read_ts_file

export load_dataset, list_available_datasets, dataset_flatten_to_matrix
using ._Loader: list_available_datasets, load_dataset, dataset_flatten_to_matrix

# List of available data set loaders
export Loaders
module Loaders

include("Loaders/UCRArchive.jl")
export UCRArchive
using ._UCRArchiveLoader: UCRArchive

end

end
