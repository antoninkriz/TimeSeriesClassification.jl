module DataSets

include("utils.jl")
include("reader.jl")
include("loader.jl")

export read_ts_file
using ._Reader: read_ts_file

export load_dataset
using ._Loader: load_dataset, dataset_flatten_to_matrix

# List of available data set loaders
module _Loaders
include("Loaders/UCRArchive.jl")
end

export UCRArchive
using ._Loaders: UCRArchive

end