module DataSets

include("utils.jl")
include("reader.jl")
include("loader.jl")

include("Loaders/UCRArchive.jl")

export read_ts_file
using ._Reader: read_ts_file

end