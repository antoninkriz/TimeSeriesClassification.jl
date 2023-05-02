using .._Utils: unzip
using .._Loader: DEFAULT_DIR, load_dataset
import .._Loader

using Logging: @debug
using Downloads: download
using ProgressMeter: AbstractProgress, ProgressUnknown, Progress, update!

export UCRArchive


const URL = "https://www.timeseriesclassification.com/Downloads/Archives/Univariate2018_ts.zip"
const FILE = "Univariate2018_ts.zip"
const ZIP_FOLDER = "Univariate_ts"

abstract type UCRArchive end

function _Loader.load_dataset(::Type{UCRArchive}, name::Symbol, dataset_path::Union{Nothing, AbstractString}=nothing, tmp_path::Union{Nothing, AbstractString}=nothing)
    if dataset_path === nothing
        dataset_path = joinpath(homedir(), DEFAULT_DIR, "UCRArchive")
    end

    dataset_full_path = isabspath(dataset_path) ? dataset_path : joinpath(pwd(), dataset_path)
    if isdir(dataset_full_path)
        @debug "Dataset UCRArchive folder found at \"$dataset_full_path\", skipping download"

        @debug "Reading datasets..."
        trainX, trainY = load_dataset(joinpath(dataset_full_path, ZIP_FOLDER, string(name), "$(name)_TRAIN.ts"))
        testX, testY = load_dataset(joinpath(dataset_full_path, ZIP_FOLDER, string(name), "$(name)_TEST.ts"))
        return trainX, trainY, testX, testY
    end

    @debug "Creating directory for the UCRArchive dataset at \"$dataset_full_path\""
    mkpath(dataset_full_path)

    if tmp_path === nothing
        tmp_path = tempname()
    end
    tmp_full_path = isabspath(tmp_path) ? tmp_path : joinpath(pwd(), tmp_path)
    if isdir(tmp_full_path)
        @debug "Temporary directory for downlaoded files for the UCRArchive found at \"$dataset_full_path\""
    else
        @debug "Creating temporary directory for downlaoded files for the UCRArchive at \"$dataset_full_path\""
        mkpath(tmp_full_path)
    end
    tmp_file = joinpath(tmp_full_path, FILE)

    p::AbstractProgress = ProgressUnknown("Downloading (kB):")
    prog(total::Integer, now::Integer) = begin
        if total != 0 && typeof(p) == ProgressUnknown
            p = Progress(total ÷ 1000, "Downloading (kB):")
        end

        update!(p, now ÷ 1000)
    end
    download(URL, tmp_file, progress=prog)

    unzip(tmp_file, dataset_full_path)

    @debug "Reading datasets..."
    trainX, trainY = load_dataset(joinpath(dataset_full_path, ZIP_FOLDER, string(name), "$(name)_TRAIN.ts"))
    testX, testY = load_dataset(joinpath(dataset_full_path, ZIP_FOLDER, string(name), "$(name)_TEST.ts"))
    return trainX, trainY, testX, testY
end
