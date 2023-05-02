using ._Utils: unzip
using ._Loader: load_dataset, DEFAULT_DIR

using Logging: @debug
using Downloads: download
using ProgressMeter: AbstractProgress, ProgressUnknown, Progress

const URL = "https://www.timeseriesclassification.com/Downloads/Archives/Univariate2018_ts.zip"
const FILE = "Univariate2018_ts.zip"

abstract type UCRArchive end

function load_dataset(::UCRArchive, name::Symbol, dataset_path::Some{AbstractString}=nothing, tmp_path::Some{AbstractString}=nothing)
    if dataset_path === nothing
        dataset_path = joinpath(homedir(), DEFAULT_DIR, "UCRArchive")
    end

    dataset_full_path = isabspath(dataset_path) ? dataset_path : joinpath(pwd(), dataset_path)
    if isdir(dataset_full_path)
        @debug "Dataset UCRArchive folder found at \"$dataset_full_path\", skipping download"

        @debug "Reading datasets..."
        trainX, trainY = load_dataset(joinpath(dataset_full_path, string(name), "$name_TRAIN.ts"))
        testX, testY = load_dataset(joinpath(dataset_full_path, string(name), "$name_TEST.ts"))
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
            p = Progress(total / 1000, "Downloading (kB):")
        end

        update!(p, now / 1000)
    end
    download(URL, tmp_file, progress=prog)

    unzip(tmp_file, dataset_full_path)

    @debug "Reading datasets..."
    trainX, trainY = load_dataset(joinpath(dataset_full_path, string(name), "$name_TRAIN.ts"))
    testX, testY = load_dataset(joinpath(dataset_full_path, string(name), "$name_TEST.ts"))
    return trainX, trainY, testX, testY
end
