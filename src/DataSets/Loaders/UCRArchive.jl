module _UCRArchiveLoader

using ..._Utils: unzip
using ..._Loader: DEFAULT_DIR, load_dataset, load_dataset_metadata, AbstractLoader
import ..._Loader

using Logging: @debug, @info, @warn
using Downloads: download
using ProgressMeter: ProgressUnknown, Progress, update!, cancel, finish!
using StaticArrays: SVector

export UCRArchive

const URL = "https://www.timeseriesclassification.com/Downloads/Archives/Univariate2018_ts.zip"
const FILE = "Univariate2018_ts.zip"
const ZIP_FOLDER = "Univariate_ts"
const AVAILABLE_DATASETS::SVector{128, Symbol} = [
    :ACSF1,
    :Adiac,
    :AllGestureWiimoteX,
    :AllGestureWiimoteY,
    :AllGestureWiimoteZ,
    :ArrowHead,
    :Beef,
    :BeetleFly,
    :BirdChicken,
    :BME,
    :Car,
    :CBF,
    :Chinatown,
    :ChlorineConcentration,
    :CinCECGTorso,
    :Coffee,
    :Computers,
    :CricketX,
    :CricketY,
    :CricketZ,
    :Crop,
    :DiatomSizeReduction,
    :DistalPhalanxOutlineAgeGroup,
    :DistalPhalanxOutlineCorrect,
    :DistalPhalanxTW,
    :DodgerLoopDay,
    :DodgerLoopGame,
    :DodgerLoopWeekend,
    :Earthquakes,
    :ECG200,
    :ECG5000,
    :ECGFiveDays,
    :ElectricDevices,
    :EOGHorizontalSignal,
    :EOGVerticalSignal,
    :EthanolLevel,
    :FaceAll,
    :FaceFour,
    :FacesUCR,
    :FiftyWords,
    :Fish,
    :FordA,
    :FordB,
    :FreezerRegularTrain,
    :FreezerSmallTrain,
    :Fungi,
    :GestureMidAirD1,
    :GestureMidAirD2,
    :GestureMidAirD3,
    :GesturePebbleZ1,
    :GesturePebbleZ2,
    :GunPoint,
    :GunPointAgeSpan,
    :GunPointMaleVersusFemale,
    :GunPointOldVersusYoung,
    :Ham,
    :HandOutlines,
    :Haptics,
    :Herring,
    :HouseTwenty,
    :InlineSkate,
    :InsectEPGRegularTrain,
    :InsectEPGSmallTrain,
    :InsectWingbeatSound,
    :ItalyPowerDemand,
    :LargeKitchenAppliances,
    :Lightning2,
    :Lightning7,
    :Mallat,
    :Meat,
    :MedicalImages,
    :MelbournePedestrian,
    :MiddlePhalanxOutlineAgeGroup,
    :MiddlePhalanxOutlineCorrect,
    :MiddlePhalanxTW,
    :MixedShapesRegularTrain,
    :MixedShapesSmallTrain,
    :MoteStrain,
    :NonInvasiveFetalECGThorax1,
    :NonInvasiveFetalECGThorax2,
    :OliveOil,
    :OSULeaf,
    :PhalangesOutlinesCorrect,
    :Phoneme,
    :PickupGestureWiimoteZ,
    :PigAirwayPressure,
    :PigArtPressure,
    :PigCVP,
    :PLAID,
    :Plane,
    :PowerCons,
    :ProximalPhalanxOutlineAgeGroup,
    :ProximalPhalanxOutlineCorrect,
    :ProximalPhalanxTW,
    :RefrigerationDevices,
    :Rock,
    :ScreenType,
    :SemgHandGenderCh2,
    :SemgHandMovementCh2,
    :SemgHandSubjectCh2,
    :ShakeGestureWiimoteZ,
    :ShapeletSim,
    :ShapesAll,
    :SmallKitchenAppliances,
    :SmoothSubspace,
    :SonyAIBORobotSurface1,
    :SonyAIBORobotSurface2,
    :StarLightCurves,
    :Strawberry,
    :SwedishLeaf,
    :Symbols,
    :SyntheticControl,
    :ToeSegmentation1,
    :ToeSegmentation2,
    :Trace,
    :TwoLeadECG,
    :TwoPatterns,
    :UMD,
    :UWaveGestureLibraryAll,
    :UWaveGestureLibraryX,
    :UWaveGestureLibraryY,
    :UWaveGestureLibraryZ,
    :Wafer,
    :Wine,
    :WordSynonyms,
    :Worms,
    :WormsTwoClass,
    :Yoga,
]

function download_datasets(
    dataset_path::Union{Nothing, AbstractString} = nothing,
    tmp_path::Union{Nothing, AbstractString} = nothing,
    force::Bool = false,
)::String
    if dataset_path === nothing
        dataset_path = joinpath(homedir(), DEFAULT_DIR, "UCRArchive")
    end

    dataset_full_path = isabspath(dataset_path) ? dataset_path : joinpath(pwd(), dataset_path)
    if isdir(dataset_full_path) && !force
        @info "Dataset folder already found at \"$dataset_full_path\", skipping download"
        return dataset_full_path
    end

    if tmp_path === nothing
        tmp_path = tempname()
    end
    tmp_full_path = isabspath(tmp_path) ? tmp_path : joinpath(pwd(), tmp_path)
    if isdir(tmp_full_path)
        @debug "Temporary directory for downlaoding all dataset files at \"$tmp_full_path\""
    else
        @info "Creating temporary directory for downlaoding all dataset files at \"$tmp_full_path\""
        mkpath(tmp_full_path)
    end
    tmp_file = joinpath(tmp_full_path, FILE)

    lk = ReentrantLock()
    p::Union{Nothing, Progress} = nothing
    prog(total::Int, now::Int) = begin
        lock(lk) do
            tKB = total ÷ 1000

            # ProgressUnknown can not be canceled due to a bug (https://github.com/timholy/ProgressMeter.jl/pull/217), so this needs to be hacked around
            if p === nothing
                if total == 0
                    p = Progress(typemax(Int), "Downloading - unknown file size")
                else
                    p = Progress(tKB, "Downloading - $(total / 1000) kB")
                end
            end

            if (p.n == typemax(Int) && total != 0) || (p.n != typemax(Int) && p.n != tKB)
                cancel(p, "", keep = false)
                p = Progress(tKB, "Downloading - $(total / 1000) kB")
            end

            if p.counter != p.n
                update!(p, min(now ÷ 1000, tKB))
            end
        end
    end

    @info "Downloading datasets"
    download(URL, tmp_file, progress = prog)
    finish!(p)

    if isdir(dataset_full_path) && force
        @warn "Deleting the folder before creating a new one at \"$dataset_full_path\""
        @warn "DELETING THIS FOLDER WITH ALL ITS CONTENTS"
        @warn "DELETING THIS FOLDER WITH ALL ITS CONTENTS"
        @warn "DELETING THIS FOLDER WITH ALL ITS CONTENTS"
        @warn "DELETING THIS FOLDER WITH ALL ITS CONTENTS"
        @warn "Waiting for 10 seconds before proceeding"
        sleep(10)

        @info "Removing the folder at \"$dataset_full_path\""
        rm(dataset_full_path, force=true, recursive=true)
    end

    @info "Creating directory for the UCRArchive datasets at \"$dataset_full_path\""
    mkpath(dataset_full_path)

    @info "Unzipping datasets"
    unzip(tmp_file, dataset_full_path)

    @info "Moving datasets to the correct path"
    for dataset_name in readdir(joinpath(dataset_full_path, ZIP_FOLDER))
        mv(joinpath(dataset_full_path, ZIP_FOLDER, dataset_name), joinpath(dataset_full_path, dataset_name))
    end
    rm(joinpath(dataset_full_path, ZIP_FOLDER))

    return dataset_full_path
end

abstract type UCRArchive <: AbstractLoader end

_Loader.list_available_datasets(::Type{UCRArchive})::AbstractVector{Symbol} = AVAILABLE_DATASETS

function _Loader.load_dataset(
    ::Type{UCRArchive},
    name::Symbol,
    dataset_path::Union{Nothing, AbstractString} = nothing,
    tmp_path::Union{Nothing, AbstractString} = nothing;
    force::Bool = false,
)
    dataset_full_path = download_datasets(dataset_path, tmp_path, force)

    @info "Reading datasets..."
    trainX, trainY = load_dataset(joinpath(dataset_full_path, string(name), "$(name)_TRAIN.ts"))
    testX, testY = load_dataset(joinpath(dataset_full_path, string(name), "$(name)_TEST.ts"))
    return trainX, trainY, testX, testY
end

function _Loader.load_dataset_metadata(
    ::Type{UCRArchive},
    name::Symbol,
    dataset_path::Union{Nothing, AbstractString} = nothing,
    tmp_path::Union{Nothing, AbstractString} = nothing;
    force::Bool = false,
)
    dataset_full_path = download_datasets(dataset_path, tmp_path, force)

    train_metadata = load_dataset_metadata(joinpath(dataset_full_path, string(name), "$(name)_TRAIN.ts"))
    test_metadata = load_dataset_metadata(joinpath(dataset_full_path, string(name), "$(name)_TRAIN.ts"))
    return train_metadata, test_metadata
end

end
