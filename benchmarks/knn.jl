using MLJBase
using StatisticalMeasures
using TimeSeriesClassification
using TimeSeriesClassification.DataSets.Loaders
import Logging

# Disable most logs
Logging.disable_logging(Logging.Info)

# Precompilation
pre() = begin
    DataSets.list_available_datasets(UCRArchive)
    DataSets.load_dataset_metadata(UCRArchive, :Chinatown)
    trainX, trainY, _, _ = DataSets.load_dataset(UCRArchive, :Chinatown, Float32)
    trainX = [x[1] for x in trainX]
    trainY = categorical(trainY, levels=unique(trainY))
    knndtw = KNNDTWModel(
        K = 1,
        weights = :uniform,
        distance = DTWSakoeChiba{Float32}(radius=10)
    )
    mach = machine(knndtw, trainX, trainY)
    fit!(mach, verbosity = 0)
    yhat = predict_mode(mach, trainX)
    accuracy(trainY, yhat)
    println("Precompilation done.")
end
pre()

# Get all datasets in the UCR Archive
datasets = DataSets.list_available_datasets(UCRArchive)

for dataset in datasets
    println(dataset)

    # Load metadata
    @time (
        train_problem_name,
        train_dimension,
        train_series_length,
        train_has_timestamps,
        train_has_missing,
        train_is_classification,
        train_class_labels
    ), (
        test_problem_name,
        test_dimension,
        test_series_length,
        test_has_timestamps,
        test_has_missing,
        test_is_classification,
        test_class_labels
    ) = DataSets.load_dataset_metadata(UCRArchive, dataset)
    
    # Skip datasets with missing values and time series of unequal length
    if train_has_missing || test_has_missing || train_series_length == 0 || train_series_length == 0
        continue
    end

    # Load dataset
    @time begin
        trainX, trainY, testX, testY = DataSets.load_dataset(UCRArchive, dataset, Float32)
        trainX =  [x[1] for x in trainX]
        trainY = categorical(trainY, levels=collect(train_class_labels))
        testX =  [x[1] for x in testX]
        testY = categorical(testY, levels=collect(train_class_labels))
    end

    # Create model
    @time begin
        knndtw = KNNDTWModel(
            K = 1,
            weights = :uniform,
            distance = DTWSakoeChiba{Float32}(radius=10)
        )
        mach = machine(knndtw, trainX, trainY)
    end

    # Fit model
    @time fit!(mach, verbosity = 0)

    # Predict
    @time yhat = predict_mode(mach, testX)

    # Evaluate
    println(accuracy(testY, yhat))
end
