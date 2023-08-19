using MLJ
using TSC
using TSC.DataSets.Loaders
import Logging

# Disable most logs
Logging.disable_logging(Logging.Info)

# Precompilation
pre() = begin
    DataSets.list_available_datasets(UCRArchive)
    DataSets.load_dataset_metadata(UCRArchive, :Chinatown)
    trainX, trainY, _, _ = DataSets.load_dataset(UCRArchive, :Chinatown, Float32)
    trainX = DataSets.dataset_flatten_to_matrix(trainX)
    trainY = categorical(trainY, levels=unique(trainY))
    mini = MiniRocketModel()
    mach = machine(mini, (trainX, :column_based))
    fit!(mach, verbosity = 0)
    transform(mach, (trainX, :column_based))
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
        trainX =  DataSets.dataset_flatten_to_matrix(trainX)
        testX =  DataSets.dataset_flatten_to_matrix(testX)
    end

    # Create model
    @time begin
        mini = MiniRocketModel()
        mach = machine(mini, (trainX, :column_based))
    end

    # Fit model
    @time fit!(mach, verbosity = 0)

    # Transform both trainX and testX
    @time begin
        t1 = transform(mach, (trainX, :column_based))
        t2 = transform(mach, (trainX, :column_based))
    end

    # No evaluation for MiniRocket as it's just a transformer
    println("0.0")
end
