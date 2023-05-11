import os
import sys
from timeit import default_timer as timer
from sklearn.metrics import accuracy_score
from sktime.datasets import load_UCR_UEA_dataset
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from wrapt_timeout_decorator import timeout


# Splitting datasets
def chunkify(lst,n):
    return [lst[i::n] for i in range(n)]


# Precompilation
def pre():
    dir_path = os.path.expanduser('~/.sktime_ucr')
    trainX, trainY = load_UCR_UEA_dataset('Chinatown', split='train', return_X_y=True, return_type='numpy3D', extract_path=dir_path)
    knndtw = KNeighborsTimeSeriesClassifier(
        1,
        weights='uniform',
        algorithm='brute',
        distance='dtw',
        distance_params={
            'window': 10/len(trainX[0][0])
        },
        n_jobs=-1
    )
    knndtw.fit(trainX, trainY)
    yhat = knndtw.predict(trainX)
    accuracy_score(trainY, yhat)
    print("Precompilation done.")
pre()


# All datasets
datasets = ['ACSF1', 'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'BME', 'Car', 'CBF', 'Chinatown', 'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'Computers', 'CricketX', 'CricketY', 'CricketZ', 'Crop', 'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'Earthquakes', 'ECG200', 'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'EOGHorizontalSignal', 'EOGVerticalSignal', 'EthanolLevel', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords', 'Fish', 'FordA', 'FordB', 'FreezerRegularTrain', 'FreezerSmallTrain', 'Fungi', 'GunPoint', 'GunPointAgeSpan', 'GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 'Ham', 'HandOutlines', 'Haptics', 'Herring', 'HouseTwenty', 'InlineSkate', 'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'InsectWingbeatSound', 'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7', 'Mallat', 'Meat', 'MedicalImages', 'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain', 'MoteStrain', 'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2', 'OliveOil', 'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme', 'PigAirwayPressure', 'PigArtPressure', 'PigCVP', 'Plane', 'PowerCons', 'ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices', 'Rock', 'ScreenType', 'SemgHandGenderCh2', 'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SmoothSubspace', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'StarLightCurves', 'Strawberry', 'SwedishLeaf', 'Symbols', 'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG', 'TwoPatterns', 'UMD', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ', 'Wafer', 'Wine', 'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga']

# Timing function
def time(t):
    t2 = timer()
    print(f'  {t2 - t} seconds')
    return timer()

# For multiple processing
if len(sys.argv) == 3:
    batch_n = int(sys.argv[1])
    batch_i = int(sys.argv[2])
else:
    batch_n = 1
    batch_i = 0

chunks = list(chunkify(datasets, batch_n))
for dataset in chunks[batch_i]:
    print(dataset)
    t = timer()

    # Load metadata
    # Skip first timer since the datasets for Python are already filtered
    t = time(t)

    # Load dataset
    dir_path = os.path.expanduser('~/.sktime_ucr')
    trainX, trainY = load_UCR_UEA_dataset(dataset, split='train', return_X_y=True, return_type='numpy3D', extract_path=dir_path)
    testX, testY = load_UCR_UEA_dataset(dataset, split='test', return_X_y=True, return_type='numpy3D', extract_path=dir_path)
    t = time(t)

    # Create model
    knndtw = KNeighborsTimeSeriesClassifier(
        1,
        weights='uniform',
        algorithm='brute',
        distance='dtw',
        distance_params={
            'window': 10/len(trainX[0][0])
        },
        n_jobs=-1
    )
    t = time(t)

    # Fit model
    knndtw.fit(trainX, trainY)
    t = time(t)

    # Predict, timeout after 30 minutes
    @timeout(3 * 60)
    def predict():
        return knndtw.predict(testX)
    
    try:
        yhat = predict()
        t = time(t)
        # Evaluate
        print(accuracy_score(testY, yhat))
    except:
        print("  -1\n0.0")

