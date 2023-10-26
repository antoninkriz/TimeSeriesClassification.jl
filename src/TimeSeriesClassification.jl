module TimeSeriesClassification

import MLJModelInterface

include("MiniRocket/MiniRocket.jl")
include("KNNDTW/KNNDTW.jl")
include("DataSets/DataSets.jl")

export MiniRocketModel
using .MiniRocket: MiniRocketModel

export DTWType, dtw!, DTW, DTWSakoeChiba, DTWItakura, LBType, lower_bound!, LBNone, LBKeogh, KNNDTWModel
using .KNNDTW: DTWType, dtw!, DTW, DTWSakoeChiba, DTWItakura, LBType, lower_bound!, LBNone, LBKeogh, KNNDTWModel

export DataSets
import .DataSets

MLJModelInterface.metadata_pkg.(
    (MiniRocketModel, KNNDTWModel),
    name = "TimeSeriesClassification",
    uuid = "4869f98a-92d2-4a27-bbf6-5599fe134177",
    url = "https://github.com/antoninkriz/julia-ts-classification",
    julia = true,
    license = "MIT",
    is_wrapper = false,
)

MLJModelInterface.metadata_model(
    MiniRocketModel,
    input_scitype = AbstractMatrix{<:MLJModelInterface.Continuous},
    output_scitype = AbstractMatrix{<:MLJModelInterface.Continuous},
    load_path = "TimeSeriesClassification.MiniRocketModel",
)

MLJModelInterface.metadata_model(
    KNNDTWModel,
    input_scitype = Union{
        AbstractVector{<:AbstractVector{<:MLJModelInterface.Continuous}},
        AbstractMatrix{<:MLJModelInterface.Continuous},
    },
    output_scitype = AbstractMatrix{<:MLJModelInterface.Finite},
    load_path = "TimeSeriesClassification.KNNDTWModel",
)

"""
$(MLJModelInterface.doc_header(MiniRocketModel))

# MiniRocketModel

A model type for constructing MiniRocket transformer based on the [MINIROCKET paper](https://arxiv.org/abs/2012.08791) implementing both internal and MLJ model interface.

## Hyperparameters

- `num_features=10_000`:         Total number of transformed features
- `max_dilations_per_kernel=32`: Maximum number of dilations per kernel, this value is intended to be kept at it's default value
- `rng=GLOBAL_RNG`:              Random number generator instance (`::Random.AbstractRNG`)
- `shuffled=false`:              Is the passed in training dataset already shuffled? This might improve performance on already shuffled datasets thanks to non-random sequential access.

## Interfaces

This model supports both MLJ model interface and it's own API.

### Internal API

Crate an instance with default hyperparameters or override them with your own using [`MiniRocket._MiniRocket.MiniRocketModel`](@ref):

```julia
model = MiniRocketModel()
```

#### Training model

A model parameters are built using [`MiniRocket._MiniRocket.fit`](@ref):

```julia
model_params = fit(X_train; num_features = model.num_features, max_dilations_per_kernel = model.max_dilations_per_kernel, shuffled = model.shuffled, rng = model.rng)
```

where `X_train` is a column based matrix of training data.

Column based means, that each time series (sample) is in a column of the matrix.

#### Transforming data

The gathered model parameters can be used for transforming other data using [`MiniRocket._MiniRocket.transform`](@ref):

```julia
dilations, num_features_per_dilation, biases = model_params
X_transformed = transform(X_new; dilations = dilations, num_features_per_dilation = num_features_per_dilation, biases = biases)
```

where `X_train` is a column based matrix of training data, `X_new` is a column based matrix of data to be transformed and `X_transformed` is a column based matrix of transformed data.

Column based means, that each time series (sample) is in a column of the matrix.

### MLJ model API

Crate an instance with default hyperparameters or override them with your own using [`MiniRocket._MiniRocket.MiniRocketModel`](@ref) and build a MLJ machine:

```julia
minirocket_model = MiniRocketModel()
mach = machine(minirocket_model, X_train)

# or when X is column based
mach = machine(minirocket_model, transpose(X_train))
```

`X_train` is a matrix of (row based) training data.

Since this algorithm requires column based data `machine(minirocket_model, X_train)` uses `transpose(...)` to convert the data (therefore possibly without creating a copy) at the cost of performance.
If you already have column based data, passing `machine(minirocket_model, transpose(X_train))` should cancel-out both `transpose(...)` calls, thus make no copies of the data without affecting the performance.

#### Training model

Train the machine using `fit!(mach)`.

#### Transforming data

Transform the data using `transform(mach, X_new)` in case of row based data or `transform(mach, transpose(X_new))` in case of column based data.

The result is always row major (column major, but with `tranpose` applied).  
To convert the result to column major format use `transpose(...)`, which should be without any extra computational cost.

"""
MiniRocketModel

"""
$(MLJModelInterface.doc_header(KNNDTWModel))

# KNNDTWModel

KNNDTWModel is for constructing k-Nearest Neigbors model with Dynamic Time Warping with search space limitations and optional lower bounding.

## Hyperparameters

- `K=1`:                 Number of neighbors
- `weights=:uniform`:    Either `:uniform` or `:distance` based weights of the neighbors.
    - `:uniform`:            All neighbors are weighted equally.
    - `:distance`:           Each neighbor is weighted by it's distance.
- `distance=DTW()`:      DTW distance struct, for example `DTWSakoeChiba` or pure `DTW`.
    - `DTW()`:               Dynamic Time Warping without any constraints.
    - `DTWSakoeChiba()`:     Dynamic Time Warping with Sakoe Chiba bound constraint.
    - `DTWItakura()`:        Dynamic Time Warping without Itakura Parallelogram constraint.
    - You can provide your own metric by subtyping `DTWType`.
- `bounding=LBNone()`:   Lower bounding of the distance using methods like `LBKeogh()`.
    - `LBNone()`:            NO-OP, no lower bouning is being done.
    - `LBKeogh()`:           Estimating distance lower bound of the distance using the [LB_Keogh method](https://www.cs.ucr.edu/~eamonn/LB_Keogh.htm).
    - You can provide your own methofs by subtyping `LBType`.

## Interfaces

This model supports the MLJ model interface.

### MLJ model API

Crate an instance with default hyperparameters or override them with your own using [`KNNDTW._KNNDTW.KNNDTWModel`](@ref) and build a MLJ machine:

```julia
knndtw_model = KNNDTWModel()
mach = machine(knndtw_model, X_train, Y_train)

# or when X is column based
mach = machine(knndtw_model, transpose(X_train), Y_train)
```

`X_train` is either a vector of vectors of training data or a (row based) matrix.

When `X` is a matrix, the data might be copied.
Since this algorithm prefers column based data, and using purely `machine(knndtw_model, X_train, Y_train)` utilizes `tranpose(...)` while creating view over columns to convert the input into algorithm's preferred format, which might affect the final performance of the algorithm.
Column based data passed with `transpose(...)` applied as `machine(knndtw_model, transpose(X_train), Y_train)` is preferred.

#### Training model

Train the machine using `fit!(mach)`.

#### Predicting

To predict "probability" of a class you can use `predict` like `predict(mach, X_new)` for row based data or `predict(mach, tranpose(X_new))` in case of column major data.

To classify the data (to get the most probable class) you can use `predict_mode` in a similar fashion.
"""
KNNDTWModel

end
