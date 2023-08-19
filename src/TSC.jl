module TSC

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
    name = "TSC",
    uuid = "4869f98a-92d2-4a27-bbf6-5599fe134177",
    url = "https://github.com/antoninkriz/julia-ts-classification",
    julia = true,
    license = "MIT",
    is_wrapper = false,
)

MLJModelInterface.metadata_model(
    MiniRocketModel,
    input_scitype = Tuple{AbstractMatrix{<:MLJModelInterface.Continuous}, MLJModelInterface.Unknown},
    output_scitype = AbstractMatrix{<:MLJModelInterface.Continuous},
    load_path = "TSC.MiniRocketModel",
)

MLJModelInterface.metadata_model(
    KNNDTWModel,
    input_scitype = AbstractVector{<:AbstractVector{<:MLJModelInterface.Continuous}},
    output_scitype = AbstractMatrix{<:MLJModelInterface.Finite},
    load_path = "TSC.KNNDTWModel",
)

"""
$(MLJModelInterface.doc_header(MiniRocketModel))

# MiniRocketModel

A model type for constructing MiniRocket transformer based on the [MINIROCKET paper](https://arxiv.org/abs/2012.08791) implementing both internal and MLJ model interface.

## Hyperparameters

- `num_features=10_000`:               Total number of transformed features
- `max_dilations_per_kernel=32`:       Maximum number of dilations per kernel, this value is intended to be kept at it's default value
- `rng=GLOBAL_RNG`:                    Random number generator instance (`::Random.AbstractRNG`)
- `shuffled=false`:                    Is the passed in training dataset already shuffled? This might improve performance on already shuffled datasets thanks to non-random sequential access.

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

#### Transforming data

The gathered model parameters can be used for transforming other data using [`MiniRocket._MiniRocket.transform`](@ref):

```julia
dilations, num_features_per_dilation, biases = model_params
X_transformed = transform(X_new; dilations = dilations, num_features_per_dilation = num_features_per_dilation, biases = biases)
```

### MLJ model API

Crate an instance with default hyperparameters or override them with your own using [`MiniRocket._MiniRocket.MiniRocketModel`](@ref) and build a MLJ machine:

```julia
minirocket_model = MiniRocketModel()
mach = machine(minirocket_model, (X_train, :column_based))
```

You must specify if the data provided are row or column based using the `:column_based` and `:row_based` parameter.

#### Training model

Train the machine using `fit!(mach)`.

#### Transforming data

Transform the data using `transform(mach, (X_new, :column_based))` or using `transform(mach, (X_new, :row_based))` in case of row based data.

"""
MiniRocketModel

"""
$(MLJModelInterface.doc_header(KNNDTWModel))

# KNNDTWModel

KNNDTWModel is for constructing k-Nearest Neigbors model with Dynamic Time Warping with search space limitations and optional lower bounding.

## Hyperparameters

`K=1`:                                   Number of neighbors
`weights=:uniform`:                      Either `:uniform` or `:distance` based weights of the neighbors.
                                             - `:uniform`: All neighbors are weighted equally.
                                             - `:distance`: Each neighbor is weighted by it's distance.
`distance=DTW()`:                        DTW distance struct, for example `DTWSakoeChiba` or pure `DTW`.
                                             - `DTW()`: Dynamic Time Warping without any constraints.
                                             - `DTWSakoeChiba()`: Dynamic Time Warping with Sakoe Chiba bound constraint.
                                             - `DTWItakura()`: Dynamic Time Warping without Itakura Parallelogram constraint.
                                             - You can provide your own metric by subtyping `DTWType`.
`bounding=LBNone()`:                     Lower bounding of the distance using methods like `LBKeogh()`.
                                             - `LBNone()`: NO-OP, no lower bouning is being done.
                                             - `LBKeogh()`: Estimating distance lower bound of the distance using the [LB_Keogh method](https://www.cs.ucr.edu/~eamonn/LB_Keogh.htm).
                                             - You can provide your own methofs by subtyping `LBType`.

## Interfaces

This model supports the MLJ model interface.

### MLJ model API

Crate an instance with default hyperparameters or override them with your own using [`KNNDTW._KNNDTW.KNNDTWModel`](@ref) and build a MLJ machine:

```julia
knndtw_model = KNNDTWModel()
mach = machine(knndtw_model, (X_train, :column_based), Y_train)
```

You must specify if the data provided are row or column based using the `:column_based` and `:row_based` parameter.

#### Training model

Train the machine using `fit!(mach)`.

#### Predicting

To predict "probability" of a class you can use `predict` like `predict(mach, (X_new, :column_based))` for columns based data or using `predict(mach, (X_new, :row_based))` in case of row based data.

To classify the data (to get the most probable class) you can use `predict_mode` in a similar fashion.
"""
KNNDTWModel

end
