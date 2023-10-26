module _DTW

using .._Utils: euclidean_distance
using Base: RefValue

export DTWType,
    dtw!,
    dtw_with_itakura_max_slope,
    dtw_with_itakura_max_slope!,
    dtw_with_sakoe_chiba_radius,
    dtw_with_sakoe_chiba_radius!

abstract type DTWType end

"Structure implementing vanilla DTW."
mutable struct DTW{T <: AbstractFloat} <: DTWType
    matrix::Matrix{T}
end

"Constructor to instantiate vanilla DTW structure."
DTW{T}() where {T <: AbstractFloat} =
    DTW{T}(T[;;])

"Structure implementing DTW with Sakoe-Chiba band."
mutable struct DTWSakoeChiba{T <: AbstractFloat} <: DTWType
    matrix::Matrix{T}
    radius::Int64
    match_sizes::Bool
end

"
Constructor to instantiate Sakoe-Chiba DTW structure.

`radius` sets the radius of the band.
`match_sizes` allow the radius to extend in the direction of the opposing corner when the matrices aren't equally sized.
"
DTWSakoeChiba{T}(;
    radius::Int64 = 0,
    match_sizes::Bool = false
) where {T <: AbstractFloat} = DTWSakoeChiba{T}(T[;;], radius, match_sizes)

"Structure implementing DTW with Itakura parallelogram."
mutable struct DTWItakura{T <: AbstractFloat} <: DTWType
    matrix::Matrix{T}
    slope::Float64
end

"
Constructor to instantiate Itakura DTW structure.

`slope` sets the slope of the parallelogram.
"
function DTWItakura{T}(;
    slope::Float64 = 1.0,
) where {T <: AbstractFloat}
    @assert slope >= 1.0

    DTWItakura{T}(T[;;], slope)
end

"Function to calucate vanilla DTW distance between `x` and `y`."
function dtw!(model::DTW{T}, x::AbstractVector{T}, y::AbstractVector{T})::T where {T <: AbstractFloat}
    row_count, col_count = length(x), length(y)

    # Julia is column major, to make things faster let longer timeseries = columns and shorter timeseries = rows
    if row_count < col_count
        row_count, col_count = col_count, row_count
        x, y = y, x
    end

    if size(model.matrix, 1) < row_count || size(model.matrix, 2) < col_count
        model.matrix = fill(prevfloat(typemax(T)), max(row_count, size(model.matrix, 1)), max(col_count, size(model.matrix, 2)))
    else
        fill!(model.matrix, prevfloat(typemax(T)))
    end

    @inbounds model.matrix[1, 1] = euclidean_distance(x[1], y[1])

    @inbounds for i in 2:row_count
        model.matrix[i, 1] = model.matrix[i-1, 1] + euclidean_distance(x[i], y[1])
    end

    @inbounds for i in 2:col_count
        model.matrix[1, i] = model.matrix[1, i-1] + euclidean_distance(x[1], y[i])
    end

    @inbounds for c in 2:col_count
        for r in 2:row_count
            model.matrix[r, c] =
                euclidean_distance(x[r], y[c]) + min(model.matrix[r-1, c], model.matrix[r, c-1], model.matrix[r-1, c-1])
        end
    end

    @inbounds return model.matrix[row_count, col_count]
end

"Function to calucate Sakoe-Chiba band limited DTW distance between `x` and `y`."
function dtw!(model::DTWSakoeChiba{T}, x::AbstractVector{T}, y::AbstractVector{T})::T where {T <: AbstractFloat}
    row_count, col_count = length(x), length(y)

    # Julia is column major, to make things faster let longer timeseries = columns and shorter timeseries = rows
    if row_count < col_count
        row_count, col_count = col_count, row_count
        x, y = y, x
    end

    if size(model.matrix, 1) < row_count || size(model.matrix, 2) < col_count
        model.matrix = fill(prevfloat(typemax(T)), max(row_count, size(model.matrix, 1)), max(col_count, size(model.matrix, 2)))
    else
        fill!(model.matrix, prevfloat(typemax(T)))
    end

    s = Int64(model.radius)
    band = (model.match_sizes ? (row_count - col_count) : 0) + s

    @inbounds model.matrix[1, 1] = euclidean_distance(x[1], y[1])

    @inbounds for r in 2:min(row_count, 1 + band)
        model.matrix[r, 1] = model.matrix[r-1, 1] + euclidean_distance(x[r], y[1])::T
    end

    @inbounds for c in 2:min(col_count, s + 1)
        model.matrix[1, c] = model.matrix[1, c-1] + euclidean_distance(x[1], y[c])::T
    end

    @inbounds for c in 2:col_count
        for r in max(2, c - s):min(row_count, c + band)
            model.matrix[r, c] =
                euclidean_distance(x[r], y[c]) + min(model.matrix[r-1, c], model.matrix[r, c-1], model.matrix[r-1, c-1])
        end
    end

    @inbounds return model.matrix[row_count, col_count]
end

"Function to calucate Itakura parallelogram limited DTW distance between `x` and `y`."
function dtw!(model::DTWItakura{T}, x::AbstractVector{T}, y::AbstractVector{T})::T where {T <: AbstractFloat}
    row_count, col_count = length(x), length(y)

    # Julia is column major, to make things faster let longer timeseries = columns and shorter timeseries = rows
    if row_count < col_count
        row_count, col_count = col_count, row_count
        x, y = y, x
    end

    if size(model.matrix, 1) < row_count || size(model.matrix, 2) < col_count
        model.matrix = fill(prevfloat(typemax(T)), max(row_count, size(model.matrix, 1)), max(col_count, size(model.matrix, 2)))
    else
        fill!(model.matrix, prevfloat(typemax(T)))
    end

    sm_ratio = max(model.slope * (row_count / (col_count - 1)), 1)
    sn_ratio = min((1 / model.slope) * ((row_count - 1) / col_count), 1)

    @inbounds model.matrix[1, 1] = euclidean_distance(x[1], y[1])::T

    @inbounds for i in 2:floor(Int64, (1 - row_count - (sm_ratio * col_count)) / sm_ratio)
        model.matrix[1, i] = model.matrix[1, i-1] + euclidean_distance(x[1], y[i])::T
    end

    @inbounds for c in 2:col_count
        for r in
        ceil(
            Int64,
            max(
                sn_ratio * (c - 1) + 1,
                sm_ratio * (c - col_count) + row_count
            )
        ):floor(
            Int64,
            min(
                sm_ratio * (c - 1) + 1,
                sn_ratio * (c - col_count) + row_count
            )
        )
            model.matrix[r, c] =
                euclidean_distance(x[r], y[c]) + min(model.matrix[r-1, c], model.matrix[r, c-1], model.matrix[r-1, c-1])
        end
    end

    @inbounds return model.matrix[row_count, col_count]
end

end
