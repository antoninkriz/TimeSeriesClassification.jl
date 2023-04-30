module _DTW

using .._Utils: euclidean_distance

export dtw, dtw!, dtw_with_itakura_max_slope, dtw_with_itakura_max_slope!, dtw_with_sakoe_chiba_radius, dtw_with_sakoe_chiba_radius!


abstract type DTWType end

mutable struct DTW{T <: AbstractFloat} <: DTWType
    distance::Function
    matrix::Matrix{T}
end

DTW{T}(;distance::Function = euclidean_distance, matrix::Matrix{T} = T[;;]) where {T <: AbstractFloat} = DTW(distance, matrix)

mutable struct DTWSakoeChiba{T <: AbstractFloat} <: DTWType
    distance::Function
    matrix::Matrix{T}
    radius::Unsigned
end

DTWSakoeChiba{T}(;distance::Function = euclidean_distance, matrix::Matrix{T} = T[;;], radius::Unsigned = 0) where {T <: AbstractFloat} = DTWSakoeChiba(distance, matrix, radius)

mutable struct DTWItakura{T <: AbstractFloat} <: DTWType
    distance::Function
    matrix::Matrix{T}
    slope::Float64
end

function DTWItakura{T}(;distance::Function = euclidean_distance, matrix::Matrix{T} = T[;;], slope::Float64 = 1.0) where {T <: AbstractFloat}
    @assert slope >= 1.0

    DTWItakura(distance, matrix, slope)
end


function dtw(model::DTW, x::AbstractVector{T}, y::AbstractVector{T})::T where {T <: AbstractFloat}
    row_count, col_count = length(x), length(y)

    # Julia is column major, to make things faster let longer timeseries = columns and shorter timeseries = rows
    if row_count < col_count
        row_count, col_count = col_count, row_count
        x, y = y, x
    end

    if model.matrix === nothing || any(size(model.matrix) .< (row_count, col_count))
        model.matrix = fill(prevfloat(typemax(T)), row_count, col_count)
    else
        fill!(model.matrix, prevfloat(typemax(T)))
    end

    model.matrix[1, 1] = model.distance(x[1], y[1])

    for i in 2:row_count
        model.matrix[i, 1] = model.matrix[i - 1, 1] + model.distance(x[i], y[1])
    end

    for i in 2:col_count
        model.matrix[1, i] = model.matrix[1, i - 1] + model.distance(x[1], y[i])
    end

    for c in 2:col_count, r in 2:row_count
        model.matrix[r, c] = model.distance(x[r], y[c]) + min(model.matrix[r - 1, c], model.matrix[r, c - 1], model.matrix[r - 1, c - 1])
    end

    return sqrt(model.matrix[row_count, col_count])
end

function dtw(model::DTWSakoeChiba, x::AbstractVector{T}, y::AbstractVector{T})::T where {T <: AbstractFloat}
    row_count, col_count = length(x), length(y)

    # Julia is column major, to make things faster let longer timeseries = columns and shorter timeseries = rows
    if row_count < col_count
        row_count, col_count = col_count, row_count
        x, y = y, x
    end

    if model.matrix === nothing || any(size(model.matrix) .< (row_count, col_count))
        model.matrix = fill(prevfloat(typemax(T)), row_count, col_count)
    else
        fill!(model.matrix, prevfloat(typemax(T)))
    end

    model.matrix[1, 1] = model.distance(x[1], y[1])

    for i in 2:row_count
        model.matrix[i, 1] = model.matrix[i - 1, 1] + model.distance(x[i], y[1])
    end

    for i in 2:col_count
        model.matrix[1, i] = model.matrix[1, i - 1] + model.distance(x[1], y[i])
    end

    s = Int64(model.radius)
    band = (row_count - col_count) + s
    for c in 2:col_count,  r in max(2, c - s): min(row_count, c + band)
        model.matrix[r, c] = model.distance(x[r], y[c]) + min(model.matrix[r - 1, c], model.matrix[r, c - 1], model.matrix[r - 1, c - 1])
    end

    return sqrt(model.matrix[row_count, col_count])
end

function dtw(model::DTWItakura, x::AbstractVector{T}, y::AbstractVector{T})::T where {T <: AbstractFloat}
    row_count, col_count = length(x), length(y)

    # Julia is column major, to make things faster let longer timeseries = columns and shorter timeseries = rows
    if row_count < col_count
        row_count, col_count = col_count, row_count
        x, y = y, x
    end

    if model.matrix === nothing || any(size(model.matrix) .< (row_count, col_count))
        model.matrix = fill(prevfloat(typemax(T)), row_count, col_count)
    else
        fill!(model.matrix, prevfloat(typemax(T)))
    end

    matrix[1, 1] = model.distance(x[1], y[1])

    for i in 2:row_count
        matrix[i, 1] = matrix[i - 1, 1] + model.distance(x[i], y[1])
    end

    for i in 2:col_count
        matrix[1, i] = matrix[1, i - 1] + model.distance(x[1], y[i])
    end

    sm_ratio = max(model.slope * (row_count / (col_count - 1)), 1)
    sn_ratio = min((1 / model.slope) * ((row_count - 1) / col_count), 1)

    for c in 2:col_count, r in ceil(Int64, max(sn_ratio *  (c - 1) + 1, sm_ratio *  (c - col_count) + row_count)):floor(Int64, min(sm_ratio * (c - 1) + 1, sn_ratio * (c - col_count) + row_count))
        matrix[r, c] = model.distance(x[r], y[c]) + min(matrix[r - 1, c], matrix[r, c - 1], matrix[r - 1, c - 1])
    end

    return sqrt(model.matrix[row_count, col_count])
end

end