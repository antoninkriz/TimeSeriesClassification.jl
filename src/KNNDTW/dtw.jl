module _DTW

using .._Utils: euclidean_distance

export DTWType,
    dtw!,
    dtw_with_itakura_max_slope,
    dtw_with_itakura_max_slope!,
    dtw_with_sakoe_chiba_radius,
    dtw_with_sakoe_chiba_radius!

abstract type DTWType end

mutable struct DTW{T <: AbstractFloat, Tfun <: Function} <: DTWType
    distance::Tfun
    matrix::Matrix{T}
end

DTW{T}(; distance::Tfun = euclidean_distance, matrix::Matrix{T} = T[;;]) where {T <: AbstractFloat, Tfun <: Function} =
    DTW{T, Tfun}(distance, matrix)

mutable struct DTWSakoeChiba{T <: AbstractFloat, Tfun <: Function} <: DTWType
    distance::Tfun
    matrix::Matrix{T}
    radius::Int64
end

DTWSakoeChiba{T}(;
    distance::Tfun = euclidean_distance,
    matrix::Matrix{T} = T[;;],
    radius::Int64 = 0,
) where {T <: AbstractFloat, Tfun <: Function} = DTWSakoeChiba{T, Tfun}(distance, matrix, radius)

mutable struct DTWItakura{T <: AbstractFloat, Tfun <: Function} <: DTWType
    distance::Tfun
    matrix::Matrix{T}
    slope::Float64
end

function DTWItakura{T}(;
    distance::Tfun = euclidean_distance,
    matrix::Matrix{T} = T[;;],
    slope::Float64 = 1.0,
) where {T <: AbstractFloat, Tfun <: Function}
    @assert slope >= 1.0

    DTWItakura{T, Tfun}(distance, matrix, slope)
end

@inbounds function dtw!(model::Tdtw, x::AbstractVector{T}, y::AbstractVector{T})::T where {T <: AbstractFloat, Tdtw <: DTW}
    row_count, col_count = length(x), length(y)

    # Julia is column major, to make things faster let longer timeseries = columns and shorter timeseries = rows
    if row_count < col_count
        row_count, col_count = col_count, row_count
        x, y = y, x
    end

    if any(size(model.matrix) .< (row_count, col_count))
        model.matrix = fill(prevfloat(typemax(T)), max(row_count, size(model.matrix, 1)), max(col_count, size(model.matrix, 2)))
    else
        fill!(model.matrix, prevfloat(typemax(T)))
    end

    model.matrix[1, 1] = model.distance(x[1], y[1])

    for i in 2:row_count
        model.matrix[i, 1] = model.matrix[i-1, 1] + model.distance(x[i], y[1])
    end

    for i in 2:col_count
        model.matrix[1, i] = model.matrix[1, i-1] + model.distance(x[1], y[i])
    end

    for c in 2:col_count, r in 2:row_count
        model.matrix[r, c] =
            model.distance(x[r], y[c]) + min(model.matrix[r-1, c], model.matrix[r, c-1], model.matrix[r-1, c-1])
    end

    return sqrt(model.matrix[row_count, col_count])
end

@inbounds function dtw!(model::Tdtw, x::AbstractVector{T}, y::AbstractVector{T})::T where {T <: AbstractFloat, Tdtw <: DTWSakoeChiba}
    row_count, col_count = length(x), length(y)

    # Julia is column major, to make things faster let longer timeseries = columns and shorter timeseries = rows
    if row_count < col_count
        row_count, col_count = col_count, row_count
        x, y = y, x
    end

    if model.matrix === nothing || any(size(model.matrix) .< (row_count, col_count))
        model.matrix = fill(prevfloat(typemax(T)), max(row_count, size(model.matrix, 1)), max(col_count, size(model.matrix, 2)))
    else
        fill!(model.matrix, prevfloat(typemax(T)))
    end

    s = Int64(model.radius)

    model.matrix[1, 1] = model.distance(x[1], y[1])

    for r in 2:min(row_count, 1 + s)
        model.matrix[r, 1] = model.matrix[r-1, 1] + model.distance(x[r], y[1])
    end

    for c in 2:min(col_count, s + 1)
        model.matrix[1, c] = model.matrix[1, c-1] + model.distance(x[1], y[c])
    end

    for c in 2:col_count, r in max(2, c - s):min(row_count, c + s)
        model.matrix[r, c] =
            model.distance(x[r], y[c]) + min(model.matrix[r-1, c], model.matrix[r, c-1], model.matrix[r-1, c-1])
    end

    return sqrt(model.matrix[row_count, col_count])
end

@inbounds function dtw!(model::Tdtw, x::AbstractVector{T}, y::AbstractVector{T})::T where {T <: AbstractFloat, Tdtw <: DTWItakura}
    row_count, col_count = length(x), length(y)

    # Julia is column major, to make things faster let longer timeseries = columns and shorter timeseries = rows
    if row_count < col_count
        row_count, col_count = col_count, row_count
        x, y = y, x
    end

    if model.matrix === nothing || any(size(model.matrix) .< (row_count, col_count))
        model.matrix = fill(prevfloat(typemax(T)), max(row_count, size(model.matrix, 1)), max(col_count, size(model.matrix, 2)))
    else
        fill!(model.matrix, prevfloat(typemax(T)))
    end

    sm_ratio = max(model.slope * (row_count / (col_count - 1)), 1)
    sn_ratio = min((1 / model.slope) * ((row_count - 1) / col_count), 1)

    model.matrix[1, 1] = model.distance(x[1], y[1])

    for i in 2:floor(Int64, (1 - row_count - (sm_ratio * col_count)) / sm_ratio)
        model.matrix[1, i] = model.matrix[1, i-1] + model.distance(x[1], y[i])
    end

    for c in 2:col_count, r in
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
            model.distance(x[r], y[c]) + min(model.matrix[r-1, c], model.matrix[r, c-1], model.matrix[r-1, c-1])
    end

    return sqrt(model.matrix[row_count, col_count])
end

end
