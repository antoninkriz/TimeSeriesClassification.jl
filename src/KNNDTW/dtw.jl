function dist(a, b)
    return (a - b)^2
end

function dtw_with_sakoe_chiba_radius(x::Vector{T}, y::Vector{T}; sakoe_chiba_radius::Unsigned) where {T <: AbstractFloat}
    row_count, col_count = length(x), length(y)

    # Julia is column major, to make things faster let longer timeseries = columns and shorter timeseries = rows
    if row_count < col_count
        row_count, col_count = col_count, row_count
        x, y = y, x
    end

    # TODO: Reduce memory allocations by checking for previous allocations
    M = fill(prevfloat(typemax(T)), row_count, col_count)
    M[1, 1] = dist(x[1], y[1])

    for i in 2:row_count
        M[i, 1] = M[i - 1, 1] + dist(x[i], y[1])
    end

    for i in 2:col_count
        M[1, i] = M[1, i - 1] + dist(x[1], y[i])
    end

    s = Int64(sakoe_chiba_radius)
    band = (row_count - col_count) + s
    for c in 2:col_count
        for r in max(2, c - s): min(row_count, c + band)
            M[r, c] = dist(x[r], y[c]) + min(M[r - 1, c], M[r, c - 1], M[r - 1, c - 1])
        end
    end

    return sqrt(M[end, end])
end

function dtw_with_itakura_max_slope(x::Vector{T}, y::Vector{T}; itakura_max_slope::T2) where {T <: AbstractFloat, T2 <: AbstractFloat}
    row_count, col_count = length(x), length(y)

    # Julia is column major, to make things faster let longer timeseries = columns and shorter timeseries = rows
    if row_count < col_count
        row_count, col_count = col_count, row_count
        x, y = y, x
    end

    # TODO: Reduce memory allocations by checking for previous allocations
    M = fill(prevfloat(typemax(T)), row_count, col_count)
    M[1, 1] = dist(x[1], y[1])

    for i in 2:row_count
        M[i, 1] = M[i - 1, 1] + dist(x[i], y[1])
    end

    for i in 2:col_count
        M[1, i] = M[1, i - 1] + dist(x[1], y[i])
    end

    # TODO: Fix itakura to actually work properly
    s_ratio = itakura_max_slope * (row_count / col_count)
    sinv_ratio = (1 / itakura_max_slope) * (row_count / col_count)

    for c in 2:col_count
        for r in ceil(Int64, max(sinv_ratio * (c - 1) + 1, s_ratio * (c - row_count) + col_count)):floor(Int64, min(s_ratio * (c - 1) + 1, sinv_ratio * (c - row_count) + col_count))
            M[r, c] = dist(x[r], y[c]) + min(M[r - 1, c], M[r, c - 1], M[r - 1, c - 1])
        end
    end

    return sqrt(M[end, end])
end

function dtw(x::Vector{T}, y::Vector{T}) where {T <: AbstractFloat}
    row_count, col_count = length(x), length(y)

    # Julia is column major, to make things faster let longer timeseries = columns and shorter timeseries = rows
    if row_count < col_count
        row_count, col_count = col_count, row_count
        x, y = y, x
    end

    # TODO: Reduce memory allocations by checking for previous allocations
    M = fill(prevfloat(typemax(T)), row_count, col_count)
    M[1, 1] = dist(x[1], y[1])

    for i in 2:row_count
        M[i, 1] = M[i - 1, 1] + dist(x[i], y[1])
    end

    for i in 2:col_count
        M[1, i] = M[1, i - 1] + dist(x[1], y[i])
    end

    for c in 2:col_count, r in 2:row_count
        M[r, c] = dist(x[r], y[c]) + min(M[r - 1, c], M[r, c - 1], M[r - 1, c - 1])
    end

    return sqrt(M[end, end])
end
