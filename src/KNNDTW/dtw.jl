function dist(a::T, b::T) where {T <: AbstractFloat}
    return (a - b)^2
end

function dtw_with_sakoe_chiba_radius(x::AbstractVector{T}, y::AbstractVector{T}; sakoe_chiba_radius::Unsigned) where {T <: AbstractFloat}
    # TODO
end

function dtw_with_itakura_max_slope(x::AbstractVector{T}, y::AbstractVector{T}; itakura_max_slope::T2) where {T <: AbstractFloat, T2 <: Real}
    # TODO
end

function dtw(x::AbstractVector{T}, y::AbstractVector{T}) where {T <: AbstractFloat}
    row_count, col_count = length(x), length(y)

    # Julia is column major, to make things faster let longer timeseries = columns and shorter timeseries = rows
    if row_count < col_count
        row_count, col_count = col_count, row_count
        x, y = y, x
    end

    M = zeros(T, row_count, col_count)
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
