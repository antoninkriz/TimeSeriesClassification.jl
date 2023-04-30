module _Utils

export sorted_unique_counts, logspace

function sorted_unique_counts(arr::Vector{T})::Tuple{Vector{T},Vector{Int64}} where {T}
    if isempty(arr)
        return [], []
    end

    uniq_count = 1
    @inbounds for i = 2:length(arr)
        if arr[i] != arr[i-1]
            uniq_count += 1
        end
    end

    unq = similar(arr, uniq_count)
    cnt = zeros(Int64, uniq_count)

    @inbounds unq[1] = arr[1]
    @inbounds cnt[1] = 1

    pos = 1
    @inbounds for x in @views arr[2:end]
        if x === unq[pos]
            cnt[pos] += 1
        else
            pos += 1
            unq[pos] = x
            cnt[pos] = 1
        end
    end

    return unq, cnt
end


@inline logspace(start, stop, n; base=10) = base .^ range(start, stop, n)

end