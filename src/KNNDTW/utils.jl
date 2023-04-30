module _Utils

import Base.isempty, Base.empty!, Base.max, Base.push!

export euclidean_distance, FastMaxHeap


function euclidean_distance(a, b)
    return (a - b)^2
end


mutable struct FastMaxHeap{T, Tpayload}
    data::Vector{Tuple{T, Tpayload}}
    n::Int64
end

FastMaxHeap{T, Tpayload}(capacity::Int64) where {T <: Real, Tpayload} = FastMaxHeap(Vector{Tuple{T, Tpayload}}(undef, capacity), 0)

Base.isempty(heap::FastMaxHeap) = heap.n === 0

Base.empty!(heap::FastMaxHeap) = begin
    heap.n = 0
    return heap
end

Base.max(heap::FastMaxHeap) = heap.data[1]

@inline parent(i) =  (i - 1) ÷ 2 + 1
@inline left(i) = (2 * i + 1) + 1
@inline right(i) = (2 * i + 2) + 1

@inbounds function Base.push!(heap::FastMaxHeap{T, Tpayload}, value::Tuple{T, Tpayload}) where {T, Tpayload}
    if heap.n === length(heap.data)
        heap.data[1] = heap.data[heap.n]
        heap.n -= 1

        k = 1
        while true
            l = left(k)
            r = right(k)
            largest = k

            if l < heap.n && heap.data[l][1] > heap.data[i][1]
                largest = l
            end
            
            if r < heap.n && heap.data[r][1] > heap.data[largest][1]
                largest = r
            end

            if largest === k
                break
            end

            heap.data[k], heap.data[largest] = heap.data[largest], heap.data[k]
            k = largest
        end
    end

    heap.n += 1
    i = heap.n
    heap.data[i] = value

    if i === 1
        return
    end

    while heap.data[parent(i)][1] < heap.data[i][1]
        heap.data[i], heap.data[parent(i)] = heap.data[parent(i)], heap.data[i]
        i = parent(i)
    end
end

end
