module _Utils

import Base.isempty, Base.empty!, Base.first, Base.length, Base.push!

export euclidean_distance, FastHeap

function euclidean_distance(a, b)
    return (a - b)^2
end

mutable struct FastHeap{T, Tpayload}
    data::Vector{Tuple{T, Tpayload}}
    n::Int64
    is_min::Bool
end

function FastHeap{T, Tpayload}(capacity::Int64, type::Symbol) where {T, Tpayload}
    @assert capacity > 0 "Heap capacity must be greater than zero"
    @assert type in (:min, :max) "Heap can be only maxheap or minheap"

    FastHeap{T, Tpayload}(Vector{Tuple{T, Tpayload}}(undef, capacity), 0, type == :min)
end

Base.isempty(heap::FastHeap{T, Tpayload}) where {T, Tpayload} = heap.n == 0

Base.empty!(heap::FastHeap{T, Tpayload}) where {T, Tpayload} = begin
    heap.n = 0
    return heap
end

Base.first(heap::FastHeap{T, Tpayload}) where {T, Tpayload} = begin
    @assert heap.n > 0 "Heap is empty"
    return heap.data[1]
end

Base.length(heap::FastHeap{T, Tpayload}) where {T, Tpayload} = heap.n

@inline parent(i) = (i - 1) ÷ 2 + 1
@inline left(i) = (2 * i + 1) + 1
@inline right(i) = (2 * i + 2) + 1

function Base.push!(heap::FastHeap{T, Tpayload}, value::Tuple{T, Tpayload}) where {T, Tpayload}
    @inbounds if heap.n == length(heap.data)
        heap.data[1] = heap.data[heap.n]
        heap.n -= 1

        k = 1
        while true
            l = left(k)
            r = right(k)
            largest = k

            if l < heap.n && ((heap.is_min && heap.data[l][1] < heap.data[k][1]) || (!heap.is_min && heap.data[l][1] > heap.data[k][1]))
                largest = l
            end

            if r < heap.n && ((heap.is_min && heap.data[r][1] < heap.data[largest][1]) || (!heap.is_min && heap.data[r][1] > heap.data[largest][1]))
                largest = r
            end

            if largest == k
                break
            end

            heap.data[k], heap.data[largest] = heap.data[largest], heap.data[k]
            k = largest
        end
    end

    heap.n += 1
    i = heap.n
    heap.data[i] = value

    if i == 1
        return
    end

    while (heap.is_min && heap.data[i][1] < heap.data[parent(i)][1]) || (!heap.is_min && heap.data[i][1] > heap.data[parent(i)][1])
        heap.data[i], heap.data[parent(i)] = heap.data[parent(i)], heap.data[i]
        i = parent(i)
    end
end

end
