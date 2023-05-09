import Base.isempty, Base.empty!, Base.first, Base.length, Base.push!, Base.pushfirst!, Base.pop!

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
    return
end

Base.first(heap::FastHeap{T, Tpayload}) where {T, Tpayload} = begin
    @assert heap.n > 0 "Heap is empty"
    return heap.data[1]
end

Base.length(heap::FastHeap{T, Tpayload}) where {T, Tpayload} = heap.n

@inline parent(i) = i ÷ 2
@inline left(i) = 2 * i
@inline right(i) = 2 * i + 1

function Base.push!(heap::FastHeap{T, Tpayload}, value::Tuple{T, Tpayload}) where {T, Tpayload}
    if heap.n == length(heap.data)
        return
    end

    heap.n += 1
    i = heap.n
    heap.data[i] = value

    @inbounds while i > 1 && ((heap.is_min && heap.data[i][1] < heap.data[parent(i)][1]) || (!heap.is_min && heap.data[i][1] > heap.data[parent(i)][1]))
        heap.data[i], heap.data[parent(i)] = heap.data[parent(i)], heap.data[i]
        i = parent(i)
    end
end

function Base.pushfirst!(heap::FastHeap{T, Tpayload}, value::Tuple{T, Tpayload}) where {T, Tpayload}
    heap.data[1] = value

    i = 1
    while true
        l = left(i)
        r = right(i)

        if l > length(heap)
            break
        end

        c = if r > length(heap)
            l
        elseif heap.is_min 
            heap.data[l][1] < heap.data[l][1] ? l : r
        else
            heap.data[l][1] > heap.data[l][1] ? l : r
        end

        if (heap.is_min && heap.data[i][1] > heap.data[c][1]) || (!heap.is_min && heap.data[i][1] < heap.data[c][1])
            heap.data[i], heap.data[c] = heap.data[c], heap.data[i]
            i = c
        else
            break
        end
    end
end

function Base.pop!(heap::FastHeap{T, Tpayload}) where {T, Tpayload}
    if length(heap) == 0
        return
    elseif length(heap) == 1
        heap.n = 0
        return
    end
    
    heap.data[1] = heap.data[length(heap)]
    heap.n -= 1

    i = 1
    while true
        l = left(i)
        r = right(i)

        if l > length(heap)
            break
        end

        c = if r > length(heap)
            l
        elseif heap.is_min 
            heap.data[l][1] < heap.data[l][1] ? l : r
        else
            heap.data[l][1] > heap.data[l][1] ? l : r
        end

        if (heap.is_min && heap.data[i][1] > heap.data[c][1]) || (!heap.is_min && heap.data[i][1] < heap.data[c][1])
            heap.data[i], heap.data[c] = heap.data[c], heap.data[i]
            i = c
        else
            break
        end
    end
end