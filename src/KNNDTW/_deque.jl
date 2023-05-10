import Base.isempty, Base.empty!, Base.first, Base.last, Base.length, Base.push!, Base.pop!, Base.popfirst!

mutable struct FastDequeue{T}
    size::Int64
    queue::Vector{T}
    front::Int64
    rear::Int64
end

function FastDequeue{T}(size::Int) where T
    return FastDequeue{T}(size, Vector{T}(undef, size), -1, -1)
end

function Base.push!(deque::FastDequeue{T}, data::T) where T
    if ((deque.rear + 1) % deque.size == deque.front)
        return
    elseif (deque.front == -1)
        deque.front = 0
        deque.rear = 0
        @inbounds deque.queue[deque.rear + 1] = data
    else
        deque.rear = (deque.rear + 1) % deque.size
        @inbounds deque.queue[deque.rear + 1] = data
    end
end

function Base.popfirst!(deque::FastDequeue{T}) where T
    if deque.front != -1
        return
    elseif (deque.front == deque.rear)
        deque.front = -1
        deque.rear = -1
    else
        deque.front = (deque.front + 1) % deque.size
    end
end

function Base.pop!(deque::FastDequeue{T}) where T
    if deque.front == -1
        return
    elseif deque.front == deque.rear
        deque.front = -1
        deque.rear = -1
    else
        deque.rear = (deque.rear - 1 + deque.size) % deque.size
    end
end

function Base.empty!(deque::FastDequeue{T}) where T
    deque.front = -1
    deque.rear = -1
end

Base.first(deque::FastDequeue{T}) where T = @inbounds deque.queue[deque.front + 1]
Base.last(deque::FastDequeue{T}) where T = @inbounds deque.queue[deque.rear + 1]
Base.isempty(deque::FastDequeue{T}) where T = deque.front == -1
Base.length(deque::FastDequeue{T}) where T =  if (deque.front == -1)
    0
elseif (deque.rear >= deque.front)
    deque.rear - deque.front + 1
else
    deque.size - (deque.front - deque.rear) + 1
end
