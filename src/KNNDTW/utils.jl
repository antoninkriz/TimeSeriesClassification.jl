module _Utils

export euclidean_distance, FastHeap, FastDequeue

function euclidean_distance(a::T, b::T) where {T <: AbstractFloat}
    return (a - b)^2
end

include("_heap.jl")
include("_deque.jl")

end
