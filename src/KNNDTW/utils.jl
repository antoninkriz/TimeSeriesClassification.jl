module _Utils

export euclidean_distance, FastHeap, FastDequeue

function euclidean_distance(a::T, b::T) where {T}
    return (a - b)^T(2)    
end

include("_heap.jl")
include("_deque.jl")

end
