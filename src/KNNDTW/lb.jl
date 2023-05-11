module _LB

using Base: RefValue
using LoopVectorization: @turbo
using VectorizedStatistics: vsum
using .._Utils: FastDequeue

export LBType, lower_bound!, LBNone, LBKeogh

abstract type LBType end

mutable struct LBNone <: LBType end

"Structure implementing the LB_Keogh lower bounding method"
mutable struct LBKeogh{T} <: LBType where {T <: AbstractFloat}
    radius::Int64
    lower_envelope::Vector{T}
    upper_envelope::Vector{T}
    diff::Vector{T}
end

"Constructor for the structure implementing the LB_Keogh lower bound method."
function LBKeogh{T}(;
    radius::Int64,
    lower_envelope::Vector{T} = T[],
    upper_envelope::Vector{T} = T[],
    diff::Vector{T} = T[],
) where {T <: AbstractFloat}
    @assert radius >= 0 "Parameter radius can not be smaller than 0"
    return LBKeogh(radius, lower_envelope, upper_envelope, diff)
end

"Lower bound method that always returns zero."
function lower_bound!(::LBNone, ::Tarr, ::Tarr; update::Bool = true)::T where {T <: AbstractFloat, Tarr <: AbstractVector{T}}
    zero(T)
end

"
Function implementing the lower bound LB_Keogh method.

Set update=true to update the envelope forcefully.
"
function lower_bound!(lb::LBKeogh{T}, enveloped::Tarr, query::Tarr; update::Bool = true)::T where {T <: AbstractFloat, Tarr <: AbstractVector{T}}
    @assert length(enveloped) === length(query) "Enveloped serires and query series must be of the same length"
    @assert length(enveloped) >= lb.radius + 1 "Window raidus can not be larger than the series itself"

    if length(lb.lower_envelope) < length(enveloped)
        lb.lower_envelope = Vector{T}(undef, length(enveloped))
        update = true
    end
    
    if length(lb.upper_envelope) < length(enveloped)
        lb.upper_envelope = Vector{T}(undef, length(enveloped))
        update = true
    end

    if length(lb.diff) < length(enveloped)
        lb.diff = Vector{T}(undef, length(enveloped))
    end

    @inbounds if update
        upper_deque = FastDequeue{Tuple{T, Int64}}(2 * lb.radius + 1)
        lower_deque = FastDequeue{Tuple{T, Int64}}(2 * lb.radius + 1)

        # Init first elements
        for i in 1:lb.radius + 1
            push!(upper_deque, (enveloped[i], i))
            push!(lower_deque, (enveloped[i], i))
        end

        # Move min+max window over the series
        for i in 1:length(enveloped) - lb.radius
            if !isempty(upper_deque) && first(upper_deque)[2] <= i - lb.radius
                popfirst!(upper_deque)
            end

            if !isempty(lower_deque) && first(lower_deque)[2] <= i - lb.radius
                popfirst!(lower_deque)
            end

            while !isempty(lower_deque) && last(upper_deque)[1] > enveloped[i + lb.radius]
                pop!(lower_deque)
            end

            while !isempty(upper_deque) && last(upper_deque)[1] < enveloped[i + lb.radius]
                pop!(upper_deque)
            end

            push!(upper_deque, (enveloped[i + lb.radius], i))
            push!(lower_deque, (enveloped[i + lb.radius], i))

            lb.upper_envelope[i] = first(upper_deque)[1]
            lb.lower_envelope[i] = first(lower_deque)[1]
        end

        # Fill rest
        for i in (length(enveloped) - lb.radius):length(enveloped)
            lb.upper_envelope[i] = first(upper_deque)[1]
            lb.lower_envelope[i] = first(lower_deque)[1]
        end
    end

    @inbounds @simd for i in 1:length(enveloped)
        lb.diff[i] = max(query[i] - lb.upper_envelope[i], zero(T)) + max(lb.lower_envelope[i] - query[i], zero(T))
    end
    return vsum(@views lb.diff[begin:length(enveloped)])
end

end