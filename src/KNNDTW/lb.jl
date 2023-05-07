module _LB

using LoopVectorization: @turbo
using VectorizedStatistics: vsum
using .._Utils: FastHeap

export LBType, lower_bound!, LBNone, LBKeogh

abstract type LBType end

mutable struct LBNone <: LBType end

mutable struct LBKeogh{T} <: LBType where {T <: AbstractFloat}
    radius::Int64
    lower_envelope::Vector{T}
    upper_envelope::Vector{T}
    diff::Vector{T}
end

function LBKeogh{T}(;
    radius::Int64,
    lower_envelope::Vector{T} = T[],
    upper_envelope::Vector{T} = T[],
    diff::Vector{T} = T[],
) where {T <: AbstractFloat}
    @assert radius >= 0 "Parameter radius can not be smaller than 0"
    return LBKeogh(radius, lower_envelope, upper_envelope, diff)
end

function lower_bound!(::LBNone, ::AbstractVector, ::AbstractVector; update_envelope::Bool = true)
    0
end

function lower_bound!(lb::LBKeogh{T}, enveloped::AbstractVector{T}, query::AbstractVector{T}; update_envelope::Bool = true) where {T <: AbstractFloat}
    @assert length(enveloped) === length(query) "Enveloped serires and query series must be of the same length"
    @assert length(enveloped) >= lb.radius + 1 "Window raidus can not be larger than the series itself"

    if length(lb.lower_envelope) < length(enveloped)
        lb.lower_envelope = Vector{T}(undef, length(enveloped))
        update_envelope = true
    end
    
    if length(lb.upper_envelope) < length(enveloped)
        lb.upper_envelope = Vector{T}(undef, length(enveloped))
        update_envelope = true
    end

    if length(lb.diff) < length(enveloped)
        lb.diff = Vector{T}(undef, length(enveloped))
    end

    if update_envelope
        upper_heap = FastHeap{T, Nothing}(2 * lb.radius + 1, :min)
        lower_heap = FastHeap{T, Nothing}(2 * lb.radius + 1, :max)

        # Init first elements
        @inbounds for x in @views enveloped[begin:lb.radius + 1]
            push!(upper_heap, (x, nothing))
            push!(lower_heap, (x, nothing))
        end

        # Move min+max window over the series
        @inbounds for i in 1:length(enveloped) - lb.radius
            push!(upper_heap, (enveloped[i + lb.radius], nothing))
            push!(lower_heap, (enveloped[i + lb.radius], nothing))

            lb.upper_envelope[i] = first(upper_heap)[1]
            lb.lower_envelope[i] = first(lower_heap)[1]
        end

        # Fill rest
        @inbounds for i in (length(enveloped) - lb.radius):length(enveloped)
            lb.upper_envelope[i] = first(upper_heap)[1]
            lb.lower_envelope[i] = first(lower_heap)[1]
        end
    end

    @inbounds @simd for i in 1:length(enveloped)
        lb.diff[i] = max(query[i] - lb.upper_envelope[i], zero(T)) + max(lb.lower_envelope[i] - query[i], zero(T))
    end
    return vsum(@views lb.diff[begin:length(enveloped)])
end

end