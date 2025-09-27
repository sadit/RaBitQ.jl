module RaBitQ
using LinearAlgebra
using SimilaritySearch
export BitQuantizer, rabitq_bitencode!, rabitq_dot, rabitq_estimate_dot, dim64, rabitq_estimate_l2

function dot32(p::AbstractVector{<:AbstractFloat}, o::AbstractVector{<:AbstractFloat})
    d = 0.0f0
    @inbounds @fastmath @simd for i in eachindex(p)
        d += Float32(p[i]) * Float32(o[i])
    end

    d
end

function dot32(p::AbstractVector{Float32}, o::AbstractVector{<:AbstractFloat})
    d = 0.0f0
    @inbounds @fastmath @simd for i in eachindex(p)
        d += p[i] * Float32(o[i])
    end

    d
end

function norm32(o::AbstractVector{<:AbstractFloat})
    d = 0.0f
    @inbounds @fastmath @simd for i in eachindex(o)
        d += Float32(o[i])^2
    end

    sqrt(d)
end

include("rp.jl")
include("quant.jl")
include("dataset.jl")

end
