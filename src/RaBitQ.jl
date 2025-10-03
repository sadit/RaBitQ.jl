module RaBitQ
using LinearAlgebra
using SimilaritySearch
export RaBitQ_Quantizer, RaBitQ_bitencode!, RaBitQ_bitencode_queries!, RaBitQ_dot_ō_o, RaBitQ_dot_ō_qinv, RaBitQ_estimate_dot, dim64, RaBitQ_estimate_l2
export RaBitQ_Database, RaBitQ_Vector, RaBitQ_VectorInfo, RaBitQ_Queries, RaBitQ_QueryVector, RaBitQ_CosineDistance, RaBitQ_L2Distance, RaBitQ_rerank_cosine!

function dot32(p::AbstractVector{<:AbstractFloat}, o::AbstractVector{<:AbstractFloat})
    d = 0.0f0
    @inbounds @simd for i in eachindex(p)
        d += Float32(p[i]) * Float32(o[i])
    end

    d
end

function dot32(p::AbstractVector{Float32}, o::AbstractVector{<:AbstractFloat})
    d = 0.0f0
    @inbounds @simd for i in eachindex(p)
        d += p[i] * Float32(o[i])
    end

    d
end

function norm32(o::AbstractVector{<:AbstractFloat})
    d = 0.0f0
    @inbounds @simd for i in eachindex(o)
        d += Float32(o[i])^2
    end

    sqrt(d)
end

@inline function _b64indices(i_::UInt64)
    i = i_ - one(UInt64)
    (i >>> 6) + 1, (i & 63)
end

@inline function _b64get(v, i_)::Bool
    b, i = _b64indices(UInt64(i_))
    @inbounds (v[b] >>> i) & one(UInt64)
end

@inline function _b64set!(v, i_)
    b, i = _b64indices(UInt64(i_))
    @inbounds v[b] |= (one(UInt64) << i)
    nothing
end

include("rp.jl")

struct RaBitQ_Quantizer{IQ<:InvertibleRandomProjections}
    m::Float32 # 1/sqrt(dim)
    rp::IQ
end

function RaBitQ_Quantizer(dim::Int)
    m = Float32(1 / sqrt(dim))
    Q = invertible(QRRandomProjection(Float32, dim, dim))
    RaBitQ_Quantizer(m, Q)
end

in_dim(Q::RaBitQ_Quantizer) = in_dim(Q.rp)
dim64(Q::RaBitQ_Quantizer) = ceil(Int, in_dim(Q) / 64)

struct RaBitQ_VectorInfo
    dot_ō_o::Float32
    norm::Float32
    err::Float32
end

struct RaBitQ_Database{BQ<:RaBitQ_Quantizer} <: AbstractDatabase
    quant::BQ
    matrix::Matrix{UInt64}
    info::Vector{RaBitQ_VectorInfo}
end

function RaBitQ_Database(db::Matrix{<:Number})
    n = size(db, 2)
    quant = RaBitQ_Quantizer(size(db, 1))

    matrix = Matrix{UInt64}(undef, dim64(quant), n)
    info = Vector{RaBitQ_VectorInfo}(undef, n)

    RaBitQ_bitencode!(quant, matrix, db, info)
    RaBitQ_Database(quant, matrix, info)
end

in_dim(db::RaBitQ_Database) = in_dim(db.quant)

@inline Base.eltype(db::RaBitQ_Database) = typeof(db[1])
@inline Base.setindex!(db::RaBitQ_Database, value, i::Integer) = error("unsupported")
@inline push_item!(db::RaBitQ_Database, v) = error("unsupported push!")
@inline append_items!(a::RaBitQ_Database, b) = error("unsupported append!")
@inline Base.length(db::RaBitQ_Database) = length(db.info)

struct RaBitQ_Queries{MATRIX<:AbstractMatrix{<:Number}} <: AbstractDatabase
    Qinv::MATRIX
    norm::Vector{Float32}
end

struct RaBitQ_QueryVector{VEC}
    qinv::VEC
    norm::Float32
end

Base.getindex(db::RaBitQ_Queries, i::Integer) = RaBitQ_QueryVector(view(db.Qinv, :, i), db.norm[i])
@inline Base.eltype(db::RaBitQ_Queries) = typeof(db[1])
@inline Base.setindex!(db::RaBitQ_Queries, value, i::Integer) = error("unsupported")
@inline push_item!(db::RaBitQ_Queries, v) = error("unsupported push!")
@inline append_items!(a::RaBitQ_Queries, b) = error("unsupported append!")
@inline Base.length(db::RaBitQ_Queries) = size(db.Qinv, 2)

function RaBitQ_Queries(quant::RaBitQ_Quantizer, Q::AbstractMatrix)
    Qinv = invtransform(quant.rp, Q)
    N = [norm32(c) for c in eachcol(Q)]
    RaBitQ_Queries(Qinv, N)
end

struct RaBitQ_Vector{VECU64}
    x_b::VECU64
    info::RaBitQ_VectorInfo
end

Base.getindex(db::RaBitQ_Database, i::Integer) = RaBitQ_Vector(view(db.matrix, :, i), db.info[i])

function RaBitQ_bitencode!(Q::RaBitQ_Quantizer, x_b::AbstractVector{UInt64}, o::AbstractVector{<:AbstractFloat})
    Pinv = Q.rp.inv
    dim = in_dim(Q)
    fill!(x_b, zero(UInt64))

    # this procedure performs an online inverse
    @inbounds for i in 1:dim
        p = view(Pinv, :, i)
        x_i = dot32(p, o)
        if x_i >= 0
            _b64set!(x_b, i)
        end
    end

    x_b
end

function RaBitQ_bitencode_queries!(
    Q::RaBitQ_Quantizer,
    matrix::Matrix{UInt64},
    db::Matrix{<:Number}
)
    n = size(matrix, 2)

    @batch minbatch = 4 per = thread for i in 1:n
        x_b = view(matrix, :, i)
        oraw = view(db, :, i)
        RaBitQ_bitencode!(Q, x_b, oraw)
    end

    matrix
end

function RaBitQ_bitencode!(
    Q::RaBitQ_Quantizer,
    matrix::Matrix{UInt64},
    db::Matrix{<:Number},
    info::Vector{RaBitQ_VectorInfo}
)
    err_factor = Float32(1.9 / sqrt(in_dim(Q)-1))
    n = size(matrix, 2)
    
    @batch minbatch = 4 per = thread for i in 1:n
        x_b = view(matrix, :, i)
        oraw = view(db, :, i)
        N = norm32(oraw)
        RaBitQ_bitencode!(Q, x_b, oraw)
        dot_ō_o = RaBitQ_dot_ō_o(Q, x_b, oraw) / N
        x = dot_ō_o^2
        err = sqrt((1f0 - x) / x)  * err_factor  # sim. RaBitQ_dot_confidence_interval
        info[i] = RaBitQ_VectorInfo(dot_ō_o, N, err)
    end
end

function RaBitQ_dot_confidence_interval(dot_ō_o, D)
    x = dot_ō_o^2
    sqrt((1f0 - x) / x) * 1.9f0 / sqrt(D-1)
end

function RaBitQ_dot_with_bitencoded_vector(x_b::AbstractVector{UInt64}, p::AbstractVector{Float32}, m::Float32)::Float32
    d = 0.0f0

    @inbounds @simd for i in eachindex(p)
        p_i = p[i]
        d += ifelse(_b64get(x_b, i), p_i, -p_i)
    end

    d * m
end

function RaBitQ_dot_ō_o(Q::RaBitQ_Quantizer, x_b::AbstractVector{UInt64}, o::AbstractVector)::Float32
    dim = in_dim(Q)
    P = Q.rp.map
    m = Q.m
    d = 0.0f0

    # this function do inline projection of x_b in P
    @inbounds for i in 1:dim
        p = view(P, :, i)
        x_i = RaBitQ_dot_with_bitencoded_vector(x_b, p, m)  # projection of x_b in the i-th column
        d = muladd(x_i, Float32(o[i]), d)
    end

    d
end

function RaBitQ_dot_ō_qinv(Q::RaBitQ_Quantizer, x_b::AbstractVector{UInt64}, qinv::AbstractVector)::Float32
    dim = in_dim(Q)
    #Pinv = Q.rp.inv
    # m = Q.m
    d = 0.0f0

    # this function do inline inv. projection of q in P^-1
    @inbounds @simd for i in 1:dim
        #x_i = m * qinv[i]
        #d += ifelse(_b64get(x_b, i), x_i, -x_i)
        x_i = qinv[i]
        d += ifelse(_b64get(x_b, i),x_i, -x_i)
    end

    d * Q.m
end

struct RaBitQ_CosineDistance{BQ<:RaBitQ_Quantizer} <: SemiMetric
    Q::BQ
end

struct RaBitQ_L2Distance{BQ<:RaBitQ_Quantizer} <: SemiMetric
    Q::BQ
end

function RaBitQ_estimate_dot(Q::RaBitQ_Quantizer, o::RaBitQ_Vector, q::RaBitQ_QueryVector)::Float32
    RaBitQ_dot_ō_qinv(Q, o.x_b, q.qinv) / o.info.dot_ō_o # / q.norm
end

SimilaritySearch.evaluate(dist::RaBitQ_CosineDistance, o, q) = error("unsupported generic evaluate")
SimilaritySearch.evaluate(dist::RaBitQ_L2Distance, o, q) = error("unsupported generic evaluate")

function SimilaritySearch.evaluate(dist::RaBitQ_CosineDistance, o::RaBitQ_Vector, q::RaBitQ_QueryVector)
    1.0f0 - RaBitQ_estimate_dot(dist.Q, o, q) / q.norm
end

SimilaritySearch.evaluate(dist::RaBitQ_CosineDistance, q::RaBitQ_QueryVector, o::RaBitQ_Vector) = evaluate(dist, o, q)

function SimilaritySearch.evaluate(dist::RaBitQ_L2Distance, o::RaBitQ_Vector, q::RaBitQ_QueryVector)
    dot_o_q = RaBitQ_estimate_dot(dist.Q, o, q) / q.norm
    sqrt(o.info.norm^2 + q.norm^2 - 2 * o.info.norm * q.norm * dot_o_q)
end

SimilaritySearch.evaluate(dist::RaBitQ_L2Distance, a::AbstractVector, b::RaBitQ_Vector) = evaluate(dist, b, a)s

end
