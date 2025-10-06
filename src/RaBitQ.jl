module RaBitQ
using LinearAlgebra
using SimilaritySearch
export RaBitQ_estimate_dot, RaBitQ_estimate_l2
export RaBitQ_Database, RaBitQ_Vector, RaBitQ_VectorInfo, RaBitQ_Queries, RaBitQ_QueryVector, RaBitQ_CosineDistance, RaBitQ_L2Distance, RaBitQ_rerank_cosine!
export RaBitQ_bitencode!, RaBitQ_bitencode, dim64

using AllocCheck
function dot32(p::AbstractVector{<:Number}, o::AbstractVector{<:Number})::Float32
    d = 0.0f0
    @inbounds @simd for i in eachindex(p)
        d += Float32(p[i]) * Float32(o[i])
    end

    d
end

function dot32(p::AbstractVector{Float32}, o::AbstractVector{<:Number})::Float32
    d = 0.0f0
    @inbounds @simd for i in eachindex(p)
        d += p[i] * Float32(o[i])
    end

    d
end

function norm32(o::AbstractVector{<:Number})::Float32
    d = 0.0f0
    @inbounds @simd for i in eachindex(o)
        d += Float32(o[i])^2
    end

    sqrt(d)
end

struct b64indices
    b::Int
    i::Int

    function b64indices(i::Int)
        i -= 1
        new((i >>> 6) + 1, (i & 63))
    end
end

#Base.getproperty(x::RaBitQ.b64indices, f::Symbol) = error(f)


function _b64get(v::AbstractVector{UInt64}, i_::Int)::Bool
    p = b64indices(i_)
    @inbounds b = v[p.b]
    val = one(UInt64) << p.i
    (v[p.b] & val) !== zero(UInt64)
end

function _b64set!(v::AbstractVector{UInt64}, i_::Int)
    p = b64indices(i_)
    val = one(UInt64) << p.i
    val |= v[p.b]
    v[p.b] = val
    nothing
end

include("rp.jl")

struct RaBitQ_VectorInfo
    dot_ō_o::Float32
    norm::Float32
    err::Float32
end

struct RaBitQ_Database <: AbstractDatabase
    P::Matrix{Float32}
    Pinv::Matrix{Float32}
    m::Float32
    sketches::Matrix{UInt64}
    info::Vector{RaBitQ_VectorInfo}
end

function RaBitQ_Database(X::Matrix{<:Number})
    dim, n = size(X)
    sketches = Matrix{UInt64}(undef, dim64(dim), n)
    info = Vector{RaBitQ_VectorInfo}(undef, n)
    Q = invertible(QRRandomProjection(Float32, dim))
    m = Float32(1 / sqrt(dim))

    RaBitQ_bitencode_matrix!(Q.map, Q.inv, m, sketches, X, info)
    RaBitQ_Database(Q.map, Q.inv, m, sketches, info)
end

in_dim(db::RaBitQ_Database) = in_dim(db.quant)
dim64(dim::Integer) = ceil(Int, dim / 64)

@inline Base.eltype(db::RaBitQ_Database) = typeof(db[1])
@inline Base.setindex!(db::RaBitQ_Database, i::Integer) = error("unsupported")
@inline push_item!(db::RaBitQ_Database, v) = error("unsupported push!")
@inline append_items!(a::RaBitQ_Database, b) = error("unsupported append!")
@inline Base.length(db::RaBitQ_Database) = length(db.info)
Base.getindex(db::RaBitQ_Database, i::Integer) = RaBitQ_Vector(view(db.sketches, :, i), db.info[i])

struct RaBitQ_Queries <: AbstractDatabase
    Qinv::Matrix{Float32}
    norm::Vector{Float32}
end

struct RaBitQ_QueryVector{VEC}
    qinv::VEC
    norm::Float32
end

Base.getindex(db::RaBitQ_Queries, i::Int) = RaBitQ_QueryVector(view(db.Qinv, :, i), db.norm[i])
@inline Base.eltype(db::RaBitQ_Queries) = typeof(db[1])
@inline Base.setindex!(db::RaBitQ_Queries, value, i::Int) = error("unsupported")
@inline push_item!(db::RaBitQ_Queries, v) = error("unsupported push!")
@inline append_items!(a::RaBitQ_Queries, b) = error("unsupported append!")
@inline Base.length(db::RaBitQ_Queries) = size(db.Qinv, 2)

function RaBitQ_Queries(db::RaBitQ_Database, Q::AbstractMatrix)
    rp = InvertibleRandomProjections(db.P, db.Pinv)
    #out = Matrix{Float16}(undef, out_dim(rp), size(Q, 2))  # 1.5 times slower and queries mem is not a big problem
    #Qinv = invtransform!(rp, out, Q)
    Qinv = invtransform(rp, Q)
    N = [norm32(c) for c in eachcol(Q)]
    RaBitQ_Queries(Qinv, N)
end

struct RaBitQ_Vector{VECU64}
    x_b::VECU64
    info::RaBitQ_VectorInfo
end

function RaBitQ_bitencode_!(P::AbstractMatrix{Float32}, x_b::AbstractVector{UInt64}, o::AbstractVector{<:Number})
    dim = size(P, 2) # in_dim(Q)
    fill!(x_b, zero(UInt64))

    # this procedure performs an online inverse
    r = 1:dim
    for i in r
        p = view(P, :, i)
        x_i = dot32(p, o)
        x_i >= 0 && _b64set!(x_b, i)
    end

    x_b
end


"""
RaBitQ_bitencode_matrix!(
    P::Matrix, Pinv::Matrix, m::Float32,
    sketches::Matrix{UInt64},
    X::Matrix{<:Number},
    info::Vector{RaBitQ_VectorInfo})

Internal function that transform dataset and computes all related information for RaBitQ_Database
"""
function RaBitQ_bitencode_matrix!(
    P::AbstractMatrix, Pinv::AbstractMatrix, m::Float32,
    sketches::AbstractMatrix{UInt64},
    X::AbstractMatrix{<:Number},
    info::Vector{RaBitQ_VectorInfo})

    err_factor = Float32(1.9 / sqrt(size(P, 1) - 1))
    n = size(sketches, 2)

    @batch minbatch = 4 per = thread for i in 1:n
        x_b = view(sketches, :, i)
        oraw = view(X, :, i)
        N = norm32(oraw)
        RaBitQ_bitencode_!(Pinv, x_b, oraw)
        dot_ō_o = RaBitQ_dot_ō_o(P, m, x_b, oraw) / N
        x = dot_ō_o^2
        err = sqrt((1.0f0 - x) / x) * err_factor  # sim. RaBitQ_dot_confidence_interval
        info[i] = RaBitQ_VectorInfo(dot_ō_o, N, err)
    end
end

"""
    RaBitQ_bitencode!(
        P::AbstractMatrix{Float32},
        sketches::AbstractMatrix{UInt64},
        db::AbstractMatrix{<:Number}
    )
    
Computes a binary sketches projecting `db` to a Binary Hamming space storing binary sketches into `sketches` 
"""
function RaBitQ_bitencode!(
        P::AbstractMatrix{Float32},
        sketches::AbstractMatrix{UInt64},
        db::AbstractMatrix{<:Number}
    )
    n = size(sketches, 2)

    @batch minbatch = 4 per = thread for i in 1:n
        x_b = view(sketches, :, i)
        oraw = view(db, :, i)
        RaBitQ_bitencode_!(P, x_b, oraw)
    end

    sketches
end

"""
    RaBitQ_bitencode(
        P::AbstractMatrix{Float32},
        db::AbstractMatrix{<:Number}
    ) -> Matrix{UInt64}

Computes binary sketches for `db` to be able to use Hamming space. Returns a UInt64 matrix with the binary sketches.
    
"""
function RaBitQ_bitencode(
        P::AbstractMatrix{Float32},
        db::AbstractMatrix{<:Number}
    )
    n = size(db, 2)
    m = ceil(Int, size(P, 2) / 64)
    sketches = Matrix{UInt64}(undef, m, n)

    RaBitQ_bitencode!(P, sketches, db)
end

function RaBitQ_dot_confidence_interval(dot_ō_o, D)
    x = dot_ō_o^2
    sqrt((1.0f0 - x) / x) * 1.9f0 / sqrt(D - 1)
end

function RaBitQ_dot_with_bitencoded_vector(x_b::AbstractVector{UInt64}, p::AbstractVector{Float32}, m::Float32)::Float32
    d = 0.0f0

    @inbounds @simd for i in eachindex(p)
        p_i = p[i]
        d += ifelse(_b64get(x_b, i), p_i, -p_i)
    end

    d * m
end

"""
    RaBitQ_dot_ō_o(P::AbstractMatrix{Float32}, m::Float32, x_b::AbstractVector{UInt64}, o::AbstractVector)::Float32

Computes `dot(P * x̄, o)` note that `P * x̄` is computed online with P, m, and x_b
"""
function RaBitQ_dot_ō_o(P::AbstractMatrix{Float32}, m::Float32, x_b::AbstractVector{UInt64}, o::AbstractVector)::Float32
    dim = size(P, 1)  # original dimension
    d = 0.0f0

    # this function do inline projection of x_b in P
    @inbounds for i in 1:dim
        p = view(P, :, i)
        x_i = RaBitQ_dot_with_bitencoded_vector(x_b, p, m)  # projection of x_b in the i-th column
        d = muladd(x_i, Float32(o[i]), d)
    end

    d
end

function RaBitQ_dot_ō_qinv(m::Float32, x_b::AbstractVector{UInt64}, qinv::AbstractVector{Float32})::Float32
    d = 0.0f0

    @inbounds @simd for i in eachindex(qinv)
        x_i = qinv[i]
        d += ifelse(_b64get(x_b, i), x_i, -x_i)
    end

    d * m
end

#= ALL OF THE FOLLOWING ARE SLOWER OR EQ THAN RaBitQ_dot_ō_qinv_v1 ==
=====================================================================

function RaBitQ_dot_ō_qinv_v2(m::Float32, x_b::AbstractVector{UInt64}, qinv::AbstractVector{Float32})::Float32
    d = 0.0f0

    n = length(x_b)
    j = 1
    @inbounds for i in 1:n-1
        B = x_b[i]

        @inbounds @simd for k in 0:63
            b = B & (one(UInt64) << k)
            x = qinv[j + k]
            d += ifelse(zero(UInt64) === b, -x, x)
        end

        j += 64
    end

    B = x_b[n]

    @simd for k in 0:length(qinv)-j-1
        b = B & (one(UInt64) << k)
        x = qinv[j + k]
        d += ifelse(zero(UInt64) === b, -x, x)
    end

    d * m
end

const QINVMASK = Matrix{Float32}(undef, 8, 2^8)

for i in 1:2^8
    ib = UInt64(i - 1)
    for j in 1:8
        jb = one(UInt64) << (j-1) & ib
        if jb === zero(UInt64)
            QINVMASK[j, i] = Float16(-1f0)
        else
            QINVMASK[j, i] = Float16(1f0)
        end
    end
end


function RaBitQ_dot_ō_qinv_v3(m::Float32, x_b::AbstractVector{UInt64}, qinv::AbstractVector{Float32})::Float32
    d = 0.0f0
    n = length(x_b)
    j = 1
    for i in 1:n-1
        B = x_b[i]
        for _ in 1:8
            b = Int(B & 0xff) + 1
            v = view(QINVMASK, :, b)
            d += dot32(v, view(qinv, j:j+7))
            j += 8
            B >>>= 8
        end
    end

    @inbounds B = x_b[n]

    @inbounds @simd for k in 0:length(qinv)-j-1
        x = qinv[j + k]
        b = B & (one(UInt64) << k)
        d += ifelse(zero(UInt64) === b, -x, x)
    end

    d * m
end

function RaBitQ_dot_ō_qinv_v4(m::Float32, x_b::AbstractVector{UInt64}, qinv::AbstractVector{Float32})::Float32
    @inbounds @simd for i in eachindex(qinv)
        qinv[i] = ifelse(_b64get(x_b, i), qinv[i], -qinv[i])
    end

    d = sum(qinv) * m

    @inbounds @simd for i in eachindex(qinv)
        qinv[i] = ifelse(_b64get(x_b, i), qinv[i], -qinv[i])
    end

    d
end =#

#const RaBitQ_dot_ō_qinv = RaBitQ_dot_ō_qinv_v1

"""
    struct RaBitQ_CosineDistance <: SemiMetric
        m::Float32
    end

A cosine distance estimator, intended to be used with `evaluate`

"""
struct RaBitQ_CosineDistance <: SemiMetric
    m::Float32
end

"""
    struct RaBitQ_L2Distance <: SemiMetric
        m::Float32
    end

An Euclidean distance estimator, intended to be used with `evaluate`

"""
struct RaBitQ_L2Distance <: SemiMetric
    m::Float32
end

"""
    RaBitQ_estimate_dot(m::Float32, o::RaBitQ_Vector, q::RaBitQ_QueryVector)::Float32

Estimates the dot product with the given `m` value (`m=1/sqrt(dim)`)
"""
function RaBitQ_estimate_dot(m::Float32, o::RaBitQ_Vector, q::RaBitQ_QueryVector)::Float32
    RaBitQ_dot_ō_qinv(m, o.x_b, q.qinv) / o.info.dot_ō_o # / q.norm
end

SimilaritySearch.evaluate(dist::RaBitQ_CosineDistance, o, q) = error("unsupported generic evaluate")
SimilaritySearch.evaluate(dist::RaBitQ_L2Distance, o, q) = error("unsupported generic evaluate")

function SimilaritySearch.evaluate(dist::RaBitQ_CosineDistance, o::RaBitQ_Vector, q::RaBitQ_QueryVector)::Float32
    1.0f0 - RaBitQ_estimate_dot(dist.m, o, q) / q.norm
end

SimilaritySearch.evaluate(dist::RaBitQ_CosineDistance, q::RaBitQ_QueryVector, o::RaBitQ_Vector) = evaluate(dist, o, q)

function SimilaritySearch.evaluate(dist::RaBitQ_L2Distance, o::RaBitQ_Vector, q::RaBitQ_QueryVector)::Float32
    dot_o_q = RaBitQ_estimate_dot(dist.m, o, q) / q.norm
    sqrt(o.info.norm^2 + q.norm^2 - 2 * o.info.norm * q.norm * dot_o_q)
end

SimilaritySearch.evaluate(dist::RaBitQ_L2Distance, a::AbstractVector, b::RaBitQ_Vector) = evaluate(dist, b, a)s

end
