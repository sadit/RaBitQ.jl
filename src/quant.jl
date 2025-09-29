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

struct BitQuantizer{IQ<:InvertibleRandomProjections}
    m::Float32 # 1/sqrt(dim)
    rp::IQ
    c::Vector{Float32}
end

BitQuantizer(dim::Int) = BitQuantizer(Float32(1 / sqrt(dim)), invertible(QRRandomProjection(Float32, dim, dim)), zeros(Float32, dim))
in_dim(Q::BitQuantizer) = in_dim(Q.rp)
dim64(Q::BitQuantizer) = ceil(Int, in_dim(Q) / 64)

function rabitq_bitencode!(Q::BitQuantizer, x_b::AbstractVector{UInt64}, o::AbstractVector{<:AbstractFloat})
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

function rabitq_bitencode_queries!(Q::BitQuantizer, data::Matrix{UInt64}, c::AbstractVector, db::Matrix{<:Number};
    center::Bool, normalize::Bool)
    n = size(data, 2)
    dist = L2_asf32()

    @batch minbatch = 4 per = thread for i in 1:n
        x_b = view(data, :, i)
        o = view(db, :, i)
        l2_oraw_c[i] = evaluate(dist, o, c)
        center && (o .= o .- c)
        normalize && normalize!(o)
        rabitq_bitencode!(Q, x_b, o)
        dot_o_ō[i] = rabitq_dot(Q, x_b, o)
    end

end

function rabitq_bitencode!(Q::BitQuantizer, data::Matrix{UInt64}, dot_o_ō::Vector{Float32}, l2_oraw_c::Vector{Float32}, c::AbstractVector, db::Matrix{<:Number};
    center::Bool, normalize::Bool)
    n = size(data, 2)
    dist = L2_asf32()

    @batch minbatch = 4 per = thread for i in 1:n
        x_b = view(data, :, i)
        o = view(db, :, i)
        l2_oraw_c[i] = evaluate(dist, o, c)
        center && (o .= o .- c)
        normalize && normalize!(o)
        rabitq_bitencode!(Q, x_b, o)
        dot_o_ō[i] = rabitq_dot(Q, x_b, o)
    end

end

function rabitq_dot_with_bitencoded_vector(x_b::AbstractVector{UInt64}, p::AbstractVector{Float32}, m::Float32)::Float32
    d = 0.0f0

    @inbounds @simd for i in eachindex(p)
        p_i = m * p[i]
        d += ifelse(_b64get(x_b, i), p_i, -p_i)
    end

    d
end

function rabitq_dot(Q::BitQuantizer, x_b::AbstractVector{UInt64}, o::AbstractVector{<:AbstractFloat})::Float32
    dim = in_dim(Q)
    P = Q.rp.map
    m = Q.m
    d = 0.0f0

    # this function do inline projection of x_b in P
    @inbounds for i in 1:dim
        p = view(P, :, i)
        x_i = rabitq_dot_with_bitencoded_vector(x_b, p, m) # projection of x_b in the i-th column
        d += x_i * Float32(o[i])
    end

    d
end

function rabitq_estimate_dot(Q::BitQuantizer, x_b::AbstractVector{UInt64}, dot_oō::Float32, q::AbstractVector{<:AbstractFloat})::Float32
    rabitq_dot(Q, x_b, q) / dot_oō
end

#=
function rabitq_l2_with_encoded(Q::BitQuantizer, x_b::AbstractVector{UInt64}, o::AbstractVector{<:AbstractFloat})
    d = 0.0f0
    m = Q.m

    @inbounds @fastmath @simd for i in eachindex(o)
        o_i = Float32(o[i])
        s = if _b64get(x_b, i)
            o_i - m
        else
            o_i + m
        end

        d += s^2
    end

    sqrt(d)
end=#

function rabitq_estimate_l2(Q::BitQuantizer, x_b::AbstractVector{UInt64}, dot_o_ō::Float32, l2_oraw_c::Float32, qraw::AbstractVector{<:AbstractFloat}, norm::Float32)
    l2_qraw_c = evaluate(L2_asf32(), Q.c, qraw)
    est_o_q = rabitq_estimate_dot(Q, x_b, dot_o_ō, qraw) / norm
    sqrt(l2_oraw_c^2 + l2_qraw_c^2 - 2 * l2_oraw_c * l2_qraw_c * est_o_q)
end

