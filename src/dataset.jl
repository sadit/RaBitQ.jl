# dataset and distances
#

struct RaBitQCosineDistance <: SemiMetric end
struct RaBitQL2Distance <: SemiMetric end

struct RaBitQDatabase <: AbstractDatabase
    data::Matrix{UInt64}
    dot_o_ō::Vector{Float32}
    l2_oraw_c::Vector{Float32}
end

function RaBitQDatabase(db::Matrix{<:Number};
    sample_centroid::Int=0,
    center::Bool=false,
    normalize::Bool=false || center)
    n = size(db, 2)
    Q = BitQuantizer(size(db, 1))
    if !centered
        sample_centroid = sample_centroid < 2 ? ceil(Int, sqrt(n)) : sample_centroid
        p = Float32(1 / sample_centroid)
        c = Q.c
        for _ in 1:sample_centroid
            i = rand(1:n)
            v = view(db, :, i)
            for j in eachindex(v)
                c[j] += Float32(v[j]) * p
            end
        end
    end

    data = Matrix{UInt64}(undef, dim64(Q))
    dot_o_ō = zeros(n)
    l2_oraw_c = zeros(n)
    dist = L2_asf32()
    for (i, x_b, o) in zip(1:n, eachcol(data), eachcol(db))
        l2_oraw_c[i] = evaluate(dist, o, c)
        center && (o .= o .- c)
        normalize && normalize!(o)
        rabitq_bitencode!(Q, x_b, o)
        dot_o_ō[i] = rabitq_dot(Q, x_b, o)
    end

    RaBitQDatabase(data, dot_o_ō, l2_oraw_c)
end

@inline Base.eltype(db::RaBitQDatabase) = typeof(db[1])
@inline Base.setindex!(db::RaBitQDatabase, value, i::Integer) = error("not supported")
@inline push_item!(db::RaBitQDatabase, v) = error("push! is not supported for RaBitQDatabase")
@inline append_items!(a::RaBitQDatabase, b) = error("append! is not supported for RaBitQDatabase")
@inline Base.length(db::RaBitQDatabase) = length(db.dot_o_ō)

struct RaBitQVector{VECU64}
    x_b::VECU64
    dot_o_ō::Float32
    dot_oraw_c::Float32
end

Base.getindex(db::RaBitQDatabase, i::Integer) = RaBitQVector(view(db.data, :, i), db.dot_o_ō[i], db.l2_oraw_c[i])

