# dataset and distances
#

struct RaBitQDatabase <: AbstractDatabase
    data::Matrix{UInt64}
    dot_o_ō::Vector{Float32}
    l2_oraw_c::Vector{Float32}
end

function RaBitQDatabase(db::Matrix{<:Number})

end

struct RaBitQCosineDistance <: SemiMetric

end
