using Polyester, Random, LinearAlgebra, Distributions
export AbstractRandomProjections, InvertibleRandomProjections, RandomProjections, invertible, invertible
export GaussianRandomProjection, QRRandomProjection, Achioptas3RandomProjection, Achioptas2RandomProjection
export out_dim, in_dim, transform, transform!, invtransform, invtransform!

abstract type AbstractRandomProjections end


struct RandomProjections{M<:AbstractMatrix} <: AbstractRandomProjections
    map::M
end

struct InvertibleRandomProjections{M<:AbstractMatrix} <: AbstractRandomProjections
    map::M
    inv::M
end

getmap(rp::RandomProjections) = rp.map
getmap(rp::InvertibleRandomProjections) = rp.map

function invertible(rp::RandomProjections)
    if in_dim(rp) == out_dim(rp)
        InvertibleRandomProjections(rp.map, permutedims(inv(rp.map)))
    else
        InvertibleRandomProjections(rp.map, perutedims(pinv(rp.map)))
    end
end

function GaussianRandomProjection(FloatType, in_dim::Int, out_dim::Int=in_dim)
    #(in_dim, out_dim) = map_dims
    N = Normal(zero(FloatType), FloatType(1 / out_dim))
    M = rand(N, in_dim, out_dim)

    for c in eachcol(M)
        normalize!(c)
    end

    RandomProjections(M)
end

function QRRandomProjection(FloatType, in_dim::Int, out_dim::Int=in_dim; seed::Int=0)
    rng = Xoshiro(seed)
    M, _ = qr(rand(rng, FloatType, (in_dim, in_dim)))
    M = Matrix(M)

    if in_dim != out_dim
        RandomProjections(M[:, 1:out_dim])
    else
        RandomProjections(M)
    end
end

function Achioptas2RandomProjection(FloatType, in_dim::Int, out_dim::Int=in_dim; seed::Int=0)
    rng = Xoshiro(seed)
    M = Matrix{FloatType}(undef, in_dim, in_dim)

    v = FloatType(sqrt(1 / out_dim))

    for i in CartesianIndices(M)
        if rand(rng, Float32) < 0.5f0 # 1 / 2
            M[i] = v
        else
            M[i] = -v
        end
    end

    RandomProjections(M)
end

function Achioptas3RandomProjection(FloatType, in_dim::Int, out_dim::Int=in_dim; seed::Int=0)
    rng = Xoshiro(seed)
    M = Matrix{FloatType}(undef, in_dim, in_dim)

    v = FloatType(sqrt(3 / out_dim))

    for i in CartesianIndices(M)
        if rand(rng) < 0.16666667f0 # 1 / 6
            M[i] = v
        elseif rand(rng) < 0.16666667f0
            M[i] = -v
        else
            M[i] = zero(FloatType)
        end
    end

    RandomProjections(M)
end

Base.size(rp::AbstractRandomProjections) = size(getmap(rp))
in_dim(rp::AbstractRandomProjections) = size(getmap(rp), 1)
out_dim(rp::AbstractRandomProjections) = size(getmap(rp), 2)
Base.eltype(rp::AbstractRandomProjections) = eltype(getmap(rp))

function transform!(rp::AbstractRandomProjections, out::AbstractVector, v::AbstractVector)
    for (i, x) in enumerate(eachcol(getmap(rp)))
        @inbounds out[i] = dot(x, v)
    end

    out
end

function transform(rp::AbstractRandomProjections, v::AbstractVector)
    out = Vector{eltype(rp)}(undef, out_dim(rp))
    transform!(rp, out, v)
end

function transform(rp::AbstractRandomProjections, X::AbstractMatrix)
    O = Matrix{eltype(rp)}(undef, out_dim(rp), length(X))
    transform!(rp, O, X)
end

function transform!(rp::AbstractRandomProjections, O::AbstractMatrix, X::AbstractMatrix; minbatch::Int=4)
    n = size(X, 2)

    @batch per = thread minbatch = minbatch for i in 1:n
        o = view(O, :, i)
        x = view(X, :, i)
        transform!(rp, o, x)
    end

    O
end

function invtransform!(rp::InvertibleRandomProjections, out::AbstractVector, v::AbstractVector)
    #@assert out_dir(rp) == length(out) && in_dir(rp) == length(v)
    mul!(out, rp.inv, v)
end

function invtransform(rp::InvertibleRandomProjections, v::AbstractVector)
    out = Vector{eltype(rp)}(undef, out_dim(rp))
    invtransform!(rp, out, v)
end

function invtransform!(rp::InvertibleRandomProjections, O::AbstractMatrix, X::AbstractMatrix; minbatch::Int=4)
    #@assert out_dir(rp) == length(out) && in_dir(rp) == length(v)
    n = size(X, 2)

    @batch per = thread minbatch = minbatch for i in 1:n
        o = view(O, :, i)
        x = view(X, :, i)
        invtransform!(rp, o, x)
    end
end

function invtransform(rp::InvertibleRandomProjections, X::AbstractMatrix)
    O = Matrix{eltype(rp)}(undef, out_dim(rp), size(X, 2))
    invtransform!(rp, O, X)
end

