using Polyester, Random, LinearAlgebra, Distributions, SimilaritySearch, StatsBase
export AbstractRandomProjections, InvertibleRandomProjections, RandomProjections, invertible, refine
export GaussianRandomProjection, QRRandomProjection, QRXRandomProjection, Achlioptas3RandomProjection, Achlioptas2RandomProjection
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
        InvertibleRandomProjections(rp.map, inv(rp.map))
    else
        InvertibleRandomProjections(rp.map, pinv(rp.map))
    end
end

struct OrthoDistance <: SemiMetric end
SimilaritySearch.evaluate(::OrthoDistance, u, v) = 1f0 - abs(dot32(u, v))

function refine(rp::RandomProjections, out_dim::Int; start::Int=1, verbose::Bool=true)
    C = fft(OrthoDistance(), MatrixDatabase(rp.map), out_dim; start, verbose)
    @show C.dmax, quantile(C.dists, 0.0:0.25:1.0)
    RandomProjections(rp.map[:, C.centers])
end

function Base.hcat(rp1::RandomProjections, rp2::RandomProjections)
    RandomProjections(hcat(rp1.map, rp2.map))
end

function GaussianRandomProjection(rng::AbstractRNG, FloatType::Type, in_dim::Int, out_dim::Int)
    N = Normal(zero(FloatType), convert(FloatType, 1 / out_dim))
    M = rand(rng, N, in_dim, out_dim)
    for c in eachcol(M)
        normalize!(c)
    end

    RandomProjections(M)
end

function QRRandomProjection(rng::AbstractRNG, FloatType::Type, in_dim::Int, out_dim::Int)
    M, _ = qr(rand(rng, FloatType, (in_dim, in_dim)))
    M = Matrix(M)

    if in_dim != out_dim
        RandomProjections(M[:, 1:out_dim])
    else
        RandomProjections(M)
    end
end

function QRXRandomProjection(rng::AbstractRNG, FloatType::Type, in_dim::Int, out_dim::Int; factor::Int=3)
    rp0 = QRRandomProjection(rng, FloatType, in_dim, in_dim)
    rp = GaussianRandomProjection(rng, FloatType, in_dim, factor * out_dim)
    rp = hcat(rp0, rp)
    refine(rp, out_dim)
end

function Achlioptas2RandomProjection(rng::AbstractRNG, FloatType::Type, in_dim::Int, out_dim::Int)
    M = Matrix{FloatType}(undef, in_dim, out_dim)
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

function Achlioptas3RandomProjection(rng::AbstractRNG, FloatType::Type, in_dim::Int, out_dim::Int)
    M = Matrix{FloatType}(undef, in_dim, out_dim)
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

GaussianRandomProjection(in_dim::Int, out_dim::Int=in_dim) = GaussianRandomProjection(Xoshiro(0), Float32, in_dim, out_dim)
QRRandomProjection(in_dim::Int, out_dim::Int=in_dim) = QRRandomProjection(Xoshiro(0), Float32, in_dim, out_dim)
QRXRandomProjection(in_dim::Int, out_dim::Int; factor::Int=3) = QRXRandomProjection(Xoshiro(0), Float32, in_dim, out_dim; factor)
Achlioptas2RandomProjection(in_dim::Int, out_dim::Int=in_dim) = Achlioptas2RandomProjection(Xoshiro(0), Float32, in_dim, out_dim)
Achlioptas3RandomProjection(in_dim::Int, out_dim::Int=in_dim) = Achlioptas3RandomProjection(Xoshiro(0), Float32, in_dim, out_dim)

Base.size(rp::AbstractRandomProjections) = size(getmap(rp))
in_dim(rp::AbstractRandomProjections) = size(getmap(rp), 1)
out_dim(rp::AbstractRandomProjections) = size(getmap(rp), 2)
Base.eltype(rp::AbstractRandomProjections) = eltype(getmap(rp))

function transform!(rp::AbstractRandomProjections, out::AbstractVector, v::AbstractVector)
    for (i, x) in enumerate(eachcol(getmap(rp)))
        @inbounds out[i] = dot32(x, v)
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
    for (i, x) in enumerate(eachcol(rp.inv))
        @inbounds out[i] = dot32(x, v)
    end

    out
end

function invtransform(rp::InvertibleRandomProjections, v::AbstractVector)
    out = Vector{eltype(rp)}(undef, in_dim(rp))
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

    O
end

function invtransform(rp::InvertibleRandomProjections, X::AbstractMatrix)
    O = Matrix{eltype(rp)}(undef, in_dim(rp), size(X, 2))
    invtransform!(rp, O, X)
end

