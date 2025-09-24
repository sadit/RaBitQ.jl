using RaBitQ
using Test
using SimilaritySearch

@testset "Random Projections" begin
    dim = 384
    x = rand((-1.0f0, 1.0f0), dim)
    y = rand((-1.0f0, 1.0f0), dim)
    dist = L2Distance()
    d1 = evaluate(dist, x, y)

    for (rptype, err) in [
        QRRandomProjection => 0.001,
        GaussianRandomProjection => 3.0,
        Achioptas2RandomProjection => 3.0,
        Achioptas3RandomProjection => 3.0
    ]
        @info rptype, err, dim
        rp = rptype(Float32, dim) |> invertible
        x̂ = transform(rp, x)
        ŷ = transform(rp, y)
        d2 = evaluate(dist, x̂, ŷ)
        @test abs(d1 - d2) < err
        @test evaluate(dist, invtransform(rp, x̂), x) <= 0.1
    end

    for (rptype, err) in [
        QRRandomProjection => 30,
        GaussianRandomProjection => 30,
        Achioptas2RandomProjection => 30,
        Achioptas3RandomProjection => 30
    ]
        @info rptype, err, 128
        rp = rptype(Float32, dim, 128)
        x̂ = transform(rp, x)
        ŷ = transform(rp, y)
        d2 = evaluate(dist, x̂, ŷ)
        @test abs(d1 - d2) < err
    end

end

@testset "RaBitQ.jl" begin
    # Write your tests here.
end
