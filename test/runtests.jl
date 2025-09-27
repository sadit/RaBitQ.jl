using RaBitQ
using Test
using LinearAlgebra
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
        @test evaluate(dist, invtransform(rp, x̂), x) <= 0.2
    end

    for (rptype, err) in [
        QRRandomProjection => 25,
        GaussianRandomProjection => 25,
        Achioptas2RandomProjection => 25,
        Achioptas3RandomProjection => 25
    ]

        odim = 128
        @info rptype, err, odim
        rp = rptype(Float32, dim, odim) |> invertible
        @show size(rp)

        @test in_dim(rp) == dim
        @test out_dim(rp) == odim
        x̂ = transform(rp, x)
        ŷ = transform(rp, y)
        d2 = evaluate(dist, x̂, ŷ)
        @test abs(d1 - d2) < err
        @test evaluate(dist, invtransform(rp, x̂), x) < 18
    end

end

@testset "RaBitQ" begin
    # Write your tests here.
    dim = 384
    rp = invertible(QRRandomProjection(Float32, dim))
    oraw = rand(-1.0f0:1.0f-9:1.0f0, dim)
    o = normalize(oraw)
    qraw = rand(-1.0f0:1.0f-9:1.0f0, dim)
    qraw .= o .+ 0.03 .* qraw
    q = normalize(qraw)

    m = Float32(1 / sqrt(dim))
    #ō = transform(rp, [ifelse(x >= 0, m, -m) for x in o])
    #p
    x̄ = Float32[ifelse(x >= 0, m, -m) for x in invtransform(rp, o)]
    ō = transform(rp, x̄)
    dot_o_q = dot(o, q)
    dot_o_ō = dot(o, ō)
    @show dot_o_q
    @show dot(o, ō)
    @show dot(q, ō)
    est = dot(q, ō) / dot(o, ō)
    @show est
    @show evaluate(L2Distance(), oraw, qraw)

    Q = BitQuantizer(m, rp, zeros(Float32, in_dim(rp)))
    x_b = Vector{UInt64}(undef, ceil(Int, in_dim(Q.rp) / 64))
    rabitq_bitencode!(Q, x_b, o)
    @test BitVector(sign.(x̄) .> 0).chunks == x_b
    @show x_b
    dot_o_ō_ = rabitq_dot(Q, x_b, o)
    @test abs(dot_o_ō - dot_o_ō_) < 1e-4
    est_ = rabitq_estimate_dot(Q, x_b, dot_o_ō_, q)
    @show dot_o_q, est, est_
    @test abs(est - est_) < 1e-4
    @test 0.95 < dot_o_q / est_ < 1.05
    # L2
    l2_oraw_c_ = evaluate(L2_asf32(), oraw, Q.c)
    #@test abs(l2_oraw_c_ - evaluate(L2Distance(), x̄, Q.c)) < 1e-4
    @show rabitq_estimate_l2(Q, x_b, dot_o_ō_, l2_oraw_c_, qraw, norm(qraw))

end

@testset "RaBitQ Database" begin end
