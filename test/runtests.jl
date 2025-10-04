using RaBitQ
using Test
using AllocCheck, LinearAlgebra, SimilaritySearch, JLD2, StatsBase, Downloads

@testset "Random Projections" begin
    dim = 384
    x = rand((-1.0f0, 1.0f0), dim)
    y = rand((-1.0f0, 1.0f0), dim)
    dist = L2Distance()
    d1 = evaluate(dist, x, y)

    for (rptype, err) in [
        QRRandomProjection => 0.001,
        GaussianRandomProjection => 3.0,
        Achioptas2RandomProjection => 3.3,
        Achioptas3RandomProjection => 3.3
    ]
        @info rptype, err, dim
        rp = rptype(Float32, dim) |> invertible
        x̂ = transform(rp, x)
        ŷ = transform(rp, y)
        d2 = evaluate(dist, x̂, ŷ)
        @test abs(d1 - d2) < err
        @test evaluate(dist, invtransform(rp, x̂), x) <= 0.27
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
    dot_ō_o = dot(o, ō)
    @show dot_o_q
    @show dot(o, ō)
    @show dot(q, ō)
    est = dot(q, ō) / dot(o, ō)
    @show est
    l2 = evaluate(L2Distance(), oraw, qraw)

    Q = RaBitQ_Quantizer(m, rp)
    x_b = Vector{UInt64}(undef, ceil(Int, in_dim(Q.rp) / 64))
    RaBitQ.RaBitQ_bitencode_!(Q.rp.inv, x_b, o)
    @test BitVector(sign.(x̄) .> 0).chunks == x_b
    @show x_b
    dot_ō_o_ = RaBitQ.RaBitQ_dot_ō_o(Q, x_b, o)
    @test abs(dot_ō_o - dot_ō_o_) < 1e-4
    @info "analyzing allocating functions"
    @check_allocs function run(o, q)
        a = RaBitQ_estimate_dot(Q, o, q)
        b = RaBitQ_estimate_dot(Q, o, q)
        a, b
    end

    o_ = RaBitQ_Vector(x_b, RaBitQ_VectorInfo(dot_ō_o_, norm(oraw), RaBitQ.RaBitQ_dot_confidence_interval(dot_ō_o, length(oraw))))
    q_ = RaBitQ_QueryVector(invtransform(Q.rp, q), norm(qraw))
    try
        @show run(o_, q_)
    catch err
        #rethrow(err)
        for e in err.errors
            display(e)
        end
    end

    est_ = RaBitQ_estimate_dot(Q, o_, q_)
    @show dot_o_q, est, est_
    @test abs(est - est_) < 1e-4
    @show o_.info
    @test 0.93 < dot_o_q / est_ < 1.07
    norm_oraw = norm(oraw)
    #@test abs(l2_oraw_c_ - evaluate(L2Distance(), x̄, Q.c)) < 1e-4
    est_l2 = evaluate(RaBitQ_L2Distance(Q), o_, q_)
    @show l2, est_l2
    @test abs(l2 - est_l2) < 0.15
end

function gendata(n, dim)
    X = rand((-1.0f0, 1.0f0), dim, n)
    for c in eachcol(X)
        normalize!(c)
    end

    X
end

@testset "RaBitQ Database" begin
    dim = 384
    k = 16
    X = gendata(2^12, dim)
    Q = gendata(128, dim)

    G = ExhaustiveSearch(; db=MatrixDatabase(X), dist=CosineDistance())
    ctx = getcontext(G)
    @time "computing gold" gold_knns = searchbatch(G, ctx, MatrixDatabase(Q), k)
    @time db = RaBitQ_Database(X)

    dist = RaBitQ_CosineDistance(db.quant)
    S = ExhaustiveSearch(; db, dist)
    queries = RaBitQ_Queries(db.quant, Q)
    @time "computing knns" knns = searchbatch(S, ctx, queries, k)
    recall = macrorecall(gold_knns, knns)
    @show recall
    @test recall > 0.25
end


@testset "RaBitQ ReRanking Hamming" begin
    k = 10
    filename = "benchmark-dev-ccnews-fp16.h5"
    if !isfile(filename)
        Downloads.download("https://huggingface.co/datasets/sadit/SISAP2025/resolve/main/$filename?download=true", filename)
    end

    X, Q, gold_knns, gold_dists = jldopen(filename) do f
        kind = "otest"
        f["train"], f["$kind/queries"], f["$kind/knns"][1:k, :], f["$kind/dists"][1:k, :]
    end

    rp = GaussianRandomProjection(Float32, size(X, 1), 3size(X, 1))
    #rp = QRRandomProjection(Float32, size(X, 1))
    db = MatrixDatabase(RaBitQ_bitencode(rp.map, X))

    S = ExhaustiveSearch(; db, dist=BinaryHammingDistance())
    ctx = getcontext(S)
    
    queriesQ = MatrixDatabase(RaBitQ_bitencode(rp.map, Q))
    #queries = RaBitQ_Queries(dbQ.quant, Q)
    @show size(X), size(Q)
    #dist = RaBitQ_CosineDistance(dbQ.quant)

    local knns
    for (Δ, minrecall) in [2 => 0.6, 8 => 0.8, 16 => 0.9]
        @info "==================="
        @time "ReRanking computing knns Δ=$Δ" knns = searchbatch(S, ctx, MatrixDatabase(queriesQ), Δ * k)
        @show Δ, size(gold_knns), size(knns)
        recall = macrorecall(Set.(eachcol(gold_knns)), Set.(IdView.(eachcol(knns))))
        @show Δ, recall
        @test recall > minrecall
        @show quantile(gold_dists[1, :], 0:0.25:1.0)
        @show quantile(gold_dists[:, 1], 0:0.25:1.0)
        #=@show "DIST HAMMING"
        @show quantile(DistView(knns[1, :]), 0:0.25:1.0)
        @show quantile(DistView(knns[:, 1]), 0:0.25:1.0)
        @show "RERANKED Estimation"=#
        #=@time "RERANK" rerank!(dist, dbQ, queries, knns)
        @show quantile(DistView(knns[1, :]), 0:0.25:1.0)
        @show quantile(DistView(knns[1:k, 1]), 0:0.25:1.0)
        @show "reranked Δ=$Δ", macrorecall(gold_knns, knns)=#
    end

    @info "======== Real ReRank"
    @time "RERANK" rerank!(NormalizedCosine_asf32(), MatrixDatabase(X), MatrixDatabase(Q), knns)
    @time "RERANK" rerank!(NormalizedCosine_asf32(), MatrixDatabase(X), MatrixDatabase(Q), knns)
    @show quantile(DistView(knns[1, :]), 0:0.25:1.0)
    @show quantile(DistView(knns[1:k, 1]), 0:0.25:1.0)
    recall = macrorecall(gold_knns, knns)
    @show "reranked ", recall
    @test recall > 0.9
    # @info quantile([p.err for p in dbQ.info], 0:0.25:1)
end

