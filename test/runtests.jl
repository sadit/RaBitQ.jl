using RaBitQ
using Test
using AllocCheck, JET, LinearAlgebra, SimilaritySearch, JLD2, StatsBase, Downloads

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

    x_b = Vector{UInt64}(undef, ceil(Int, dim64(in_dim(rp))))
    RaBitQ.RaBitQ_bitencode_!(rp.inv, x_b, o)


    @test BitVector(sign.(x̄) .> 0).chunks == x_b
    @show x_b
    dot_ō_o_ = RaBitQ.RaBitQ_dot_ō_o(rp.map, m, x_b, o)
    #=try
        fun()
    catch err
        @info typeof(err)
        if typeof(err) !== MethodError
            for e in err.errors
                @info e
            end
        else
            rethrow(err)
        end
    end=#
    @test abs(dot_ō_o - dot_ō_o_) < 1e-4
    @info "analyzing allocating functions"

    @check_allocs function run(o, q)
        a = RaBitQ_estimate_dot(m, o, q)
        b = RaBitQ_estimate_dot(m, o, q)
        a, b
    end

    o_ = RaBitQ_Vector(x_b, RaBitQ_VectorInfo(dot_ō_o_, norm(oraw), RaBitQ.RaBitQ_dot_confidence_interval(dot_ō_o, length(oraw))))
    q_ = RaBitQ_QueryVector(invtransform(rp, q), norm(qraw))
    @time RaBitQ_estimate_dot(m, o_, q_)

    try
        @show run(o_, q_)
    catch err
        if err isa MethodError
            rethrow(err)
        end
        for e in err.errors
            display(e)
        end
    end

    @test_opt RaBitQ_estimate_dot(m, o_, q_)
    est_ = RaBitQ_estimate_dot(m, o_, q_)
    @show dot_o_q, est, est_
    @test abs(est - est_) < 1e-4
    @show o_.info
    @test 0.93 < dot_o_q / est_ < 1.07
    norm_oraw = norm(oraw)
    #@test abs(l2_oraw_c_ - evaluate(L2Distance(), x̄, Q.c)) < 1e-4
    est_l2 = evaluate(RaBitQ_L2Distance(m), o_, q_)
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
    dist = RaBitQ_CosineDistance(db.m)
    S = ExhaustiveSearch(; db, dist)
    queries = RaBitQ_Queries(db, Q)
    @time "computing knns" knns = searchbatch(S, ctx, queries, k)
    recall = macrorecall(gold_knns, knns)
    @show recall
    @test recall > 0.25
end

using InteractiveUtils


@testset "RaBit DB real" begin
    k = 10
    filename = "benchmark-dev-ccnews-fp16.h5"
    if !isfile(filename)
        Downloads.download("https://huggingface.co/datasets/sadit/SISAP2025/resolve/main/$filename?download=true", filename)
    end

    X, Q, gold_knns, gold_dists = jldopen(filename) do f
        kind = "otest"
        f["train"], f["$kind/queries"], f["$kind/knns"][1:k, :], f["$kind/dists"][1:k, :]
    end

    @time "ESTIMATION Database" db = RaBitQ_Database(X)
    dist = RaBitQ_CosineDistance(db.m)
    S = ExhaustiveSearch(; db, dist)
    ctx = getcontext(S)
    queries = RaBitQ_Queries(db, Q)
    knns = Matrix{IdWeight}(undef, k, length(queries))
    let x = db[1], q = queries[1]
        @time "dist" evaluate(dist, x, q)
        @time "dist" evaluate(dist, x, q)
        @time "estimate" RaBitQ_estimate_dot(db.m, x, q)
        @time "estimate" RaBitQ_estimate_dot(db.m, x, q)
        @code_warntype RaBitQ_estimate_dot(db.m, x, q)
        @code_warntype RaBitQ.RaBitQ_dot_ō_qinv(db.m, x.x_b, q.qinv)
    end

    @time "ESTIMATION computing knns" knns = searchbatch!(S, ctx, queries, knns)
    @time "ESTIMATION computing knns" knns = searchbatch!(S, ctx, queries, knns)
    recall = macrorecall(gold_knns, knns)
    @show recall
    @test recall > 0.59


    @info "\n\n========================"
    @info "Now using Hammming"
    @info "========================"

    rp = GaussianRandomProjection(Float32, size(X, 1), 4 * size(X, 1))
    db = MatrixDatabase(RaBitQ_bitencode(rp.map, X))

    S = ExhaustiveSearch(; db, dist=BinaryHammingDistance())
    ctx = getcontext(S)

    queriesQ = MatrixDatabase(RaBitQ_bitencode(rp.map, Q))
    @show size(X), size(Q)

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
