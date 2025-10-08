using RaBitQ, SimilaritySearch, JLD2, DataFrames, CSV

function main(;
    filename,
    klist,
    αlist=[1, 2, 4, 8, 12, 16],
    κlist=[1, 2, 4, 8, 16, 32],
    outdir="results",
)
    outname_ = joinpath(outdir, replace(basename(filename), ".h5" => ""))

    X, oQ, ogold_knns, iQ, igold_knns = jldopen(filename) do f
        Matrix{Float16}(f["train"]), Matrix{Float16}(f["otest/queries"]), f["otest/knns"], Matrix{Float16}(f["itest/queries"]), f["itest/knns"]
    end

    dim = size(X, 1)

    for α in αlist
        outname = outname_ * "-alpha=$α.csv"
        if isfile(outname)
            @info "$outname already exists"
            continue
        end

        D = DataFrame(kind=[], k=[], α=[], κ=[], recall=[], query_time=[], indexing_time=[], rerank_time=[], rerank_recall=[], db_enc_time=[], query_enc_time=[], filename=[])
        # rp = GaussianRandomProjection(dim, α * dim)
        rp = QRXRandomProjection(dim, α * dim)
        db_enc_time = @elapsed db = MatrixDatabase(RaBitQ_bitencode(rp.map, X))

        dist = BinaryHammingDistance()
        G = SearchGraph(; db, dist)
        ctx = SearchGraphContext(KnnHeap;
            neighborhood=Neighborhood(SatNeighborhood(; nndist=1.0f0), logbase=1.2),
            hints_callback=KCentersHints(; logbase=1.2),
            hyperparameters_callback=OptimizeParameters(MinRecall(0.99))
        )

        indexing_time = @elapsed index!(G, ctx)
        optimize_index!(G, ctx, MinRecall(0.95))

        rerank_dist = NormalizedCosine_asf32()

        for (kind, Q, gold_knns_) in [("otest", oQ, ogold_knns), ("itest", iQ, igold_knns)]
            query_enc_time = @elapsed queries = MatrixDatabase(RaBitQ_bitencode(rp.map, Q))
            for k in klist
                gold_knns = view(gold_knns_, 1:k, :)
                for κ in κlist
                    query_time = @elapsed knns = searchbatch(G, ctx, queries, κ * k)
                    recall = macrorecall(Set.(eachcol(gold_knns)), Set.(IdView.(eachcol(knns))))
                    rerank_time = @elapsed rerank!(rerank_dist, MatrixDatabase(X), MatrixDatabase(Q), knns)
                    rerank_recall = macrorecall(Set.(eachcol(gold_knns)), Set.(IdView.(eachcol(knns))))

                    push!(D, (; kind, k, α, κ, recall, query_time, indexing_time, rerank_time, rerank_recall, db_enc_time, query_enc_time, filename))
                    display(D)
                end
            end
        end

        CSV.write(outname, D)
    end

end

for kind in ["otest", "itest"]
    klist = [1, 3, 10, 30, 100]
    #main(; filename="datasets/benchmark-dev-ccnews.h5", klist)
    #main(; filename="datasets/benchmark-dev-yahooaq.h5", klist)
    #main(; filename="datasets/benchmark-dev-gooaq.h5", klist)
    main(; filename="datasets/benchmark-dev-pubmed23.h5", klist)
end
