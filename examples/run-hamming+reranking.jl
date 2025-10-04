using RaBitQ, SimilaritySearch, JLD2

function main(; filename, k=10)
    X, Q, gold_knns = jldopen(filename) do f
        kind = "otest"
        f["train"], f["$kind/queries"], f["$kind/knns"][1:k, :]
    end

    @time "RaBitQ transform" dbQ = RaBitQ_Database(X)
    @info typeof(dbQ)
    quant = dbQ.quant

    # m, n = dim64(dbQ.quant), size(Q, 2)
    ##quant = Achioptas3RandomProjection(Float32, size(X, 1), size(X, 1) * 2) |> invertible
    ##quant = RaBitQ_Quantizer(Float32(1/sqrt(768)), quant)
    db = MatrixDatabase(dbQ.matrix)
    ##db = MatrixDatabase(RaBitQ_bitencode(quant, X))
    queriesQ = MatrixDatabase(RaBitQ_bitencode(quant, Q))
    dist = BinaryHammingDistance()
    G = SearchGraph(; db, dist)
    ctx = SearchGraphContext(KnnHeap;
        neighborhood=Neighborhood(SatNeighborhood(; nndist=1f0), logbase=1.3), 
        hints_callback=KCentersHints(; logbase=1.2),
        hyperparameters_callback=OptimizeParameters(MinRecall(0.99))
    )

    @time "index!" index!(G, ctx)
    optimize_index!(G, ctx, MinRecall(0.95))
    @time "searchbatch!" knns = searchbatch(G, ctx, queriesQ, k)
    @time "searchbatch!" knns = searchbatch(G, ctx, queriesQ, 20k)
    recall = macrorecall(Set.(eachcol(gold_knns)), Set.(IdView.(eachcol(knns))))
    @show recall, size(knns)
    G
end

 main(filename="examples/datasets/benchmark-dev-ccnews-fp16.h5", k=10)
