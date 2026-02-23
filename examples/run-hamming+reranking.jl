using RaBitQ, SimilaritySearch, JLD2, DataFrames, CSV, JSON
using RaBitQ: transform

function create_index(::Type{<:SearchGraph}, dist, db)
    G = SearchGraph(; dist, db)
    m = Dict{String,Any}("name" => "SearchGraph")
    m["neighborhood"] = neighborhood = Neighborhood(SatNeighborhood(; nndist=1.0f0), logbase=1.2)
    m["recall_build"] = recall_build = 0.99
    m["recall_search"] = recall_search = 0.95
    m["dist"] = string(dist)

    ctx = SearchGraphContext(KnnHeap;
        neighborhood,
        hints_callback=KCentersHints(; logbase=1.2),
        hyperparameters_callback=OptimizeParameters(MinRecall(recall_build))
    )

    m["build_time"] = @elapsed index!(G, ctx)
    m["opt_time"] = @elapsed optimize_index!(G, ctx, MinRecall(recall_search))
    G, ctx, m
end

function create_index(::Type{<:ExhaustiveSearch}, dist, db)
    m = Dict{String,Any}("name" => "ExhaustiveSearch")
    m["dist"] = string(dist)
    m["build_time"] = 0.0
    G = ExhaustiveSearch(; db, dist)
    ctx = getcontext(G)
    G, ctx, m
end

function main(indextype;
    filename,
    klist,
    αlist=[1, 2, 4, 8, 12, 16],
    κlist=[1, 2, 4, 8, 16, 32],
    outdir,
)
    mkpath(outdir)

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
        
        D = []
        meta_rerank = Dict{String,Any}()
        meta_enc = Dict{String,Any}("α" => α)
        meta = Dict{String,Any}("rerank" => meta_rerank, "enc" => meta_enc, "db" => filename)
        
        # D = DataFrame(kind=[], k=[], α=[], κ=[], recall=[], query_time=[], indexing_time=[], rerank_time=[], rerank_recall=[], db_enc_time=[], query_enc_time=[], index_params=[], search_params=[], db=[])
        H = RaBitQ_HammingEncoder(QRXRandomProjection(dim, α * dim))
        # H = RaBitQ_HammingEncoder(GaussianRandomProjection(dim, α * dim))

        meta_enc["database_time"] = @elapsed db = MatrixDatabase(transform(H, X))
        dist = BinaryHammingDistance()
        G, ctx, meta["index"] = create_index(indextype, dist, db)
        rerank_dist = NormalizedCosine_asf32()
        meta_rerank["dist"] = string(rerank_dist)

        for (kind, Q, gold_knns_) in [("itest", iQ, igold_knns), ("otest", oQ, ogold_knns)]
            meta_enc["query_time"] = @elapsed queries = MatrixDatabase(transform(H, Q))
            if kind == "otest" && occursin("wit-multi", filename)
                optimize_index!(G, ctx, MinRecall(0.95); queries=rand(queries, 64), ksearch=10)
                # B = G.algo[]
                # G.algo[] = BeamSearch(12, 1.27, 50000)
            end

            for k in klist
                meta["search"] = Dict("k" => k, "kind" => kind)
                if indextype == SearchGraph
                    meta["search"]["params"] => G.algo[]
                end
                
                gold_knns = view(gold_knns_, 1:k, :)
                for κ in κlist
                    meta_rerank["κ"] = κ
                    meta["query_time"] = @elapsed knns = searchbatch(G, ctx, queries, κ * k; sorted=true)
                    meta["recall"] = macrorecall(Set.(eachcol(gold_knns)), Set.(IdView.(eachcol(knns))))
                    meta_rerank["time"] = @elapsed rerank!(rerank_dist, MatrixDatabase(X), MatrixDatabase(Q), knns)
                    meta_rerank["recall"] = macrorecall(gold_knns, knns)

                    push!(D, deepcopy(meta))
                    display(last(D))
                end
            end
        end


        open(outname, "w") do f
            for rec in D
                println(f, json(rec))
            end
        end
    end

end

let outdir = "results-hamming+searchgraph"
    klist = [10, 30, 100]
    main(SearchGraph; filename="datasets/benchmark-dev-ccnews.h5", klist, outdir)
    main(SearchGraph; filename="datasets/benchmark-dev-yahooaq.h5", klist, outdir)
    #main(SearchGraph; filename="datasets/benchmark-dev-gooaq.h5", klist, outdir)
    #main(SearchGraph; filename="datasets/benchmark-dev-wit-multi-800k.h5", klist, outdir)
    #main(SearchGraph; filename="datasets/benchmark-dev-pubmed23.h5", klist, outdir)
end

let outdir = "results-hamming+bruteforce"
    klist = [10, 30, 100]
    main(ExhaustiveSearch; filename="datasets/benchmark-dev-ccnews.h5", klist, outdir)
    main(ExhaustiveSearch; filename="datasets/benchmark-dev-yahooaq.h5", klist, outdir)
    #main(ExhaustiveSearch; filename="datasets/benchmark-dev-gooaq.h5", klist, outdir)
    #main(ExhaustiveSearch; filename="datasets/benchmark-dev-wit-multi-800k.h5", klist, outdir)
    #main(ExhaustiveSearch; filename="datasets/benchmark-dev-pubmed23.h5", klist, outdir)
end
