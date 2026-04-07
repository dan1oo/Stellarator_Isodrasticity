using LinearAlgebra
using Statistics
using Random
using JLD2
using Printf

# -----------------------------------------------------------------------------
# Configuration / CLI
# -----------------------------------------------------------------------------
const DEFAULT_IN_PATH = raw"C:/Users/danie/Stellarator_Isodrasticity/data/magnetic_ridge3/magnetic_ridge_3.jld2"
const DEFAULT_OUT_DIR = raw"C:/Users/danie/Stellarator_Isodrasticity/data/magnetic_ridge3"

function parse_k(args)
    if !isempty(args)
        return parse(Int, args[1])
    end
    return parse(Int, get(ENV, "RIDGE_CLUSTER_K", "3"))
end

function next_nonconflicting_path(path::AbstractString)
    isfile(path) || return path
    d = dirname(path)
    stem, ext = splitext(basename(path))
    n = 1
    while true
        cand = joinpath(d, string(stem, "_", n, ext))
        isfile(cand) || return cand
        n += 1
    end
end

function load_ridge_points(path::AbstractString)
    x, y, z = JLD2.jldopen(path, "r") do f
        if haskey(f, "ridge_x") && haskey(f, "ridge_y") && haskey(f, "ridge_z")
            Vector{Float64}(f["ridge_x"]),
            Vector{Float64}(f["ridge_y"]),
            Vector{Float64}(f["ridge_z"])
        elseif haskey(f, "new_rx") && haskey(f, "new_ry") && haskey(f, "new_rz")
            Vector{Float64}(f["new_rx"]),
            Vector{Float64}(f["new_ry"]),
            Vector{Float64}(f["new_rz"])
        else
            error("Input JLD2 must contain ridge_x/y/z or new_rx/new_ry/new_rz")
        end
    end
    @assert length(x) == length(y) == length(z)
    return x, y, z
end

# -----------------------------------------------------------------------------
# Lightweight k-means (kmeans++ init + Lloyd iterations)
# -----------------------------------------------------------------------------
@inline sqdist3(x1, y1, z1, x2, y2, z2) = (x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2

function kmeanspp_init(x::Vector{Float64}, y::Vector{Float64}, z::Vector{Float64}, k::Int, rng::AbstractRNG)
    n = length(x)
    cidx = Vector{Int}(undef, k)
    cidx[1] = rand(rng, 1:n)

    d2 = fill(Inf, n)
    for j in 2:k
        cj = cidx[j - 1]
        @inbounds for i in 1:n
            dij = sqdist3(x[i], y[i], z[i], x[cj], y[cj], z[cj])
            if dij < d2[i]
                d2[i] = dij
            end
        end
        s = sum(d2)
        if s <= 0
            cidx[j] = rand(rng, 1:n)
            continue
        end
        r = rand(rng) * s
        acc = 0.0
        chosen = n
        @inbounds for i in 1:n
            acc += d2[i]
            if acc >= r
                chosen = i
                break
            end
        end
        cidx[j] = chosen
    end
    return cidx
end

function run_kmeans3(
    x::Vector{Float64},
    y::Vector{Float64},
    z::Vector{Float64},
    k::Int;
    maxiter::Int = 200,
    tol::Float64 = 1e-6,
    seed::Int = 1234,
)
    n = length(x)
    @assert 1 <= k <= n

    # z-score normalize so one axis does not dominate
    μx, μy, μz = mean(x), mean(y), mean(z)
    σx, σy, σz = std(x), std(y), std(z)
    σx = σx == 0 ? 1.0 : σx
    σy = σy == 0 ? 1.0 : σy
    σz = σz == 0 ? 1.0 : σz
    xn = (x .- μx) ./ σx
    yn = (y .- μy) ./ σy
    zn = (z .- μz) ./ σz

    rng = MersenneTwister(seed)
    cidx = kmeanspp_init(xn, yn, zn, k, rng)
    cx = xn[cidx]; cy = yn[cidx]; cz = zn[cidx]

    labels = zeros(Int, n)
    prev_sse = Inf

    for it in 1:maxiter
        # assign
        sse = 0.0
        @inbounds for i in 1:n
            bestj = 1
            bestd = sqdist3(xn[i], yn[i], zn[i], cx[1], cy[1], cz[1])
            for j in 2:k
                d = sqdist3(xn[i], yn[i], zn[i], cx[j], cy[j], cz[j])
                if d < bestd
                    bestd = d
                    bestj = j
                end
            end
            labels[i] = bestj
            sse += bestd
        end

        # update
        sumx = zeros(k); sumy = zeros(k); sumz = zeros(k); cnt = zeros(Int, k)
        @inbounds for i in 1:n
            j = labels[i]
            sumx[j] += xn[i]; sumy[j] += yn[i]; sumz[j] += zn[i]
            cnt[j] += 1
        end

        for j in 1:k
            if cnt[j] == 0
                # re-seed empty cluster at random point
                r = rand(rng, 1:n)
                cx[j] = xn[r]; cy[j] = yn[r]; cz[j] = zn[r]
            else
                cx[j] = sumx[j] / cnt[j]
                cy[j] = sumy[j] / cnt[j]
                cz[j] = sumz[j] / cnt[j]
            end
        end

        rel = abs(prev_sse - sse) / max(prev_sse, 1.0)
        @printf("iter %3d | SSE = %.6e | rel_improve = %.3e\n", it, sse, rel)
        if rel < tol
            break
        end
        prev_sse = sse
    end

    # denormalize centers to original coordinates
    cx0 = cx .* σx .+ μx
    cy0 = cy .* σy .+ μy
    cz0 = cz .* σz .+ μz

    return labels, cx0, cy0, cz0
end

function split_clusters(x::Vector{Float64}, y::Vector{Float64}, z::Vector{Float64}, labels::Vector{Int}, k::Int)
    xs = [Float64[] for _ in 1:k]
    ys = [Float64[] for _ in 1:k]
    zs = [Float64[] for _ in 1:k]
    @inbounds for i in eachindex(labels)
        j = labels[i]
        push!(xs[j], x[i]); push!(ys[j], y[i]); push!(zs[j], z[i])
    end
    return xs, ys, zs
end

function main(args)
    k = parse_k(args)
    in_path = get(ENV, "RIDGE_CLUSTER_IN", DEFAULT_IN_PATH)
    out_dir = get(ENV, "RIDGE_CLUSTER_OUT_DIR", DEFAULT_OUT_DIR)
    maxiter = parse(Int, get(ENV, "RIDGE_CLUSTER_MAXITER", "200"))
    seed = parse(Int, get(ENV, "RIDGE_CLUSTER_SEED", "1234"))

    println("Loading ridge data from: ", in_path)
    x, y, z = load_ridge_points(in_path)
    n = length(z)
    println("Points: ", n, " | k = ", k)
    if k < 1 || k > n
        error("k must satisfy 1 <= k <= number of points")
    end

    labels, cx, cy, cz = run_kmeans3(x, y, z, k; maxiter = maxiter, seed = seed)
    xs, ys, zs = split_clusters(x, y, z, labels, k)
    sizes = [length(v) for v in xs]
    println("Cluster sizes: ", sizes)

    mkpath(out_dir)
    out_base = joinpath(out_dir, @sprintf("magnetic_ridge_3_clusters_k%d.jld2", k))
    out_path = next_nonconflicting_path(out_base)

    JLD2.jldopen(out_path, "w") do f
        f["k"] = k
        f["labels"] = labels
        f["center_x"] = cx
        f["center_y"] = cy
        f["center_z"] = cz
        f["cluster_sizes"] = sizes

        # grouped vectors for each discrete part
        f["cluster_x"] = xs
        f["cluster_y"] = ys
        f["cluster_z"] = zs

        # convenience aliases per cluster index (1-based)
        for j in 1:k
            f[@sprintf("cluster_%d_x", j)] = xs[j]
            f[@sprintf("cluster_%d_y", j)] = ys[j]
            f[@sprintf("cluster_%d_z", j)] = zs[j]
        end

        f["meta"] = Dict(
            "input_path" => in_path,
            "algorithm" => "kmeans3_kmeans++",
            "maxiter" => maxiter,
            "seed" => seed,
            "note" => "Coordinates z-score normalized for clustering, centers saved in original units.",
        )
    end

    println("Saved clustered ridge sets to: ", out_path)
end

main(ARGS)
