# magnetic_ridge_valley_3.jl
#
# Same pipeline as magnetic_ridge_3.jl, but selects **magnetic ridge valleys** (positive
# second derivative of |B| along field direction) instead of ridges.

using LinearAlgebra
using ForwardDiff
using Roots
using JLD2
using GLMakie
using StaticArrays

include("Coil.jl")

"""
    next_nonconflicting_path(path)

If `path` is not an existing file, return `path`. Otherwise return `dir/stem_n.ext`
for the smallest positive integer `n` such that that file does not exist
(e.g. `run.jld2` → `run_1.jld2` → `run_2.jld2`).
"""
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

# -----------------------------------------------------------------------------
# Configuration (MAGVALLEY3_* overrides; falls back to MAGRID3_* then defaults)
# -----------------------------------------------------------------------------
const COIL_FILE = "landreman_paul.json"
const NQUAD = 64

const CONF_PATH = raw"C:/Users/danie/Stellarator_Isodrasticity/data/confinement_zone/confinement_field_lines.jld2"

const NX = parse(Int, get(ENV, "MAGVALLEY3_NX", get(ENV, "MAGRID3_NX", "200")))
const NY = parse(Int, get(ENV, "MAGVALLEY3_NY", get(ENV, "MAGRID3_NY", "200")))

const FOOTPRINT_NTHETA = parse(Int, get(ENV, "MAGVALLEY3_FOOTPRINT_NTHETA", get(ENV, "MAGRID3_FOOTPRINT_NTHETA", "180")))
const FOOTPRINT_OUTER_PAD = parse(Float64, get(ENV, "MAGVALLEY3_FOOTPRINT_OUTER_PAD", get(ENV, "MAGRID3_FOOTPRINT_OUTER_PAD", "0.07")))
const FOOTPRINT_INNER_PAD = parse(Float64, get(ENV, "MAGVALLEY3_FOOTPRINT_INNER_PAD", get(ENV, "MAGRID3_FOOTPRINT_INNER_PAD", "0.04")))
const XY_BOX_PAD = parse(Float64, get(ENV, "MAGVALLEY3_XY_BOX_PAD", get(ENV, "MAGRID3_XY_BOX_PAD", "0.03")))
const PLOT_MASK_DEBUG = parse(Bool, get(ENV, "MAGVALLEY3_PLOT_MASK_DEBUG", get(ENV, "MAGRID3_PLOT_MASK_DEBUG", "true")))
const MASK_ONLY = parse(Bool, get(ENV, "MAGVALLEY3_MASK_ONLY", get(ENV, "MAGRID3_MASK_ONLY", "false")))
const MAGVALLEY3_PROGRESS_EVERY = parse(Int, get(ENV, "MAGVALLEY3_PROGRESS_EVERY", get(ENV, "MAGRID3_PROGRESS_EVERY", "0")))

const XY_RADIUS_BASE = parse(Float64, get(ENV, "MAGVALLEY3_XY_RADIUS_BASE", get(ENV, "MAGRID3_XY_RADIUS_BASE", "0.06")))
const XY_RADIUS_MAX = parse(Float64, get(ENV, "MAGVALLEY3_XY_RADIUS_MAX", get(ENV, "MAGRID3_XY_RADIUS_MAX", "0.25")))
const MIN_NEIGHBORS = parse(Int, get(ENV, "MAGVALLEY3_MIN_NEIGHBORS", get(ENV, "MAGRID3_MIN_NEIGHBORS", "18")))
const Z_PAD = parse(Float64, get(ENV, "MAGVALLEY3_Z_PAD", get(ENV, "MAGRID3_Z_PAD", "0.01")))

const B_MAG_MAX = 10.0
const VALLEY_CURV_EPS = parse(Float64, get(ENV, "MAGVALLEY3_VALLEY_CURV_EPS", get(ENV, "MAGRID3_RIDGE_CURV_EPS", "1e-5")))

# -----------------------------------------------------------------------------
# Type-agnostic B-field for ForwardDiff jacobian
# -----------------------------------------------------------------------------
function B_agnostic(cs::Vector{Coil{T}}, x::AbstractVector) where {T}
    B_acc = StaticArrays.zeros(eltype(x), 3)
    for c in cs
        for (ri, Idri) in zip(c.rs, c.Idrs)
            delta_r = SVector(ri[1] - x[1], ri[2] - x[2], ri[3] - x[3])
            d = three_norm(delta_r)
            B_acc = B_acc + cross(delta_r, Idri) / (d^3)
        end
    end
    return B_acc
end

function d_normB_ds(cs::Vector{Coil{T}}, x, y, z) where {T}
    u = SVector{3}(x, y, z)
    B = eval_B(cs, u)
    J = ForwardDiff.jacobian(v -> B_agnostic(cs, v), u)
    return dot(B, J, B)
end

"""Valley: finite-difference derivative of `d(|B|)/ds` along **B** is **positive** (ridge uses negative)."""
function is_valley(cs::Vector{Coil{T}}, x, y, z) where {T}
    u = SVector{3}(x, y, z)
    B = eval_B(cs, u)
    nb = norm(B)
    nb < eps(Float64) * 1e6 && return false
    bh = B / nb
    epss = VALLEY_CURV_EPS
    s0 = d_normB_ds(cs, x, y, z)
    s1 = d_normB_ds(cs, x + epss * bh[1], y + epss * bh[2], z + epss * bh[3])
    return ((s1 - s0) / epss) > 0.0
end

# -----------------------------------------------------------------------------
# Confinement cloud indexing in XY (uniform hash grid)
# -----------------------------------------------------------------------------
struct XYIndex
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    h::Float64
    invh::Float64
    bins::Dict{Tuple{Int, Int}, Vector{Int}}
end

@inline cell_id(x, y, invh) = (floor(Int, x * invh), floor(Int, y * invh))

function build_xy_index(cx::Vector{Float64}, cy::Vector{Float64}, cz::Vector{Float64}; h::Float64 = XY_RADIUS_MAX)
    bins = Dict{Tuple{Int, Int}, Vector{Int}}()
    invh = 1.0 / h
    @inbounds for i in eachindex(cx)
        key = cell_id(cx[i], cy[i], invh)
        if haskey(bins, key)
            push!(bins[key], i)
        else
            bins[key] = [i]
        end
    end
    return XYIndex(cx, cy, cz, h, invh, bins)
end

function gather_within_radius(idx::XYIndex, x::Float64, y::Float64, r::Float64)
    ii, jj = cell_id(x, y, idx.invh)
    rr = ceil(Int, r / idx.h)
    r2 = r * r
    out = Int[]
    for i in (ii - rr):(ii + rr), j in (jj - rr):(jj + rr)
        ids = get(idx.bins, (i, j), nothing)
        ids === nothing && continue
        @inbounds for k in ids
            dx = idx.x[k] - x
            dy = idx.y[k] - y
            if dx * dx + dy * dy <= r2
                push!(out, k)
            end
        end
    end
    return out
end

function local_confine_bounds(idx::XYIndex, x::Float64, y::Float64, dx::Float64, dy::Float64)
    r = max(XY_RADIUS_BASE, 0.75 * hypot(dx, dy))
    ids = gather_within_radius(idx, x, y, r)
    while length(ids) < MIN_NEIGHBORS && r < XY_RADIUS_MAX
        r = min(XY_RADIUS_MAX, 1.55 * r)
        ids = gather_within_radius(idx, x, y, r)
    end
    isempty(ids) && return nothing

    zmin = Inf
    zmax = -Inf
    @inbounds for k in ids
        zk = idx.z[k]
        if zk < zmin
            zmin = zk
        end
        if zk > zmax
            zmax = zk
        end
    end
    return (zmin - Z_PAD, zmax + Z_PAD, length(ids), r)
end

# -----------------------------------------------------------------------------
# XY footprint from confinement cloud (avoid square-domain sampling)
# -----------------------------------------------------------------------------
struct XYFootprint
    ntheta::Int
    rmin::Vector{Float64}
    rmax::Vector{Float64}
end

@inline function theta_bin(theta::Float64, ntheta::Int)
    t = (theta + π) / (2π)
    return clamp(floor(Int, t * ntheta) + 1, 1, ntheta)
end

function _fill_circular_nearest!(v::Vector{Float64}; is_max::Bool = false)
    n = length(v)
    finite_idx = findall(isfinite, v)
    isempty(finite_idx) && error("Footprint bins are empty; confinement cloud may be invalid.")
    for i in eachindex(v)
        isfinite(v[i]) && continue
        best_j = finite_idx[1]
        best_d = min(abs(i - best_j), n - abs(i - best_j))
        for j in finite_idx
            d = min(abs(i - j), n - abs(i - j))
            if d < best_d
                best_d = d
                best_j = j
            end
        end
        v[i] = v[best_j]
    end
    return v
end

function build_xy_footprint(cx::Vector{Float64}, cy::Vector{Float64}; ntheta::Int = FOOTPRINT_NTHETA)
    rmin = fill(Inf, ntheta)
    rmax = fill(-Inf, ntheta)
    @inbounds for i in eachindex(cx)
        x = cx[i]
        y = cy[i]
        r = hypot(x, y)
        b = theta_bin(atan(y, x), ntheta)
        if r < rmin[b]
            rmin[b] = r
        end
        if r > rmax[b]
            rmax[b] = r
        end
    end

    _fill_circular_nearest!(rmin)
    _fill_circular_nearest!(rmax)

    rmin_s = similar(rmin)
    rmax_s = similar(rmax)
    for i in 1:ntheta
        i0 = mod1(i - 1, ntheta)
        i1 = i
        i2 = mod1(i + 1, ntheta)
        rmin_s[i] = 0.25 * rmin[i0] + 0.5 * rmin[i1] + 0.25 * rmin[i2]
        rmax_s[i] = 0.25 * rmax[i0] + 0.5 * rmax[i1] + 0.25 * rmax[i2]
    end
    return XYFootprint(ntheta, rmin_s, rmax_s)
end

@inline function in_xy_footprint(fp::XYFootprint, x::Float64, y::Float64)
    r = hypot(x, y)
    b = theta_bin(atan(y, x), fp.ntheta)
    rlo = max(0.0, fp.rmin[b] - FOOTPRINT_INNER_PAD)
    rhi = fp.rmax[b] + FOOTPRINT_OUTER_PAD
    return (r >= rlo) && (r <= rhi)
end

function save_xy_mask_debug_figure(
    conf_x::Vector{Float64},
    conf_y::Vector{Float64},
    fp::XYFootprint,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64;
    nx::Int = NX,
    ny::Int = NY,
)
    xs = collect(range(x_min, x_max; length = nx))
    ys = collect(range(y_min, y_max; length = ny))
    x_in = Float64[]
    y_in = Float64[]
    x_out = Float64[]
    y_out = Float64[]
    for x in xs, y in ys
        if in_xy_footprint(fp, x, y)
            push!(x_in, x)
            push!(y_in, y)
        else
            push!(x_out, x)
            push!(y_out, y)
        end
    end

    th = range(-π, π; length = fp.ntheta + 1)
    x_outer = Float64[]
    y_outer = Float64[]
    x_inner = Float64[]
    y_inner = Float64[]
    for t in th
        b = theta_bin(t, fp.ntheta)
        rlo = max(0.0, fp.rmin[b] - FOOTPRINT_INNER_PAD)
        rhi = fp.rmax[b] + FOOTPRINT_OUTER_PAD
        push!(x_outer, rhi * cos(t)); push!(y_outer, rhi * sin(t))
        push!(x_inner, rlo * cos(t)); push!(y_inner, rlo * sin(t))
    end

    fig = Figure(size = (1000, 900))
    ax = Axis(
        fig[1, 1];
        title = "2D XY footprint mask used by magnetic_ridge_valley_3",
        xlabel = "X (m)",
        ylabel = "Y (m)",
        aspect = DataAspect(),
    )
    !isempty(x_out) && scatter!(ax, x_out, y_out; markersize = 5, color = (:lightgray, 0.9))
    !isempty(x_in) && scatter!(ax, x_in, y_in; markersize = 6, color = (:seagreen, 0.95))
    conf_step = max(1, fld(length(conf_x), min(length(conf_x), 5000)))
    scatter!(ax, conf_x[1:conf_step:end], conf_y[1:conf_step:end]; markersize = 1.5, color = (:black, 0.25))
    lines!(ax, x_outer, y_outer; color = :dodgerblue, linewidth = 2)
    lines!(ax, x_inner, y_inner; color = :tomato, linewidth = 2)

    out = joinpath(@__DIR__, "magnetic_ridge_valley_3_mask2d.png")
    save(out, fig)
    println("Saved 2D footprint mask figure to ", out)
end

# -----------------------------------------------------------------------------
# Main scan
# -----------------------------------------------------------------------------
function map_valley_surface(
    cs::Vector{Coil{T}},
    idx::XYIndex,
    fp::XYFootprint,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64;
    nx::Int = NX,
    ny::Int = NY,
) where {T}
    xs = collect(range(x_min, x_max; length = nx))
    ys = collect(range(y_min, y_max; length = ny))
    dx = (x_max - x_min) / max(nx - 1, 1)
    dy = (y_max - y_min) / max(ny - 1, 1)

    valley_x = Float64[]
    valley_y = Float64[]
    valley_z = Float64[]

    multi_valley_xy_x = Float64[]
    multi_valley_xy_y = Float64[]
    multi_valley_valid_zs = Vector{Vector{Float64}}()
    multi_valley_chosen_z = Float64[]

    grid_x = Float64[]
    grid_y = Float64[]
    grid_z = Float64[]

    no_local_conf = 0
    outside_footprint = 0
    no_root = 0
    multi_root_xy = 0

    total_in_footprint = 0
    for x in xs
        for y in ys
            total_in_footprint += in_xy_footprint(fp, x, y) ? 1 : 0
        end
    end
    progress_every = if MAGVALLEY3_PROGRESS_EVERY > 0
        MAGVALLEY3_PROGRESS_EVERY
    else
        max(1, total_in_footprint ÷ 100)
    end

    println("Scan grid: nx=$nx ny=$ny (footprint-masked, non-rectangular effective domain)")
    println("Footprint (x,y) columns to process: $total_in_footprint (progress every ~$progress_every)")
    i_footprint = 0
    for x in xs
        for y in ys
            push!(grid_x, x)
            push!(grid_y, y)

            if !in_xy_footprint(fp, x, y)
                outside_footprint += 1
                push!(grid_z, NaN)
                continue
            end

            i_footprint += 1
            if total_in_footprint > 0 &&
               (i_footprint % progress_every == 0 || i_footprint == total_in_footprint)
                pct = 100 * i_footprint / total_in_footprint
                print(
                    "\rValley scan (inside footprint): ",
                    i_footprint,
                    " / ",
                    total_in_footprint,
                    " (",
                    round(pct; digits = 1),
                    "%)   ",
                )
                flush(stdout)
            end

            bounds = local_confine_bounds(idx, x, y, dx, dy)
            if bounds === nothing
                no_local_conf += 1
                push!(grid_z, NaN)
                continue
            end
            z_lo, z_hi, _, _ = bounds
            z_lo >= z_hi && (push!(grid_z, NaN); no_root += 1; continue)

            fz(z) = d_normB_ds(cs, x, y, z)
            roots = try
                find_zeros(fz, z_lo, z_hi)
            catch
                Float64[]
            end

            valid_z = Float64[]
            valid_B = Float64[]
            for z in roots
                Bmag = norm(eval_B(cs, SVector{3}(x, y, z)))
                Bmag > B_MAG_MAX && continue
                (z < z_lo || z > z_hi) && continue
                is_valley(cs, x, y, z) || continue
                push!(valid_z, z)
                push!(valid_B, Bmag)
            end

            if isempty(valid_z)
                no_root += 1
                push!(grid_z, NaN)
                continue
            end

            if length(valid_z) > 1
                multi_root_xy += 1
                push!(multi_valley_xy_x, x)
                push!(multi_valley_xy_y, y)
                push!(multi_valley_valid_zs, Base.copy(valid_z))
            end
            k = argmax(valid_B)
            z_pick = valid_z[k]
            if length(valid_z) > 1
                push!(multi_valley_chosen_z, z_pick)
            end

            push!(grid_z, z_pick)
            push!(valley_x, x)
            push!(valley_y, y)
            push!(valley_z, z_pick)
        end
    end

    total_in_footprint > 0 && println()

    println("Valid valley points: $(length(valley_z))")
    println("Outside XY footprint mask: $outside_footprint")
    println("No local confinement support: $no_local_conf")
    println("No valid valley root in local confinement interval: $no_root")
    println("XY columns with multiple valid roots in confinement interval: $multi_root_xy")

    return valley_x, valley_y, valley_z, grid_x, grid_y, grid_z,
        multi_valley_xy_x, multi_valley_xy_y, multi_valley_valid_zs, multi_valley_chosen_z
end

function load_confinement(path::AbstractString)
    cx, cy, cz = JLD2.jldopen(path, "r") do f
        f["confinement_x"], f["confinement_y"], f["confinement_z"]
    end
    x = Vector{Float64}(cx)
    y = Vector{Float64}(cy)
    z = Vector{Float64}(cz)
    return x, y, z
end

function main()
    coils = get_coils_from_file(Float64, COIL_FILE, NQUAD)
    conf_x, conf_y, conf_z = load_confinement(CONF_PATH)

    x_min = minimum(conf_x) - XY_BOX_PAD
    x_max = maximum(conf_x) + XY_BOX_PAD
    y_min = minimum(conf_y) - XY_BOX_PAD
    y_max = maximum(conf_y) + XY_BOX_PAD

    idx = build_xy_index(conf_x, conf_y, conf_z)
    fp = build_xy_footprint(conf_x, conf_y)

    if PLOT_MASK_DEBUG
        save_xy_mask_debug_figure(conf_x, conf_y, fp, x_min, x_max, y_min, y_max; nx = NX, ny = NY)
    end
    if MASK_ONLY
        println("MASK_ONLY=true: skipped valley root search.")
        return
    end

    vx, vy, vz, gx, gy, gz, mux, muy, muzs, muchosen = map_valley_surface(
        coils,
        idx,
        fp,
        x_min,
        x_max,
        y_min,
        y_max;
        nx = NX,
        ny = NY,
    )

    fig = Figure(size = (1200, 900))
    ax = Axis3(
        fig[1, 1];
        title = "Magnetic ridge valleys in confinement zone (NaN-masked XY grid)",
        xlabel = "X (m)",
        ylabel = "Y (m)",
        zlabel = "Z (m)",
        aspect = :data,
    )

    for c in coils
        pts = c.rs
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        zs = [p[3] for p in pts]
        push!(xs, xs[1]); push!(ys, ys[1]); push!(zs, zs[1])
        lines!(ax, xs, ys, zs; color = :steelblue, linewidth = 2, alpha = 0.6)
    end

    Rcol = sqrt.(vx .^ 2 .+ vy .^ 2)
    scatter!(ax, vx, vy, vz; markersize = 7, color = Rcol, colormap = :plasma, alpha = 0.9)

    nshow = min(length(conf_x), 8000)
    step = max(1, fld(length(conf_x), nshow))
    scatter!(ax, conf_x[1:step:end], conf_y[1:step:end], conf_z[1:step:end];
        markersize = 2, color = (:gray, 0.18))

    if !isempty(muzs)
        mx_all = Float64[]
        my_all = Float64[]
        mz_all = Float64[]
        for i in eachindex(muzs)
            for z in muzs[i]
                push!(mx_all, mux[i])
                push!(my_all, muy[i])
                push!(mz_all, z)
            end
        end
        !isempty(mz_all) && scatter!(ax, mx_all, my_all, mz_all;
            markersize = 5, color = (:gold, 0.85), strokewidth = 1, strokecolor = :black)
    end

    out_png = joinpath(@__DIR__, "magnetic_ridge_valley_3.png")
    save(out_png, fig)
    println("Saved figure to ", out_png)

    if !isempty(mux)
        fig2 = Figure(size = (900, 800))
        ax2 = Axis(
            fig2[1, 1];
            title = "XY grid cells with multiple valid valley roots (within confinement z-interval)",
            xlabel = "X (m)",
            ylabel = "Y (m)",
            aspect = DataAspect(),
        )
        scatter!(ax2, mux, muy; markersize = 14, color = (:darkorange, 0.95), strokewidth = 1.5, strokecolor = :black)
        out_multi_xy = joinpath(@__DIR__, "magnetic_ridge_valley_3_multi_xy.png")
        save(out_multi_xy, fig2)
        println("Saved multi-root XY map to ", out_multi_xy)
    end

    out_jld2_base = joinpath(@__DIR__, "data", "magnetic_ridge_valley3", "magnetic_ridge_valley_3.jld2")
    mkpath(dirname(out_jld2_base))
    out_jld2 = next_nonconflicting_path(out_jld2_base)
    JLD2.jldopen(out_jld2, "w") do f
        f["valley_x"] = vx
        f["valley_y"] = vy
        f["valley_z"] = vz
        f["grid_x"] = gx
        f["grid_y"] = gy
        f["grid_z"] = gz
        f["multi_valley_xy_x"] = mux
        f["multi_valley_xy_y"] = muy
        f["multi_valley_valid_zs"] = muzs
        f["multi_valley_chosen_z"] = muchosen
        f["meta"] = Dict(
            "surface_type" => "valley",
            "script" => "magnetic_ridge_valley_3.jl",
            "output_jld2_path" => out_jld2,
            "output_jld2_base" => out_jld2_base,
            "nx" => NX,
            "ny" => NY,
            "conf_path" => CONF_PATH,
            "xy_radius_base" => XY_RADIUS_BASE,
            "xy_radius_max" => XY_RADIUS_MAX,
            "min_neighbors" => MIN_NEIGHBORS,
            "z_pad" => Z_PAD,
            "multi_valley_note" =>
                "multi_valley_*: only rows where >1 valid valley root; valid_zs is ragged Vector{Vector}; chosen_z matches primary valley_z/valley_*",
        )
    end
    println("Saved data to ", out_jld2)

    display(fig)
end

main()
