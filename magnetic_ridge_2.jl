# magnetic_ridge_2.jl
# Volumetric search for the magnetic ridge via vertical (z) root-finding on B·∇|B| = 0,
# using bounds from a prior low-resolution ridge dataset.

using LinearAlgebra
using ForwardDiff
using Roots
using JLD2
using GLMakie
using StaticArrays

include("Coil.jl")

# --- Configuration ------------------------------------------------------------

const OLD_DATA_PATH = raw"C:/Users/danie/Stellarator_Isodrasticity/data/magnetic_ridge1/stellarator_data.jld2"
const COIL_FILE = "landreman_paul.json"
const NQUAD = 64

# 2D grid resolution (set MAGNETIC_RIDGE_NX / NY in the environment for higher resolution)
const NX = parse(Int, get(ENV, "MAGNETIC_RIDGE_NX", "64"))
const NY = parse(Int, get(ENV, "MAGNETIC_RIDGE_NY", "64"))

# Padding applied to (x,y) and z search brackets inferred from old ridge data
const BOUND_PAD_XY = 0.05
const BOUND_PAD_Z = 0.05

# Keep ridge points inside the toroidal shell inferred from the prior ridge (excludes coil-side / island ripples)
const CONFINEMENT_PAD_R = 0.03
const CONFINEMENT_PAD_Z = 0.03

# Skip |B| above this to avoid coil-proximity singularities (tune for your geometry)
const B_MAG_MAX = 10.0

# When several ridge roots exist on one vertical line, pick z closest to the JLD2 ridge
# projected by nearest neighbour in (x,y). Only trust that reference if the neighbour
# lies within this horizontal distance (m); beyond it, use RIDGE_REF_FALLBACK.
const REF_XY_MAX_DIST = parse(Float64, get(ENV, "MAGNETIC_RIDGE_REF_XY_MAX", "0.2"))
# If "skip": ambiguous columns with no nearby JLD2 point are dropped. If "max_B": use max |B| there.
const RIDGE_REF_FALLBACK = Symbol(get(ENV, "MAGNETIC_RIDGE_REF_FALLBACK", "skip"))

# JLD2-guided efficiency: only run vertical search if nearest old sample is within this xy distance (m).
# Set MAGNETIC_RIDGE_SIM_XY_MAX=Inf in the environment to scan the full padded xy box.
const SIM_XY_MAX_DIST = let s = get(ENV, "MAGNETIC_RIDGE_SIM_XY_MAX", "")
    if s == "" || s == "auto"
        REF_XY_MAX_DIST + 0.25
    else
        parse(Float64, s)
    end
end

# Minimum half-length of the z bracket around each column's z_ref (m); also scales with JLD2 z spread.
const Z_HALF_MIN = parse(Float64, get(ENV, "MAGNETIC_RIDGE_Z_HALF_MIN", "0.12"))
const Z_HALF_FACTOR = parse(Float64, get(ENV, "MAGNETIC_RIDGE_Z_HALF_FACTOR", "0.55"))

# Finite-difference step along b̂ for ridge (peak) vs well filtering
const RIDGE_CURV_EPS = 1e-5

# --- Type-agnostic Biot–Savart (ForwardDiff-safe) -----------------------------

"""
    B_agnostic(coils, x)

Same Biot–Savart sum as `eval_B`, but the accumulator uses `eltype(x)` so
`ForwardDiff` can propagate `Dual` numbers through the coil loop.
"""
function B_agnostic(cs::Vector{Coil{T}}, x::AbstractVector) where {T}
    B_acc = StaticArrays.zeros(eltype(x), 3)
    for c in cs
        for (ri, Idri) in zip(c.rs, c.Idrs)
            delta_r = SVector(ri[1] - x[1], ri[2] - x[2], ri[3] - x[3])
            dist = three_norm(delta_r)
            B_acc = B_acc + cross(delta_r, Idri) / (dist^3)
        end
    end
    return B_acc
end

# --- Directional derivative Bᵀ J_B B ≈ B · ∇(|B|²/2) scaling; shares roots with B·∇|B|=0

function d_normB_ds(cs::Vector{Coil{T}}, x, y, z) where {T}
    u = SVector{3}(x, y, z)
    B = eval_B(cs, u)
    J = ForwardDiff.jacobian(v -> B_agnostic(cs, v), u)
    return dot(B, J, B)
end

"""True if the zero of the longitudinal derivative is a ridge (peak), not a well."""
function is_ridge(cs::Vector{Coil{T}}, x, y, z) where {T}
    u = SVector{3}(x, y, z)
    B = eval_B(cs, u)
    nb = norm(B)
    nb < eps(Float64) * 1e6 && return false
    b_hat = B / nb
    ε = RIDGE_CURV_EPS
    slope_now = d_normB_ds(cs, x, y, z)
    slope_fwd = d_normB_ds(
        cs,
        x + ε * b_hat[1],
        y + ε * b_hat[2],
        z + ε * b_hat[3],
    )
    curvature = (slope_fwd - slope_now) / ε
    return curvature < 0.0
end

# --- Load prior ridge to bound the search domain --------------------------------

function load_ridge_bounds(path::AbstractString)
    ridge_x, ridge_y, ridge_z = JLD2.jldopen(path, "r") do f
        f["ridge_x"], f["ridge_y"], f["ridge_z"]
    end
    rx = Vector{Float64}(ridge_x)
    ry = Vector{Float64}(ridge_y)
    rz = Vector{Float64}(ridge_z)
    @assert length(rx) == length(ry) == length(rz)

    x_min = minimum(rx) - BOUND_PAD_XY
    x_max = maximum(rx) + BOUND_PAD_XY
    y_min = minimum(ry) - BOUND_PAD_XY
    y_max = maximum(ry) + BOUND_PAD_XY
    z_min = minimum(rz) - BOUND_PAD_Z
    z_max = maximum(rz) + BOUND_PAD_Z

    R_old = hypot.(rx, ry)
    r_conf_min = minimum(R_old) - CONFINEMENT_PAD_R
    r_conf_max = maximum(R_old) + CONFINEMENT_PAD_R
    z_conf_min = minimum(rz) - CONFINEMENT_PAD_Z
    z_conf_max = maximum(rz) + CONFINEMENT_PAD_Z

    rz_span = maximum(rz) - minimum(rz)
    z_bracket_half = max(Z_HALF_FACTOR * rz_span + BOUND_PAD_Z, Z_HALF_MIN)

    return rx, ry, rz, x_min, x_max, y_min, y_max, z_min, z_max, r_conf_min, r_conf_max, z_conf_min, z_conf_max, z_bracket_half
end

@inline function in_confinement(x, y, z, r_conf_min, r_conf_max, z_conf_min, z_conf_max)
    R = hypot(x, y)
    return (R >= r_conf_min) && (R <= r_conf_max) && (z >= z_conf_min) && (z <= z_conf_max)
end

"""Nearest JLD2 sample in the xy plane; returns (horizontal distance, z at that sample)."""
function nearest_xy_zref(
    rx::Vector{Float64},
    ry::Vector{Float64},
    rz::Vector{Float64},
    x,
    y,
)
    best_d2 = Inf
    best_i = 1
    @inbounds for i in eachindex(rx)
        dx = rx[i] - x
        dy = ry[i] - y
        d2 = muladd(dx, dx, dy * dy)
        if d2 < best_d2
            best_d2 = d2
            best_i = i
        end
    end
    return sqrt(best_d2), rz[best_i]
end

# --- Main volumetric scan -------------------------------------------------------

function map_ridge_surface(
    cs::Vector{Coil{T}},
    x_min,
    x_max,
    y_min,
    y_max,
    z_min,
    z_max,
    r_conf_min,
    r_conf_max,
    z_conf_min,
    z_conf_max,
    rx_old::Vector{Float64},
    ry_old::Vector{Float64},
    rz_old::Vector{Float64},
    z_bracket_half::Float64;
    nx::Int = NX,
    ny::Int = NY,
    ref_xy_max_dist::Float64 = REF_XY_MAX_DIST,
    ref_fallback::Symbol = RIDGE_REF_FALLBACK,
    sim_xy_max_dist::Float64 = SIM_XY_MAX_DIST,
) where {T}
    ref_fallback ∈ (:skip, :max_B) || throw(ArgumentError("ref_fallback must be :skip or :max_B"))

    x_range = range(x_min, x_max; length = nx)
    y_range = range(y_min, y_max; length = ny)

    ridge_points_x = Float64[]
    ridge_points_y = Float64[]
    ridge_points_z = Float64[]

    n_ambiguous_skipped = 0
    n_skipped_xy = 0
    zs_buf = Float64[]  # reuse buffers to avoid per-column allocations
    Bs_buf = Float64[]

    println("Volumetric ridge scan: nx=$nx ny=$ny, global z ∈ [$z_min, $z_max]")
    println(
        "JLD2-guided: per-column z search in [z_ref±$z_bracket_half] clipped to global z bounds",
    )
    println(
        "xy gate: skip column if nearest JLD2 sample has d_xy > $sim_xy_max_dist (set MAGNETIC_RIDGE_SIM_XY_MAX=Inf to disable)",
    )
    println(
        "Confinement filter: R ∈ [$r_conf_min, $r_conf_max], z ∈ [$z_conf_min, $z_conf_max]",
    )
    println(
        "Multi-root tie-break: nearest JLD2 in xy → min |z−z_ref| if d_xy ≤ $ref_xy_max_dist; else fallback=$ref_fallback",
    )

    for x in x_range
        for y in y_range
            d_xy, z_ref = nearest_xy_zref(rx_old, ry_old, rz_old, x, y)
            if d_xy > sim_xy_max_dist
                n_skipped_xy += 1
                continue
            end

            z_lo = max(z_min, z_ref - z_bracket_half)
            z_hi = min(z_max, z_ref + z_bracket_half)
            if z_lo >= z_hi
                @warn "degenerate z bracket" z_lo z_hi z_ref x y
                continue
            end

            f_z(z) = d_normB_ds(cs, x, y, z)
            possible_zs = try
                find_zeros(f_z, z_lo, z_hi)
            catch e
                @warn "find_zeros failed" exception = e x y
                continue
            end

            empty!(zs_buf)
            empty!(Bs_buf)
            for z in possible_zs
                B_mag = norm(eval_B(cs, SVector{3}(x, y, z)))
                B_mag > B_MAG_MAX && continue
                in_confinement(x, y, z, r_conf_min, r_conf_max, z_conf_min, z_conf_max) || continue
                is_ridge(cs, x, y, z) || continue
                push!(zs_buf, z)
                push!(Bs_buf, B_mag)
            end
            isempty(zs_buf) && continue

            if length(zs_buf) == 1
                push!(ridge_points_x, x)
                push!(ridge_points_y, y)
                push!(ridge_points_z, zs_buf[1])
                continue
            end

            if d_xy <= ref_xy_max_dist
                k = 1
                err_best = abs(zs_buf[1] - z_ref)
                @inbounds for i in 2:length(zs_buf)
                    e = abs(zs_buf[i] - z_ref)
                    if e < err_best
                        err_best = e
                        k = i
                    end
                end
                push!(ridge_points_x, x)
                push!(ridge_points_y, y)
                push!(ridge_points_z, zs_buf[k])
            elseif ref_fallback === :max_B
                k = argmax(Bs_buf)
                push!(ridge_points_x, x)
                push!(ridge_points_y, y)
                push!(ridge_points_z, zs_buf[k])
            else
                n_ambiguous_skipped += 1
            end
        end
    end

    println(
        "Found $(length(ridge_points_z)) ridge points. Columns skipped (xy gate): $n_skipped_xy. Ambiguous multi-root (no JLD2 within ref radius, fallback=skip): $n_ambiguous_skipped",
    )
    return ridge_points_x, ridge_points_y, ridge_points_z
end

# --- Run ------------------------------------------------------------------------

coils = get_coils_from_file(Float64, COIL_FILE, NQUAD)

ridge_x_old, ridge_y_old, ridge_z_old, x_min, x_max, y_min, y_max, z_min, z_max, r_conf_min, r_conf_max, z_conf_min, z_conf_max, z_bracket_half =
    load_ridge_bounds(OLD_DATA_PATH)

println(
    "Bounds from old data (padded): x ∈ [$x_min, $x_max], y ∈ [$y_min, $y_max], z search [$z_min, $z_max]",
)
println("JLD2 z span → bracket half-width z_bracket_half = $z_bracket_half m")

rx, ry, rz = map_ridge_surface(
    coils,
    x_min,
    x_max,
    y_min,
    y_max,
    z_min,
    z_max,
    r_conf_min,
    r_conf_max,
    z_conf_min,
    z_conf_max,
    ridge_x_old,
    ridge_y_old,
    ridge_z_old,
    z_bracket_half,
)

# --- Plot: coils + single-valued ridge surface ----------------------------------

fig = Figure(size = (1200, 900))
ax = Axis3(
    fig[1, 1];
    title = "Landreman–Paul: coils and ridge (nearest JLD2 z, multi-root tie-break)",
    xlabel = "X (m)",
    ylabel = "Y (m)",
    zlabel = "Z (m)",
    aspect = :data,
    elevation = π / 6,
    azimuth = π / 4,
)

for c in coils
    pts = c.rs
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    zs = [p[3] for p in pts]
    push!(xs, xs[1])
    push!(ys, ys[1])
    push!(zs, zs[1])
    lines!(ax, xs, ys, zs; color = :blue, linewidth = 2, alpha = 0.6)
end

R_colors = sqrt.(rx .^ 2 .+ ry .^ 2)
meshscatter!(
    ax,
    rx,
    ry,
    rz;
    markersize = 0.02,
    color = R_colors,
    colormap = :plasma,
    shading = true,
)

out_png = joinpath(@__DIR__, "magnetic_ridge_2.png")
save(out_png, fig)
println("Saved figure to ", out_png)

display(fig)
