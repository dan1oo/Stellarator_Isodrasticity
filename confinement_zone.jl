# confinement_zone.jl
#
# Point cloud along magnetic field lines using Coil.follow_B_line (same path integration as
# elsewhere in this repo). Initial conditions match Magnetic_Ridge.jl: (x,y,z) = (r cos φ, r sin φ, 0).
# Valid lines (|B| and bounding sphere checks) are saved to JLD2 for use as a confinement reference.
#
# Load with: JLD2.jldopen(path, "r") do f; f["confinement_x"], ...; end

using StaticArrays
using LinearAlgebra
using JLD2
using Printf
using Dates
using CairoMakie

include("Coil.jl")

# --- Configuration ------------------------------------------------------------

const COIL_FILE = "landreman_paul.json"
const NQUAD = 64

# Arc length along B (same parameter as Magnetic_Ridge.jl run_poincare(..., L_max))
const L_MAX = parse(Float64, get(ENV, "CONFINEMENT_L_MAX", "20000.0"))

# Samples every SAVE_DS meters of arc length (passed to follow_B_line as saveat)
const SAVE_DS = parse(Float64, get(ENV, "CONFINEMENT_SAVE_DS", "120.0"))

# Same ODE solver as Magnetic_Ridge.jl (field-line ODE only; no ridge/Poincaré callbacks)
const LINE_SOLVER = Vern9()

const B_MAG_MIN = 1e-5
const B_MAG_MAX = 10.0
const R_SPHERE_MAX = 3.5

const OUT_DIR = joinpath(@__DIR__, "data", "confinement_zone")
const OUT_FILE = joinpath(OUT_DIR, "confinement_field_lines.jld2")
const OUT_PNG = joinpath(OUT_DIR, "confinement_field_lines.png")

const DO_PLOT = parse(Bool, get(ENV, "CONFINEMENT_PLOT", "true"))

# If true: load confinement_field_lines.jld2 and redraw OUT_PNG only (no integration)
const PLOT_ONLY = parse(Bool, get(ENV, "CONFINEMENT_PLOT_ONLY", "false"))

# Same five φ planes and r ranges as Magnetic_Ridge.jl; higher nr for denser starts
const SCAN_PLANES = [
    (0.0, 1.15, 1.3, 28),
    (Float64(π / 5), 1.1, 1.25, 28),
    (Float64(2π / 5), 0.65, 1.2, 28),
    (Float64(3π / 5), 0.625, 1.2, 28),
    (Float64(4π / 5), 1.1, 1.25, 28),
]

# --- Integration success (no SciMLBase import required) -------------------------

function integration_succeeded(sol)
    r = sol.retcode
    r === :Success && return true
    sr = string(r)
    return sr == "Success" || occursin("Success", sr)
end

function trajectory_valid(coils, sol)
    integration_succeeded(sol) || return false
    for u in sol.u
        any(!isfinite, u) && return false
        norm(u) > R_SPHERE_MAX && return false
        Bm = norm(eval_B(coils, u))
        (Bm < B_MAG_MIN || Bm > B_MAG_MAX) && return false
    end
    return true
end

"""Initial position on z = 0, matching Magnetic_Ridge.jl `run_poincare`."""
@inline function initial_xyz(r, phi)
    x = r * cos(phi)
    y = r * sin(phi)
    SA[x, y, 0.0]
end

function save_confinement_figure(coils, conf_x, conf_y, conf_z, path::AbstractString)
    fig = Figure(size = (1200, 900))
    ax = Axis3(
        fig[1, 1];
        title = "Landreman–Paul: confinement samples (scatter) and coils",
        aspect = :data,
        xlabel = "X (m)",
        ylabel = "Y (m)",
        zlabel = "Z (m)",
        elevation = 0.35f0 * Float32(π),
        azimuth = 0.5f0 * Float32(π),
    )
    for c in coils
        pts = c.rs
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        zs = [p[3] for p in pts]
        push!(xs, xs[1])
        push!(ys, ys[1])
        push!(zs, zs[1])
        lines!(ax, xs, ys, zs; color = :steelblue, linewidth = 2, alpha = 0.65)
    end
    if !isempty(conf_x)
        Rcol = sqrt.(conf_x .^ 2 .+ conf_y .^ 2)
        scatter!(
            ax,
            conf_x,
            conf_y,
            conf_z;
            color = Rcol,
            colormap = :plasma,
            markersize = 5,
            alpha = 0.55,
        )
    end
    save(path, fig)
    println("Saved figure to ", path)
end

function replot_from_jld2()
    isfile(OUT_FILE) || error("Missing $(OUT_FILE); run without CONFINEMENT_PLOT_ONLY first.")
    coils = get_coils_from_file(Float64, COIL_FILE, NQUAD)
    conf_x, conf_y, conf_z = JLD2.jldopen(OUT_FILE, "r") do f
        Vector{Float64}(f["confinement_x"]),
        Vector{Float64}(f["confinement_y"]),
        Vector{Float64}(f["confinement_z"])
    end
    println("Replotting from JLD2: ", length(conf_z), " points")
    save_confinement_figure(coils, conf_x, conf_y, conf_z, OUT_PNG)
end

# --- Main ---------------------------------------------------------------------

function main()
    println("confinement_zone.jl — field lines via Coil.follow_B_line (Magnetic_Ridge ICs)")
    println("  L_MAX = $L_MAX, SAVE_DS = $SAVE_DS, solver = $(typeof(LINE_SOLVER))")
    flush(stdout)

    coils = get_coils_from_file(Float64, COIL_FILE, NQUAD)
    saveat_vec = collect(0.0:SAVE_DS:L_MAX)

    conf_x = Float64[]
    conf_y = Float64[]
    conf_z = Float64[]

    n_try = 0
    n_ok = 0
    n_bad = 0

    t_start = time()

    for (phi, rmin, rmax, nr) in SCAN_PLANES
        rs = range(rmin, rmax; length = nr)
        @inbounds for r in rs
            n_try += 1
            u0 = initial_xyz(r, phi)
            sol = follow_B_line(u0, L_MAX, coils; method = LINE_SOLVER, saveat = saveat_vec)
            if trajectory_valid(coils, sol)
                n_ok += 1
                for u in sol.u
                    push!(conf_x, u[1])
                    push!(conf_y, u[2])
                    push!(conf_z, u[3])
                end
            else
                n_bad += 1
            end
        end
        @printf("Finished phi = %.4f rad: accumulated %d valid lines so far\n", phi, n_ok)
    end

    elapsed = time() - t_start
    println()
    println(@sprintf("Lines attempted: %d, valid: %d, rejected: %d", n_try, n_ok, n_bad))
    println(@sprintf("Total points in cloud: %d (elapsed %.1f s)", length(conf_z), elapsed))

    mkpath(OUT_DIR)
    JLD2.jldopen(OUT_FILE, "w") do f
        f["confinement_x"] = conf_x
        f["confinement_y"] = conf_y
        f["confinement_z"] = conf_z
        f["meta"] = Dict{String, Any}(
            "description" =>
                "Points along B via Coil.follow_B_line; ICs (r,phi,z=0) as Magnetic_Ridge.jl",
            "integration" => "Coil.follow_B_line",
            "solver" => string(typeof(LINE_SOLVER)),
            "coil_file" => COIL_FILE,
            "Nquad" => NQUAD,
            "L_max" => L_MAX,
            "save_ds" => SAVE_DS,
            "b_mag_min" => B_MAG_MIN,
            "b_mag_max" => B_MAG_MAX,
            "r_sphere_max" => R_SPHERE_MAX,
            "scan_planes" => collect(SCAN_PLANES),
            "n_lines_attempted" => n_try,
            "n_lines_valid" => n_ok,
            "n_lines_rejected" => n_bad,
            "created_utc" => string(Dates.now(Dates.UTC)),
            "figure_png" => OUT_PNG,
        )
    end
    println("Wrote ", OUT_FILE)

    if DO_PLOT && !isempty(conf_z)
        save_confinement_figure(coils, conf_x, conf_y, conf_z, OUT_PNG)
    elseif DO_PLOT
        @warn "No confinement points to plot; skipping figure"
    end
end

if PLOT_ONLY
    replot_from_jld2()
else
    main()
end
