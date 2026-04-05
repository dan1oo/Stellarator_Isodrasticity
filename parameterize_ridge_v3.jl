# parameterize_ridge_v3.jl
#
# Global polynomial–Fourier least-squares fit:
#   z(R, φ) = Σ_{m=0}^M Σ_{n=0}^N R^m [ A_{m,n} cos(nφ) + B_{m,n} sin(nφ) ]
# For n = 0 only the cos(0·φ) = 1 term is used (no sin(0·φ) column).
#
# Loads ridge grid from JLD2; outputs coefficient vector `c` (column order matches design matrix).

using LinearAlgebra
using JLD2
using GLMakie
using Printf

# --- paths & model order -------------------------------------------------------

const DATA_PATH = get(
    ENV,
    "RIDGE_FIT_JLD2",
    raw"C:/Users/danie/Stellarator_Isodrasticity/data/magnetic_ridge3/magnetic_ridge_3.jld2",
)

const M_RADIAL = parse(Int, get(ENV, "RIDGE_FIT_M", "2"))   # max power of R
const N_TOROIDAL = parse(Int, get(ENV, "RIDGE_FIT_N", "3")) # max Fourier mode n

const SURF_NR = parse(Int, get(ENV, "RIDGE_FIT_SURF_NR", "80"))
const SURF_NPHI = parse(Int, get(ENV, "RIDGE_FIT_SURF_NPHI", "120"))

# --- design matrix -------------------------------------------------------------

"""Number of columns: (M+1) for n=0, plus (M+1)*2 for each n=1..N."""
function n_coefficients(M::Int, N::Int)
    (M + 1) + N * (M + 1) * 2
end

"""
Build design matrix A[i, k] for point i with (R_i, φ_i).
Column order: for m=0:M, n=0: R^m; for n=1:N, m=0:M: R^m cos(nφ), R^m sin(nφ).
"""
function design_matrix(R::AbstractVector{Float64}, phi::AbstractVector{Float64}, M::Int, N::Int)
    npts = length(R)
    @assert length(phi) == npts
    ncols = n_coefficients(M, N)
    A = zeros(npts, ncols)
    col = 1
    @inbounds for m in 0:M
        A[:, col] .= R .^ m
        col += 1
    end
    for n in 1:N
        @inbounds for m in 0:M
            A[:, col] .= (R .^ m) .* cos.(n .* phi)
            col += 1
            A[:, col] .= (R .^ m) .* sin.(n .* phi)
            col += 1
        end
    end
    @assert col == ncols + 1
    return A
end

"""Predict z from coefficient vector `c` (same column order as `design_matrix`)."""
function predict_z(R::AbstractArray{Float64}, phi::AbstractArray{Float64}, c::AbstractVector{Float64}, M::Int, N::Int)
    z = zeros(Float64, size(R))
    k = 1
    @inbounds for m in 0:M
        z .+= c[k] .* (R .^ m)
        k += 1
    end
    for n in 1:N
        @inbounds for m in 0:M
            z .+= c[k] .* ((R .^ m) .* cos.(n .* phi))
            k += 1
            z .+= c[k] .* ((R .^ m) .* sin.(n .* phi))
            k += 1
        end
    end
    return z
end

function coefficient_labels(M::Int, N::Int)
    labs = String[]
    for m in 0:M
        push!(labs, "R^$m (n=0)")
    end
    for n in 1:N
        for m in 0:M
            push!(labs, "R^$m cos($(n)φ)")
            push!(labs, "R^$m sin($(n)φ)")
        end
    end
    return labs
end

# --- main ----------------------------------------------------------------------

function main()
    new_rx, new_ry, new_rz = JLD2.jldopen(DATA_PATH, "r") do f
        if haskey(f, "new_rx") && haskey(f, "new_ry") && haskey(f, "new_rz")
            Vector{Float64}(f["new_rx"]),
            Vector{Float64}(f["new_ry"]),
            Vector{Float64}(f["new_rz"])
        elseif haskey(f, "ridge_x") && haskey(f, "ridge_y") && haskey(f, "ridge_z")
            println("Note: using ridge_x / ridge_y / ridge_z (new_rx etc. not found)")
            Vector{Float64}(f["ridge_x"]),
            Vector{Float64}(f["ridge_y"]),
            Vector{Float64}(f["ridge_z"])
        else
            error(
                "Need new_rx,new_ry,new_rz or ridge_x,ridge_y,ridge_z in $(DATA_PATH)",
            )
        end
    end

    npts = length(new_rz)
    @assert length(new_rx) == npts && length(new_ry) == npts

    R = hypot.(new_rx, new_ry)
    phi = atan.(new_ry, new_rx)

    M = M_RADIAL
    N = N_TOROIDAL
    A = design_matrix(R, phi, M, N)
    ncoef = size(A, 2)
    println(@sprintf("Data points: %d", npts))
    println(@sprintf("Model: M=%d (radial), N=%d (toroidal Fourier)", M, N))
    println(@sprintf("Total coefficients: %d", ncoef))

    c = A \ new_rz
    z_hat = A * c
    rmse = sqrt(sum(abs2, new_rz .- z_hat) / npts)
    println(@sprintf("RMSE (data vs fit): %.6g", rmse))

    # --- GLMakie: data + analytical surface -----------------------------------
    fig = Figure(size = (1100, 900))
    ax = Axis3(
        fig[1, 1];
        title = "Ridge fit: z(R,φ) polynomial–Fourier (M=$M, N=$N), RMSE=$(round(rmse, sigdigits=5))",
        xlabel = "X (m)",
        ylabel = "Y (m)",
        zlabel = "Z (m)",
        aspect = :data,
    )

    Rcol = R
    meshscatter!(ax, new_rx, new_ry, new_rz; markersize = 0.025, color = Rcol, colormap = :plasma, shading = true)

    Rmin = minimum(R)
    Rmax = maximum(R)
    # φ covers full circle for a smooth toroidal sheet
    Rg = range(Rmin, Rmax; length = SURF_NR)
    phig = range(-Float64(π), Float64(π); length = SURF_NPHI)
    Rmat = [r for r in Rg, _ in phig]
    phimat = [p for _ in Rg, p in phig]
    Zmat = predict_z(Rmat, phimat, c, M, N)
    Xmat = Rmat .* cos.(phimat)
    Ymat = Rmat .* sin.(phimat)

    surface!(ax, Xmat, Ymat, Zmat; colormap = :viridis, alpha = 0.7, shading = true)

    out_png = joinpath(@__DIR__, "parameterize_ridge_v3.png")
    save(out_png, fig)
    println("Saved figure to ", out_png)

    # Optional: save coefficients for downstream normals / gradients
    out_coef = joinpath(@__DIR__, "data", "magnetic_ridge3", "ridge_fit_coeffs_v3.jld2")
    mkpath(dirname(out_coef))
    JLD2.jldopen(out_coef, "w") do f
        f["c"] = c
        f["M"] = M
        f["N"] = N
        f["rmse"] = rmse
        f["n_coefficients"] = ncoef
        f["column_labels"] = coefficient_labels(M, N)
        f["source_jld2"] = DATA_PATH
        f["model"] =
            "z(R,phi)=sum_m sum_n R^m (A_{m,n} cos(n phi) + B_{m,n} sin(n phi)); n=0: cos only"
    end
    println("Saved coefficients to ", out_coef)

    display(fig)
end

main()
