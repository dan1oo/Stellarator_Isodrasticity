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
function b_field_ode(u, p, t)
    B = eval_B(coils, u)
    B / norm(B)
end
function run_poincare(r, phi, L_max)

    x = r * cos(phi)
    y = r * sin(phi)
    z = 0
    u0 = SA[x,y,z]

    # Store Variables
    cx, cy, cz = Float64[], Float64[], Float64[]
    
    
    
    # Solve
    prob = ODEProblem(b_field_ode, u0, (0.0, L_max))
    sol = solve(prob, Vern9())
    cx = [pos[1] for pos in sol.u]
    cy = [pos[2] for pos in sol.u]
    cz = [pos[3] for pos in sol.u]
    return cx, cy, cz
end


# Main 

coil_file = "landreman_paul.json"
Nquad = 64 
coils = get_coils_from_file(Float64, coil_file, Nquad)

#Initialize the Master Vectors
c_x, c_y, c_z = Float64[], Float64[], Float64[]

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
# Define your search space
multi_r1 = range(1.15, 1.3, length=10)
phi = 0


# 2. Run the loop across the radial range
for r in multi_r1
    
    println("Tracing field lines for r = ", round(r, digits=3))

    # Run the simulation for this specific radius
    # We unpack the 6 returned vectors into temporary variables
    cx, cy, cz = run_poincare(r, 0, 20000.0)

    # 3. Append the results to your master lists
    append!(c_x, cx)
    append!(c_y, cy)
    append!(c_z, cz)

    
end

# Define your search space
multi_r2 = range(1.1, 1.25, length=10)
phi = pi/5.0


# 2. Run the loop across the radial range
for r in multi_r2
    
    println("Tracing field lines for r = ", round(r, digits=3))

    # Run the simulation for this specific radius
    # We unpack the 6 returned vectors into temporary variables
    cx, cy, cz = run_poincare(r, pi/5.0, 20000.0)

    # 3. Append the results to your master lists
    append!(c_x, cx)
    append!(c_y, cy)
    append!(c_z, cz)
end

# Define your search space
multi_r3 = range(0.65, 1.2, length=10)
phi = 2.0*pi/5.0


# 2. Run the loop across the radial range
for r in multi_r3
    
    println("Tracing field lines for r = ", round(r, digits=3))

    # Run the simulation for this specific radius
    # We unpack the 6 returned vectors into temporary variables
    cx, cy, cz = run_poincare(r, 2.0*pi/5.0, 20000.0)

    # 3. Append the results to your master lists
    append!(c_x, cx)
    append!(c_y, cy)
    append!(c_z, cz)
end

# Define your search space
multi_r4 = range(0.625, 1.2, length=10)
phi = 3.0*pi/5.0


# 2. Run the loop across the radial range
for r in multi_r4
    
    println("Tracing field lines for r = ", round(r, digits=3))

    # Run the simulation for this specific radius
    # We unpack the 6 returned vectors into temporary variables
    cx, cy, cz = run_poincare(r, 3.0*pi/5.0, 20000.0)

    # 3. Append the results to your master lists
    append!(c_x, cx)
    append!(c_y, cy)
    append!(c_z, cz)
end


# Define your search space
multi_r5 = range(1.1, 1.25, length=10)
phi = 4.0*pi/5.0
# 2. Run the loop across the radial range
for r in multi_r5
    
    println("Tracing field lines for r = ", round(r, digits=3))

    # Run the simulation for this specific radius
    # We unpack the 6 returned vectors into temporary variables
    cx, cy, cz = run_poincare(r, 4.0*pi/5.0, 20000.0)

    # 3. Append the results to your master lists
    append!(c_x, cx)
    append!(c_y, cy)
    append!(c_z, cz)
end





#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

# Initialize 3D Scene
fig = Figure(size = (1200, 900))
# Axis3 provides the 3D container for our coils and field lines
ax = Axis3(fig[1, 1], title = "3D Landreman-Paul Magnetic Topology", aspect = :data)

# Plot the Coils
# Each coil is an array of quadrature points 'rs'
for (i, c) in enumerate(coils)
    # Extract X, Y, Z coordinates for the wire path
    pts = c.rs
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    zs = [p[3] for p in pts]
    
    # Close the loop (connect last point to first)
    push!(xs, xs[1]); push!(ys, ys[1]); push!(zs, zs[1])
    
    # Draw the physical wire
    lines!(ax, xs, ys, zs, color = :blue, linewidth = 2, alpha = 0.6)
end

# Plot Poincare surface
scatter!(ax, c_x, c_y, c_z, color = :red, markersize = 4)



display(fig)