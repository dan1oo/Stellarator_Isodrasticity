using LinearAlgebra
using ForwardDiff
using Roots



include("Coil.jl")
coil_file = "landreman_paul.json"
Nquad = 64 
coils = get_coils_from_file(Float64, coil_file, Nquad)

# Now your x_top = rx[top_idx] logic will work perfectly!
# using GLMakie # Uncomment for plotting later

# 1. Define the exact directional derivative function
# This calculates B ⋅ ∇|B| (or technically B ⋅ J ⋅ B)
function d_normB_ds(coils, x::T, y::T, z::T) where {T}
    u = SA[x, y, z]
    
    # Evaluate B-field
    B = eval_B(coils, u)
    
    # Calculate Jacobian using ForwardDiff
    # Note: Ensure eval_B is type-stable as you noted!
    J = ForwardDiff.jacobian(pos -> eval_B(coils, pos), u)
    
    # B' * J * B = B ⋅ ∇(|B|²/2). Root is the same as ∇|B| = 0
    return dot(B, J, B)
end

# 2. Define a function to check the curvature (Second Derivative)
# We only want Peaks (ridges), not Valleys (wells)
function is_ridge(coils, x, y, z)
    # To find if it's a peak, the derivative of our slope function along the field line 
    # must be negative. We can use a small finite difference step along B for speed here, 
    # or another ForwardDiff pass. Let's use a quick finite step along B.
    
    u = SA[x, y, z]
    B = eval_B(coils, u)
    b_hat = B / norm(B)
    eps = 1e-5
    
    # Evaluate slope slightly forward along the field line
    slope_now = d_normB_ds(coils, x, y, z)
    slope_fwd = d_normB_ds(coils, x + eps*b_hat[1], y + eps*b_hat[2], z + eps*b_hat[3])
    
    curvature = (slope_fwd - slope_now) / eps
    
    # Negative curvature means it's a local maximum (Ridge)
    return curvature < 0.0
end

# 3. Main Scanning Function
function map_ridge_surface(coils)
    # Define your scanning bounds based on the Landreman-Paul geometry
    # Adjust these ranges based on your specific stellarator size
    x_range = range(-2.0, 2.0, length=100)
    y_range = range(-2.0, 2.0, length=100)
    
    # The vertical bounds of the confinement space
    # This prevents finding roots inside the physical coils
    z_min, z_max = -1.0, 1.0 
    
    # Storage for the valid ridge points
    ridge_points_x = Float64[]
    ridge_points_y = Float64[]
    ridge_points_z = Float64[]
    
    println("Starting full volumetric ridge scan...")
    
    for x in x_range
        for y in y_range
            # Skip the central hole of the torus to save time
            R = sqrt(x^2 + y^2)
            if R < 0.5 || R > 2.0 
                continue 
            end
            
            # Create a 1D function for Roots.jl where only z varies
            f_z(z) = d_normB_ds(coils, x, y, z)
            
            # Find ALL roots in the bracket [z_min, z_max]
            # find_zeros is robust and will return an array of all z values where f_z(z) == 0
            possible_zs = find_zeros(f_z, z_min, z_max)
            
            for z in possible_zs
                # Filter 1: Is it actually inside the plasma volume?
                # (You can add a check here to see if |B| is within a sane range)
                B_mag = norm(eval_B(coils, SA[x,y,z]))
                if B_mag > 10.0 # Arbitrary high cutoff to avoid coil singularities
                    continue
                end
                
                # Filter 2: Is it a maximum (Ridge) and not a minimum (Valley)?
                if is_ridge(coils, x, y, z)
                    push!(ridge_points_x, x)
                    push!(ridge_points_y, y)
                    push!(ridge_points_z, z)
                end
            end
        end
    end
    
    println("Found $(length(ridge_points_z)) ridge points.")
    return ridge_points_x, ridge_points_y, ridge_points_z
end

using GLMakie
using LinearAlgebra

# 1. Run your mapping function to get the coordinates
rx, ry, rz = map_ridge_surface(coils)

# 2. Set up the 3D Figure
fig = Figure(size = (1000, 1000))
ax = Axis3(fig[1, 1], 
    title = "3D Magnetic Ridge Surface",
    xlabel = "X (m)", 
    ylabel = "Y (m)", 
    zlabel = "Z (m)",
    aspect = :data,      # Keeps the torus from looking stretched
    elevation = pi/6,    # Starting camera angle
    azimuth = pi/4
)

# 3. Calculate the radial distance to color the points 
# This helps visualize the depth of the ridge surface
R_colors = sqrt.(rx.^2 .+ ry.^2)

# 4. Plot the ridge points
# Using meshscatter provides better 3D lighting/shading than a flat scatter
meshscatter!(ax, rx, ry, rz, 
    markersize = 0.02,   # Adjust based on the density of your points
    color = R_colors, 
    colormap = :plasma,  # A good high-contrast colormap
    shading = FastShading
)

# 5. Display the interactive window
display(fig)