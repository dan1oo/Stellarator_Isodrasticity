using OrdinaryDiffEq
using CairoMakie
using StaticArrays
using LinearAlgebra
using ForwardDiff


include("Coil.jl")


# Poincare Callback ( based on initial phi, x,y,z)
function create_poincare_cb(phi, px, py, pz)

    #Condition when integration path crosses the initial azimuthal plane

    function condition(u, t, integrator)
        return -u[1] * sin(phi) + u[2] * cos(phi)
    end

    # Action
    function affect!(integrator)
        
        push!(px, integrator.u[1])
        push!(py, integrator.u[2])
        push!(pz, integrator.u[3])
    
    end

    return ContinuousCallback(condition, affect!)
end

# Magnetic Ridge Callback 
function create_ridge_cb(coils, rx, ry, rz)
    function condition(u, t, integrator)
        # Use a small step to find d|B|/ds
        eps = 1e-6
        B_vec = eval_B(coils, u)
        b_hat = B_vec / norm(B_vec)
        B_now = norm(B_vec)
        B_forward = norm(eval_B(coils, u + eps * b_hat))
        return (B_forward - B_now) / eps
    end
    function ridge_condition(u, t, integrator)
        # 1. Standard B-field for the vector part (Float64)
        B = eval_B(coils, u)
        
        # 2. Define a local, type-agnostic B-field calculator
        # This bypasses the strict eval_B(::Vector{Coil{T}}, ::Vector{T}) signature 
        # that causes the MethodError [cite: 15]
        function B_agnostic(x)
            # Initialize accumulator with the type of x (Float64 or Dual)
            B_acc = StaticArrays.zeros(eltype(x), 3)
            
            for c in coils
                # Re-implement the loop from Coil.jl line 133 [cite: 9]
                # but allow x to be any type (like ForwardDiff.Dual)
                for (ri, Idri) in zip(c.rs, c.Idrs)
                    delta_r = ri - x
                    
                    # Use the three_norm logic from Coil.jl [cite: 1, 9]
                    # It is single-argument, so it won't trigger a T-mismatch error
                    dist = three_norm(delta_r)
                    
                    # Biot-Savart kernel [cite: 9]
                    B_acc += cross(delta_r, Idri) / (dist^3)
                end
            end
            return B_acc
        end

        # 3. Calculate the Jacobian using the agnostic wrapper
        # ForwardDiff can now propagate its Dual numbers through the loop
        J = ForwardDiff.jacobian(B_agnostic, u)
        
        # 4. Return the ridge condition (Bi * ∂iBj * Bj)
        return dot(B, J, B)
    end

    function save_peak!(integrator)
       
       
        push!(rx, integrator.u[1])
        push!(ry, integrator.u[2])
        push!(rz, integrator.u[3])
        
    end

    return ContinuousCallback(ridge_condition, nothing, save_peak!)
end

#  Define the ODE: dx/ds = B / |B|v- Magnetic field ODE
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
    px, py, pz = Float64[], Float64[], Float64[]
    rx, ry, rz = Float64[], Float64[], Float64[]
    
    # Get callback objects
    cb_p = create_poincare_cb(phi, px, py, pz)
    cb_r = create_ridge_cb(coils, rx, ry, rz)
    
    cb_total = CallbackSet(cb_p, cb_r)
    
    # Solve
    prob = ODEProblem(b_field_ode, u0, (0.0, L_max))
    solve(prob, Vern9(), callback=cb_total)
    
    return px, py, pz, rx, ry, rz
end


# Main 

coil_file = "landreman_paul.json"
Nquad = 64 
coils = get_coils_from_file(Float64, coil_file, Nquad)

#Initialize the Master Vectors
poincare_x, poincare_y, poincare_z = Float64[], Float64[], Float64[]
ridge_x, ridge_y, ridge_z = Float64[], Float64[], Float64[]


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
    px, py, pz, rx, ry, rz = run_poincare(r, 0, 20000.0)

    # 3. Append the results to your master lists
    append!(poincare_x, px)
    append!(poincare_y, py)
    append!(poincare_z, pz)

    append!(ridge_x, rx)
    append!(ridge_y, ry)
    append!(ridge_z, rz)
end

# Define your search space
multi_r2 = range(1.1, 1.25, length=10)
phi = pi/5.0


# 2. Run the loop across the radial range
for r in multi_r2
    
    println("Tracing field lines for r = ", round(r, digits=3))

    # Run the simulation for this specific radius
    # We unpack the 6 returned vectors into temporary variables
    px, py, pz, rx, ry, rz = run_poincare(r, pi/5.0, 20000.0)

    # 3. Append the results to your master lists
    append!(poincare_x, px)
    append!(poincare_y, py)
    append!(poincare_z, pz)

    append!(ridge_x, rx)
    append!(ridge_y, ry)
    append!(ridge_z, rz)
end

# Define your search space
multi_r3 = range(0.65, 1.2, length=10)
phi = 2.0*pi/5.0


# 2. Run the loop across the radial range
for r in multi_r3
    
    println("Tracing field lines for r = ", round(r, digits=3))

    # Run the simulation for this specific radius
    # We unpack the 6 returned vectors into temporary variables
    px, py, pz, rx, ry, rz = run_poincare(r, 2.0*pi/5.0, 20000.0)

    # 3. Append the results to your master lists
    append!(poincare_x, px)
    append!(poincare_y, py)
    append!(poincare_z, pz)

    append!(ridge_x, rx)
    append!(ridge_y, ry)
    append!(ridge_z, rz)
end

# Define your search space
multi_r4 = range(0.625, 1.2, length=10)
phi = 3.0*pi/5.0


# 2. Run the loop across the radial range
for r in multi_r4
    
    println("Tracing field lines for r = ", round(r, digits=3))

    # Run the simulation for this specific radius
    # We unpack the 6 returned vectors into temporary variables
    px, py, pz, rx, ry, rz = run_poincare(r, 3.0*pi/5.0, 20000.0)

    # 3. Append the results to your master lists
    append!(poincare_x, px)
    append!(poincare_y, py)
    append!(poincare_z, pz)

    append!(ridge_x, rx)
    append!(ridge_y, ry)
    append!(ridge_z, rz)
end


# Define your search space
multi_r5 = range(1.1, 1.25, length=10)
phi = 4.0*pi/5.0
# 2. Run the loop across the radial range
for r in multi_r5
    
    println("Tracing field lines for r = ", round(r, digits=3))

    # Run the simulation for this specific radius
    # We unpack the 6 returned vectors into temporary variables
    px, py, pz, rx, ry, rz = run_poincare(r, 4.0*pi/5.0, 20000.0)

    # 3. Append the results to your master lists
    append!(poincare_x, px)
    append!(poincare_y, py)
    append!(poincare_z, pz)

    append!(ridge_x, rx)
    append!(ridge_y, ry)
    append!(ridge_z, rz)
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
scatter!(ax, poincare_x, poincare_y, poincare_z, color = :red, markersize = 4)

#Plot Magnetic Ridge
scatter!(ax, ridge_x,ridge_y, ridge_z, color = :purple, markersize = 4)

display(fig)

