using OrdinaryDiffEq
using CairoMakie
using StaticArrays
using LinearAlgebra


include("Coil.jl")

function run_poincare()


    #= 
    ------------------------------------------------------------------------------------------------------
    Initial Conditions:
    Define radius, azimuthal angle, and field line 

    =#

    coil_file = "landreman_paul.json"
    Nquad = 64 
    coils = get_coils_from_file(Float64, coil_file, Nquad)

    


    # Poincare Section

    # Arrays to store our points
    poincare_x = Float64[]
    poincare_y = Float64[]
    poincare_z = Float64[]

    # Define the Poincaré callback when the integrator crosses the initial phi plane

    function plane_condition(u, t, integrator)
        x = u[1]
        y = u[2]
        
        return -x*sin(phi) + y * cos(phi)
    end

    # When crosses the plane, save the x y and z coordinates. 
    
    function save_point!(integrator)
        
        push!(poincare_x, integrator.u[1])
        push!(poincare_y, integrator.u[2])
        push!(poincare_z, integrator.u[3])
        
    end

    # Set up the callback
    cb = ContinuousCallback(plane_condition, save_point!, nothing; save_positions=(false, false))


    #

    #  Define the ODE: dx/ds = B / |B|v- Magnetic field ODE
    function b_field_ode(u, p, t)
        B = eval_B(coils, u)
        B / norm(B)
    end

    # Starting Point


    radius = range(1.0,1.5, length = 15)    # Distance from origin at Z = 0
    phi = 4.0*pi/5.0        #Azimuthal angle, issues on y axis (90, 270 degrees)
    Length = 1500.0 # Integration Path length

    for r in radius
        
        L_max = Length # Total distance to trace field line
        phi = phi      # Starting Phi

        x = r * cos(phi)
        y = r * sin(phi)


        u0 = SA[x, y, 0.0]
        prob = ODEProblem(b_field_ode, u0, (0.0, L_max))
            
        # We use Vern9() for high precision, keeping tolerances tight so no drift

        solve(prob, Vern9(), reltol=1e-10, abstol=1e-10, callback=cb, save_everystep=false, save_start=false, save_end=false)
    end
    
    # Plot Coils and Magnetic Field surface

    coil_file = "landreman_paul.json"
    Nquad = 64 
    coils = get_coils_from_file(Float64, coil_file, Nquad)

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
    scatter!(ax, poincare_x, poincare_y, poincare_z, 
            color = :red, markersize = 4)

    display(fig)

    fig2 = Figure(size = (800, 800))
    ax2 = Axis(fig2[1, 1], 
        xlabel = "R (m)", 
        ylabel = "Z (m)", 
        title = "Poincaré Section at ϕ = 0",
        aspect = DataAspect()
    )

    poincare_xy = sqrt.(poincare_x .* poincare_x .+ poincare_y .* poincare_y)

    scatter!(ax2, poincare_xy, poincare_z, markersize = 2, color = :black)
    
    # Save and display
    save("poincare_landreman_paul.png", fig2)
    display(fig2)
    
    
end

@time run_poincare()

