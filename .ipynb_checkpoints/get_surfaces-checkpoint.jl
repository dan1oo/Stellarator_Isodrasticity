source_file = "./Coil.jl"
data_dir = "./data/"  
plot_dir = "./plots/"
coil_dir = "./"


coil_file = coil_dir * "landreman_paul.json"

## Get the coils
include(source_file)
Nquad = 100;
coils  = get_coils_from_file(coil_file, Nquad);

include(source_file)

# Compute the flux surfaces
compute_surfs = true
if compute_surfs
    LB = 2.
    F = get_B_map(LB, coils)

    # Landreman-Paul flux surface initial positions between (xl,0,0) and (xr,0,0)
    xl = 1.21
    xr = 1.29 
    Nsurf = 10   # Number of surfaces to compute
    

    # Initialize trajectories
    Btrajs = Matrix{SVector{3, Float64}}(undef, Nsurf, Ltraj)
    Btrajs[:, 1] = [SA[x, 0., 0.] for x in LinRange(xl, xr, Nsurf)]
    
    
    Nw0 = 2      # Torus dimension (can do Nw0=1 for surfaces on a Poincare map)
    h = (x) -> x # Torus observable (the identity)
    K = 400      # Filter resolution
    T = K        # Least-squares system height
    Nx = T+2K+1  # Necessary number of maps 
    Ltraj = Nx   # Could evolve longer if wanted
    
    # Save metadata (using JLD2.jl)
    @save "$(data_dir)Bdata.jld2" Nsurf LB Ltraj xl xr 
    
    # Evolve trajectories
    println("Computing trajectories")
    @time begin
    for ii = 1:Nsurf
        for jj = 2:Ltraj
            Btrajs[ii,jj] = F(Btrajs[ii,jj-1])
        end
    end
    end

    # Perform RRE
    for ii = 1:Nsurf
        xs = reshape(reinterpret(Float64, Btrajs[ii,:]), 3, Ltraj)
        
        # println("starting the extrapolation")
        F_dummy = (x) -> nothing
        sol_2d = birkhoff_extrapolation(h, F_dummy, xs[:,1], T, K; x_prev=xs[:,1:Nx]);
        println("Birkhoff RRE resid = $(sol_2d.resid_rre)")
        get_w0!(sol_2d, Nw0; Nsearch=20,gridratio=5)
        # println("Finding the torus")
        adaptive_get_torus!(sol_2d)
        println("ii=$ii Torus resid = $(sol_2d.resid_tor)")
        save_rre("$(data_dir)surf_$(ii).jld2", sol_2d)
    end
end
;


include(source_file)

plot_tori = true
if plot_tori
    @load "$(data_dir)Bdata.jld2" Nsurf LB Ltraj xl xr
    
    Nθs = [200,200]
    Q = Matrix(1.0I, 3, 3)
    
    for i_surf = 1:Nsurf
        println("i_surf=$(i_surf)")
        surf_sol = load_rre("$(data_dir)surf_$(i_surf).jld2")
        tor = surf_sol.tor
        xs = plot_eval_on_grid(tor, Nθs, Q)
    
        part_xs = plot_eval_on_grid(tor, Nθs, Q)
        
        f = Figure(size=(800,800))
        ax = Axis3(f[1,1], title="tor resid = $(surf_sol.resid_tor)")

        for (ii, coil) in enumerate(coils)
            rs = coil.rs
            rs = vcat(rs, rs[[1]])
            lines!(rs, linewidth=3, color = :black)
        end
        
        torus_mesh!(ax, xs)
        save("$(plot_dir)surf_$(i_surf).png", f)
    end
end