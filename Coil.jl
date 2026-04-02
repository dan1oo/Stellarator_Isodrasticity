using StaticArrays
# using SymplecticMapTools
using JSON
using LinearAlgebra
# using GLMakie
using CairoMakie
import GLMakie as MakieVersion
# using BenchmarkTools
using OrdinaryDiffEq
# import GeometricIntegrators as GI
using Interpolations
using ForwardDiff
using FastGaussQuadrature
using JLD2


# use Float128 or Double64 from DoubleFloats.jl
# MultiFloats.jl appears to be good too

function three_norm(x::SVector{3, T}) where {T}
    sqrt(x[1]*x[1] + x[2]*x[2] + x[3]*x[3])
end

struct Coil{T}
    a::Vector{SVector{3,T}} # Vector of SVectors representing the Fourier coefficients of the coil
    I::T         # Current of the coil

    rs::Vector{SVector{3,T}}   # Vector of SVectors giving position on quadrature nodes
    Idrs::Vector{SVector{3,T}} # Quadrature weights used for evaluating the magnetic vector potential
end

function copy(c::Coil{T}, S::Type) where {T}
    Coil(SVector{3,S}.(c.a),
         S(c.I),
         SVector{3,S}.(c.rs),
         SVector{3,S}.(c.Idrs))
end

function FourierEvaluateMatrix(θs::AbstractVector{T}, N::Integer) where {T}
    ns = 1:N

    F = [iseven(n) ? sin((n÷2)*θ) : cos((n÷2)*θ) for θ in θs, n in ns]
    
    F
end

function d_FourierEvaluateMatrix(θs::AbstractVector{T}, N::Integer) where {T}
    ns = 1:N

    F = [iseven(n) ? (n÷2) * cos((n÷2)*θ) : -(n÷2) * sin((n÷2)*θ) for θ in θs, n in ns]
    
    F
end

function evaluate_coil_rs(c::Coil{T}, θs::AbstractVector{T}) where {T}
    N = length(c.a)
    F = FourierEvaluateMatrix(θs, N)
    F*c.a
end

function deval_coils_rs(c::Coil{T}, θs::AbstractVector{T}) where {T}
    N = length(c.a)
    d_F = d_FourierEvaluateMatrix(θs, N)
    d_F*c.a
end

function Coil(a::Vector{SVector{3,T}}, I::T, Nquad::Integer) where {T}
    N  = length(a)
    θs = (0:Nquad-1) .* (2T(π)/Nquad)
    
    F = FourierEvaluateMatrix(θs, N)
    d_F = d_FourierEvaluateMatrix(θs, N)
    
    μ0 = 4T(π) * T(10)^-7

    rs  = F*a
    Idrs = ( (μ0*I/(4T(π))) * (2T(π)/Nquad) ) .* d_F * a

    Coil(a, I, rs, Idrs)
end

# function evaluate_A(c::Coil, x::AbstractVector)
#     Rs = [norm(ri - x) for ri in c.rs]
#     sum(c.Idrs ./ Rs)
# end

function eval_A_kahan(cs::Vector{Coil{T}}, x::AbstractVector{T}) where {T}
    A = StaticArrays.zeros(T, 3)
    for c in cs
        A = A + eval_A_kahan(c, x)
    end
    A
end

function eval_A_kahan(c::Coil{T}, x::AbstractVector{T}) where {T}
    A  = StaticArrays.zeros(T, 3)
    dA = StaticArrays.zeros(T, 3)
    
    for (ri, Idri) in zip(c.rs, c.Idrs)
        # Get the component of the field from ri
        delta_r = ri - x
        fi = Idri/three_norm(delta_r) - dA
        
        # Perform the Kahan summation step
        A_next = A + fi
        dA = (A_next - A) - fi
        # dA = tmp - fi
        A = A_next
    end

    A
end

function eval_A(c::Coil{T}, x::AbstractVector{T}) where {T}
    A = StaticArrays.zeros(T, 3)
    for (ri, Idri) in zip(c.rs, c.Idrs)
        delta_r = ri - x
        # delta_r = SA[ri[1]-x[1], ri[2]-x[2], ri[3]-x[3]]
        den = three_norm(delta_r)
        A = A + Idri ./ den
        # A = SA[A[1] + Idri[1] / den, A[2] + Idri[2] / den, A[3] + Idri[3] / den]
    end

    A
end

function eval_A_alloc(c::Coil{T}, x::AbstractVector{T}) where {T}
    # dens = [three_norm(ri - x) for ri in c.rs]
    dens = three_norm.(c.rs .- Ref(x))
    sum(c.Idrs ./ dens)
end

function eval_B(c::Coil{T}, x::AbstractVector{T}) where {T}
    B = StaticArrays.zeros(T, 3)

    for (ri, Idri) in zip(c.rs, c.Idrs)
        delta_r = ri - x
        B = B + cross(delta_r, Idri) / (three_norm(delta_r)^3)
    end

    B
end

function eval_dAdr(c::Coil{T}, x::AbstractVector{T}) where {T}
    dAdr = StaticArrays.zeros(T,3,3)

    for (ri, Idri) in zip(c.rs, c.Idrs)
        delta_r = ri - x
        dAdr = dAdr + Idri * (delta_r' ./ three_norm(delta_r)^3)
    end

    dAdr
end

function eval_A_and_dAdr(c::Coil{T}, x::AbstractVector{T}) where {T}
    A = StaticArrays.zeros(T, 3)
    dAdr = StaticArrays.zeros(T, 3, 3)

    for (ri, Idri) in zip(c.rs, c.Idrs)
        delta_r = ri - x
        A = A + Idri/three_norm(delta_r)

        dAdr = dAdr + Idri * (delta_r' ./ three_norm(delta_r)^3)
    end

    A, dAdr
end

function eval_A_and_dAdr_kahan(c::Coil{T}, x::AbstractVector{T}) where {T}
    A  = StaticArrays.zeros(T, 3)
    dA = StaticArrays.zeros(T, 3)

    dAdr   = StaticArrays.zeros(3,3)
    d_dAdr = StaticArrays.zeros(3,3)

    for (ri, Idri) in zip(c.rs, c.Idrs)
        delta_r = ri - x
        fi = Idri/three_norm(delta_r) - dA
        dfidr = Idri * (delta_r' ./ three_norm(delta_r)^3) - d_dAdr

        #
        A_next = A + fi
        dA = (A_next - A) - fi
        A = A_next

        dAdr_next = dAdr + dfidr
        d_dAdr = (dAdr_next - dAdr) - dfidr
        dAdr = dAdr_next
        # B = B + cross(Idri, delta_r) / (three_norm(delta_r)^3)
    end

    A, dAdr
end

function eval_A(cs::Vector{Coil{T}}, x::AbstractVector{T}) where {T}
    A = StaticArrays.zeros(T, 3)
    for c in cs
        A = A + eval_A(c, x)
    end
    A
end

function eval_dAdr(cs::Vector{Coil{T}}, x::AbstractVector{T}) where {T}
    dAdr = StaticArrays.zeros(T, 3, 3)
    for c in cs
        dAdr = dAdr + eval_dAdr(c, x)
    end
    dAdr
end


function eval_A_and_dAdr_kahan(cs::Vector{Coil{T}}, x::AbstractVector{T}) where {T}
    A = StaticArrays.zeros(T, 3)
    dAdr = StaticArrays.zeros(T, 3, 3)
    for c in cs
        A_c, dAdr_c = eval_A_and_dAdr_kahan(c, x)
        A = A + A_c
        dAdr = dAdr + dAdr_c
    end
    A, dAdr
end

function eval_A_and_dAdr(cs::Vector{Coil{T}}, x::AbstractVector{T}) where {T}
    A = StaticArrays.zeros(T, 3)
    dAdr = StaticArrays.zeros(T, 3, 3)
    for c in cs
        A_c, dAdr_c = eval_A_and_dAdr(c, x)
        A = A + A_c
        dAdr = dAdr + dAdr_c
    end
    A, dAdr
end

function eval_B(cs::Vector{Coil{T}}, x::AbstractVector{T}) where {T}
    B = StaticArrays.zeros(T, 3)
    for c in cs
        B = B + eval_B(c, x)
    end
    B
end

# function evaluate_B(c::Coil, x::AbstractVector)
#     delta_rs = [ri - x for ri in c.rs]
#     sum([cross(Idri, delta_ri) / (delta_ri'delta_ri) for (Idri, delta_ri) in zip(c.Idrs, delta_rs)])
# end



function get_coils_from_file(T::Type, coil_file, Nquad)
    phi_coils = T[];
    flip_coils = Bool[];
    
    coils_py = JSON.parsefile(coil_file; allownan=true);
    coils = Coil{T}[]
    objs = coils_py["simsopt_objs"]
    

    
    
    function get_current(current, scale, objs)
        cls = current["@class"]
        if cls == "ScaledCurrent"
            scale   = T(scale) * T(current["scale"])
            current = objs[current["current_to_scale"]["value"]]
            return get_current(current, scale, objs)
        elseif cls == "CurrentSum"
            current_a = objs[current["current_a"]["value"]]
            current_b = objs[current["current_b"]["value"]]
            return get_current(current_a, scale, objs) + get_current(current_b, scale, objs)
        end
    
        return T(scale) * T(current["current"])
    end
    
    for (key, value) in objs
        if key[1:4] == "Coil"
            current = objs[value["current"]["value"]]
            J = get_current(current, T(1), objs)
            
            val = value["curve"]["value"]
            curve = objs[val]
            flip = false;
            phi = T(0);
            # rotated = false
    
            while curve["@name"][1:12] == "RotatedCurve"
                flipi = curve["flip"]
                phi = T(curve["phi"])
                flip = flipi ? !flip : flip
                val = curve["curve"]["value"]
                curve = objs[val]
            end
                
            dofs = objs[curve["dofs"]["value"]]
            # names = dofs["names"]
            
            coef_data = dofs["x"]["data"]
            coeffs = zeros(T, size(coef_data))
            coeffs[:] .= T.(coef_data[:]) # We have to do this because coef_data is an array of Any, and we want to reinterpret
            
            Ms_coil = length(coeffs)÷3;
            coeffs = reshape(coeffs, Ms_coil, 3)
            
            rotmat = [cos(phi) -sin(phi) 0; sin(phi) cos(phi) 0; 0 0 1];
            

            if flip
                rotmat = rotmat * Diagonal([1, -1, -1])
            end
            coeffs = coeffs * rotmat
            r0_coil = reinterpret(SVector{3,T}, coeffs')[:]
    
            push!(coils, Coil(r0_coil, J, Nquad))
            push!(phi_coils, phi)
            push!(flip_coils, flip)
        end
    end
    
    coils
end


function get_coils_from_file(coil_file, Nquad)
    get_coils_from_file(Float64, coil_file, Nquad)
end

# tspan = (0.0, 1.0)
# prob = ODEProblem(f, u0, tspan)
# sol = solve(prob, Tsit5(), reltol = 1e-8, abstol = 1e-8)
# using Plots
# plot(sol, linewidth = 5, title = "Solution to the linear ODE with a thick line",
#     xaxis = "Time (t)", yaxis = "u(t) (in μm)", label = "My Thick Line!") # legend=false
# plot!(sol.t, t -> 0.5 * exp(1.01t), lw = 3, ls = :dash, label = "True Solution!"
function follow_B_line(x0::AbstractVector{T}, L::T, coils::Union{Coil{T}, Vector{Coil{T}}}; 
                       method=Vern8(), reltol=1e-12, abstol=1e-12, saveat=nothing, kwargs...) where {T}
    function f(x, _, _)
        B = eval_B(coils, x)
        B / sqrt(B'B)
    end

    prob = OrdinaryDiffEq.ODEProblem(f, x0, (T(0), L))
    if saveat === nothing
        solve(prob, method; reltol = reltol, abstol = abstol, kwargs...)
    else
        solve(prob, method; reltol = reltol, abstol = abstol, saveat = saveat, kwargs...)
    end
end

function get_B_map(L::T, coils::Union{Coil{T}, Vector{Coil{T}}}; 
                   method=Vern8(), reltol=1e-12, abstol=1e-12) where {T}

    function F(x0)
        sol = follow_B_line(x0, L, coils; method, reltol, abstol)
        sol.u[end]
    end

    F
end

const FUNDAMENTAL_CHARGE = 1.602176634e-19
const ELECTRON_MASS = 9.1093837e-31
const PROTON_MASS   = 1.67262192e-27
const ALPHA_MASS    = 6.64465734e-27
const EV_TO_JOULE   = 1.60217663e-19
function eV_to_velocity(energy, mass)
    sqrt(2energy * EV_TO_JOULE / mass)
end

function velocity_to_eV(v, m)
    (m*v^2)/2/EV_TO_JOULE
end

function get_charged_f(coils::Union{Coil{T}, Vector{Coil{T}}}, charge::T, mass::T, v_scale::T) where {T}
    function f(q, _, _)
        x = q[SA[1,2,3]]
        v = q[SA[4,5,6]]
        
        B = eval_B(coils, x)
        xdot = v_scale*v
        vdot = (charge/mass) * cross(v, B)

        SA[xdot[1],xdot[2],xdot[3],vdot[1],vdot[2],vdot[3]]
    end

    f
end

function charged_particle_dynamics_B(q0::AbstractVector{T}, L::T, coils::Union{Coil{T}, Vector{Coil{T}}}, 
                                     charge::T, mass::T; reltol::T=1e-14, 
                                     abstol::T=1e-14, maxiters = 1e7) where {T}
    x0 = q0[SA[1,2,3]]
    v0 = q0[SA[4,5,6]]
    v_scale = sqrt(v0'v0)

    f = get_charged_f(coils, charge, mass, v_scale)

    q0 = vcat(x0, v0/v_scale)
    
    t_span = (T(0), L/v_scale)
    prob = OrdinaryDiffEq.ODEProblem(f, q0, t_span)
    
    solve(prob, Tsit5(), reltol = reltol, abstol = reltol, maxiters=maxiters)
end

function charged_particle_map_B(coils::Union{Coil{T}, Vector{Coil{T}}}, charge::T, mass::T, 
                                v_scale::T, B_scale::T; reltol::T=1e-12, abstol::T=1e-12, 
                                maxiters = 1e7, method = Vern8(), overshoot = 3) where {T}
    f = get_charged_f(coils, charge, mass, v_scale)

    gyrofrequency = abs(charge * B_scale / mass)
    TT = overshoot * 2pi / gyrofrequency 

    callback_indicator = get_callback_indicator(coils, charge, mass)
    function callback_affect!(integrator)
        # println("calling callback_affect!")
        terminate!(integrator)
    end
    callback_affect_neg! = nothing;
    function callback_condition(q, t, integrator)
        if t < 0.001 * v_scale * 2π / gyrofrequency
            return 1.
        end
        qi = callback_indicator(q)
    end
    callback = ContinuousCallback(callback_condition, callback_affect!, 
                                  callback_affect_neg!)

    function F(q0)
        x0 = q0[SA[1,2,3]]
        v0 = q0[SA[4,5,6]]
        q0 = vcat(x0, v0/v_scale)

        t_span = (0, TT * v_scale)
        prob = OrdinaryDiffEq.ODEProblem(f, q0, t_span)
        sol = solve(prob, method, reltol = reltol, abstol = abstol, maxiters=maxiters, callback=callback)

        x1 = sol.u[end][SA[1,2,3]]
        v1 = sol.u[end][SA[4,5,6]] .* v_scale
        return sol.t[end], SA[x1[1],x1[2],x1[3],v1[1],v1[2],v1[3]]
    end

    F
end

function get_charged_vfh(coils::Union{Coil{T}, Vector{Coil{T}}}, charge::T, mass::T) where {T}
    function velocity(v::AbstractVector{T}, t, x::AbstractVector{T}, p::AbstractVector{T}, params)
        X = SA[x[1],x[2],x[3]]
        P = SA[p[1],p[2],p[3]]
        # A = eval_A_kahan(coils, X)
        A = eval_A(coils, X)
        v[SA[1,2,3]] = (P - charge*A)/mass
    end
    
    function force(f::AbstractVector{T}, t, x::AbstractVector{T}, p::AbstractVector{T}, params)
        X = SA[x[1],x[2],x[3]]
        P = SA[p[1],p[2],p[3]]
        # A, dAdr = eval_A_and_dAdr_kahan(coils, X)
        A, dAdr = eval_A_and_dAdr(coils, X)
        f[SA[1,2,3]] = dAdr' * (P - charge*A) * (charge/mass)
    end

    function hamiltonian(t, x::AbstractVector{T}, p::AbstractVector{T}, params)
        X = SA[x[1],x[2],x[3]]
        P = SA[p[1],p[2],p[3]]
        # A = eval_A_kahan(coils, X)
        A = eval_A(coils, X)
        MV = P - charge * A
        MV'MV / (2mass)
    end

    velocity, force, hamiltonian
end

function get_charged_ham_f(coils::Union{Coil{T}, Vector{Coil{T}}}, charge::T, mass::T, p_scale::T) where {T}
    function f(q, _, _)
        X = q[SA[1,2,3]]
        P = p_scale * q[SA[4,5,6]]
        
        # A, dAdr = eval_A_and_dAdr_kahan(coils, X)
        A, dAdr = eval_A_and_dAdr(coils, X)
        
        V = (P - charge*A)/mass
        F = dAdr' * (P - charge*A) * (charge/(mass * p_scale))

        SA[V[1], V[2], V[3], F[1], F[2], F[3]]
    end

    f    
end

function charged_particle_dynamics(q0::AbstractVector{T}, TT::T, 
                                   coils::Union{Coil{T}, Vector{Coil{T}}}, charge::T, mass::T, 
                                   dt::T; maxiters = 1e7) where {T}
    x0 = q0[SA[1,2,3]]
    p0 = q0[SA[4,5,6]]
    p_scale = T(1)
    
    f = get_charged_ham_f(coils, charge, mass, p_scale)
    q0 = vcat(x0, p0)
    
    t_span = (T(0), TT)
    prob = OrdinaryDiffEq.ODEProblem(f, q0, t_span)
    
    solve(prob, ImplicitMidpoint(), adaptive=false, dt=dt, maxiters=maxiters)
end

# function charged_particle_dynamics_GI(q0::AbstractVector{T}, TT::T, 
#                                       coils::Union{Coil{T}, Vector{Coil{T}}}, charge::T, mass::T, 
#                                       dt::T, method = GI.ImplicitMidpoint()) where {T}
#     p_scale = norm(q0[4:6]) / norm(q0[1:3])
#     f0 = get_charged_ham_f(coils, charge, mass, p_scale)
#     f = (xdot, t, x, params) -> xdot[1:6] = f0(x, t, params)
#     q0 = vcat(q0[1:3], q0[4:6] ./ p_scale)
    
#     t_span = (T(0),TT)
#     prob = GI.ODEProblem(f, t_span, dt, Vector(q0))
#     # methods GI.SRK3()
#     # method = GI.Gauss(4)
#     int = GI.GeometricIntegrator(prob, method)
#     sol = GI.integrate(int)

#     NN = length(sol.t)
#     t = zeros(NN)
#     u = zeros(6, NN)
#     for ii = 1:NN
#         t[ii] = sol.t[ii-1]
#         u[:, ii] = sol.q[ii-1, :]
#     end

#     u[4:6, :] .= u[4:6, :] .* p_scale
#     t, u, sol
# end

function get_callback_indicator(coils::Union{Coil{T}, Vector{Coil{T}}}, charge::T, mass::T) where {T}
    function callback(q::AbstractVector{T})
        X = q[SA[1,2,3]]
        P = q[SA[4,5,6]]

        A = eval_A(coils, X)
        B = eval_B(coils, X)
        Bhat = B / sqrt(B'B) 
        
        V = (P - charge*A)/mass

        # ez = cross(SA[0., 0., 1.], B)
        ez = SA[T(0), T(0), T(1)]

        Vperp = V - (Bhat'V)*Bhat
        ez'Vperp / sqrt(Vperp'Vperp) # / sqrt(ez'ez)

        # V[3] / eV_to_velocity(energy, mass)
    end

    callback
end

function find_root(f::Function, x)
    x = 1.0 * x
    for ii = 1:5
        fp = ForwardDiff.derivative(f, x)
        x = x - f(x)/fp
    end
    
    return x
end

function get_crossings(t::AbstractVector, u::AbstractVector, coils, charge::Number, mass::Number)
    callback = get_callback_indicator(coils, charge, mass)
    c = callback.(u)

    t_cs = Float64[]
    u_cs = SVector{6, Float64}[]

    spline = BSpline(Cubic(Free(OnGrid())))
    c_itp = interpolate(c, spline)
    t_itp = interpolate(t, spline)
    u_itp = interpolate(u, spline)

    Nu = length(u)
    for ii = 1:Nu-1
        if (c[ii] < 0) && (c[ii+1] > 0)
            # s = bisection(c_itp, ii, ii+1)
            s = find_root(c_itp, ii)
            
            push!(t_cs, t_itp(s))
            push!(u_cs, u_itp(s))
        end
    end

    t_cs, u_cs
end


# Get 3D grid of points to plot on the torus
function plot_eval_on_grid(tor, Nθs, Q)
    θvec = [(0:Nθ-1) .* (2π/Nθ) for Nθ in Nθs]
    x = evaluate_on_grid(tor, θvec);
    Nisland = size(x,2)
    # display(Nisland)
    # display(x)
    
    x_plot = zeros(size(Q,2),Nisland,Nθs...)
    for ii = 1:Nθs[1], jj = 1:Nθs[2]
        x_plot[:,:,ii,jj] = Q'x[:,:,ii,jj]
    end
    x_plot
end

function plot_eval_on_grid_3D(tor, Nθs, Q)
    θvec = [(0:Nθ-1) .* (2π/Nθ) for Nθ in Nθs]
    x = evaluate_on_grid(tor, θvec);
    Nisland = size(x,2)
    # display(Nisland)
    # display(x)
    
    x_plot = zeros(size(Q,2),Nisland,Nθs...)
    for ii = 1:Nθs[1], jj = 1:Nθs[2], kk = 1:Nθs[3]
        x_plot[:,:,ii,jj,kk] = Q'x[:,:,ii,jj,kk]
    end
    x_plot
end

function torus_mesh!(ax, xs; color=:lightgray)
    Ns = size(xs)[3:4]
    Nisland = size(xs,2)
    D = size(xs,1)
    
    faces = zeros(Integer, 3, 2, Ns[1], Ns[2], Nisland)
    for ii = 1:Nisland
        for jj = 1:Ns[1], kk = 1:Ns[2]
            j1 = jj
            j2 = mod1(jj+1,Ns[1])
            k1 = kk
            k2 = mod1(kk+1,Ns[2])

            l1 = (k1-1) * Ns[1] + j1
            l2 = (k1-1) * Ns[1] + j2
            l3 = (k2-1) * Ns[1] + j1
            l4 = (k2-1) * Ns[1] + j2
            
            faces[:,1,jj,kk,ii] = [l1, l2, l3]
            faces[:,2,jj,kk,ii] = [l4, l3, l2]
        end
        # MakieVersion.mesh!(reshape(xs[:,ii,:,:], D, Ns[1]*Ns[2]), reshape(faces[:,:,:,:,ii], 3, 2*Ns[1]*Ns[2])', color = color, backlight=0.1)
        MakieVersion.mesh!(ax, reshape(xs[:,ii,:,:], D, Ns[1]*Ns[2]), reshape(faces[:,:,:,:,ii], 3, 2*Ns[1]*Ns[2])', color=color, transparency = true)
    end
end
