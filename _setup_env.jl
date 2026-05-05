using Pkg
const DIR = joinpath(@__DIR__)
Pkg.activate(DIR)
pkgs = [
    "OrdinaryDiffEq",
    "CairoMakie",
    "StaticArrays",
    "ForwardDiff",
    "JSON",
    "Interpolations",
    "FastGaussQuadrature",
    "JLD2",
    "Roots",
    "GLMakie",
    "QuadGK",
    "DataInterpolations",
    "DifferentialEquations",
    "Statistics",
]
Pkg.add(pkgs)
println("OK: environment at ", DIR)
