# Stellarator Isodrasticity — Project Structure

## Where the data is

- Main data directory: `data/`
- Ridge surface outputs: `data/magnetic_ridge3/`
  - `magnetic_ridge_3_1.jld2`
  - `ridge_fit_coeffs_v3.jld2`
- Ridge/valley paired output: `data/magnetic_ridge_valley3/`
  - `magnetic_ridge_valley_3.jld2`
- Ridge fit coefficients used by invariant calculations:
  - `data/ridge_coeffs.jld2`

## Three main notebooks

1. `Magnetic_Ridge_Set_1.ipynb`  
   Upstream ridge-generation workflow: builds ridge peak datasets from coil field traces and dense refinement.

2. `Magnetic_Ridge_Set_2.ipynb`  
   Surface parameterization workflow: processes ridge/valley data and fits the polynomial-Fourier ridge representation.

3. `calculate_invariant_J.ipynb`  
   Invariant evaluation workflow: traces field lines, finds bounce points, and computes longitudinal invariant `J` (including phase-space diagnostics and scan cells).

## Other notebook

- `Stellarator_Isodrasticity.ipynb`  
  General analysis/visualization notebook for loading ridge-valley outputs and inspecting isodrasticity-related geometry.
