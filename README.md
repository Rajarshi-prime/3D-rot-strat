# Codes for dimensional Rotating stratified code.
* Contains for GPU and MPI parallelized CPU turbulence.
* Incorporates both 2/3rd and Phase-shifted dealiasing. 
* Forces the balanced and the unbalanced flows separately. 
* Can run both viscous and hyperviscous simulations.
* The (hyper)viscosity can be calculated explicitly, implicitly, or exponentially.

<u>Note</u>q: The MPI code is based on Mortensen's [Spectral DNS code](https://github.com/spectralDNS/spectralDNS) with the following improvements:

1. Saves energy and energy flux spectra on the fly with minimal computational overhead. 
2. Uses constant power input forcing similar to Pope's paper. 
3. Time-step, viscosity and hyper-viscosity automatically scales with resolution. 
4. Uses better [phase-shifted dealiasing](https://www.researchgate.net/profile/G-Patterson/publication/252764310_Spectral_Calculations_of_Isotropic_Turbulence_Efficient_Removal_of_Aliasing_Interactions/links/5e6fd588299bf14570f26312/Spectral-Calculations-of-Isotropic-Turbulence-Efficient-Removal-of-Aliasing-Interactions.pdf) in files labelled `_ps' to compute the non-linear terms. 


<u>Note</u>: The GPU code is inspired from Anikat's code with additional phase-shifted dealiasing and constant power input forcing for balanced and unbalanced flows separately.

The GPU code requires JAX and is written for single GPU. 


**Snapshots**: 
