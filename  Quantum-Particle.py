 import numpy as np
import matplotlib.pyplot as plt

# --- 1. Physical Parameters ---
# Using scaled natural units (G = M = m = hbar = 1) for numerical stability
rs = 2.0                 # Schwarzschild radius (rs = 2GM/c^2 -> 2 in these units)
L = 100.0                # Size of the spatial grid
N = 4096                 # Number of grid points (power of 2 is optimal for FFT)
r = np.linspace(rs + 0.05, rs + L, N)  # Radial coordinate starting just outside horizon
dr = r[1] - r[0]         # Spatial step size

# Momentum grid (frequencies matched to the spatial grid for the FFT)
k = np.fft.fftfreq(N, d=dr) * 2 * np.pi

# --- 2. Time Parameters ---
dt = 0.005               # Time step
total_steps = 3000       # Total iterations
plot_interval = 50       # Update visualization every N steps

# --- 3. Paczyński-Wiita Potential ---
# V(r) = -GMm / (r - rs)
V = -1.0 / (r - rs)

# We cap the potential near the event horizon to prevent infinite values
# which would cause the phase exponent to oscillate wildly
V[V < -100] = -100       

# --- 4. Initial Wave Packet (Gaussian) ---
r0 = 30.0                # Starting distance from the black hole
sigma = 2.0              # Width of the wave packet
p0 = -2.5                # Initial momentum (negative = moving inward)

# Construct and normalize the initial wavefunction
psi = np.exp(-(r - r0)**2 / (4 * sigma**2)) * np.exp(1j * p0 * r)
psi = psi / np.sqrt(np.sum(np.abs(psi)**2 * dr))

# --- 5. Split-Operator Phase Factors ---
# Pre-computing these operators saves processing time inside the main loop
U_V = np.exp(-1j * V * (dt / 2.0))      # Half-step potential operator
U_K = np.exp(-1j * (k**2 / 2.0) * dt)   # Full-step kinetic operator

# --- 6. Visualization Setup ---
plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the pseudo-Newtonian potential (scaled down to fit on the same axis)
ax.plot(r, V * 0.05, 'r--', alpha=0.6, label='Scaled Potential $V_{PW}(r)$')
ax.axvline(rs, color='k', linestyle=':', label='Event Horizon ($r_s$)')

# Plot line for the probability density
line, = ax.plot(r, np.abs(psi)**2, 'b-', linewidth=2, label='Probability Density $|\psi|^2$')

ax.set_xlim(0, 50)
ax.set_ylim(-0.3, 0.4)
ax.set_xlabel('Radial Distance $r$')
ax.set_ylabel('Amplitude')
ax.set_title('Quantum Particle Falling into a Black Hole')
ax.legend(loc='upper right')

# --- 7. Main Time-Evolution Loop ---
for step in range(total_steps):
    # Step A: Apply half-step potential phase
    psi *= U_V
    
    # Step B: Transform to momentum space
    psi_k = np.fft.fft(psi)
    
    # Step C: Apply full-step kinetic phase
    psi_k *= U_K
    
    # Step D: Transform back to position space
    psi = np.fft.ifft(psi_k)
    
    # Step E: Apply half-step potential phase
    psi *= U_V
    
    # --- Boundary Absorption (The Event Horizon) ---
    # FFTs assume periodic boundaries. Without an absorber, a wave falling into 
    # the black hole will wrap around and appear on the right side of the grid.
    # We apply an imaginary absorbing layer near rs.
    absorber = np.ones_like(r)
    absorber[r < rs + 1.0] = 0.90  # Dampens the wave packet near the horizon
    psi *= absorber
    
    # Renormalize to conserve probability for the surviving part of the wave
    norm = np.sum(np.abs(psi)**2 * dr)
    if norm > 0:
        psi = psi / np.sqrt(norm)

    # Update the animation
    if step % plot_interval == 0:
        line.set_ydata(np.abs(psi)**2)
        plt.pause(0.01)

plt.ioff()
plt.show()
