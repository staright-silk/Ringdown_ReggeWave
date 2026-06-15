import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Quantum Wave Packet Falling into a Black-Hole-Like Potential
# Corrected & Stable Version (No FFT wrap-around artifacts)
# ==========================================================

# --------------------------
# Physical Parameters
# --------------------------
rs = 2.0                  # Schwarzschild radius
L = 100.0                 # Box size
N = 4096                  # Grid points

r_start = rs + 0.05

# endpoint=False matches FFT periodicity requirements
r = np.linspace(r_start, r_start + L, N, endpoint=False)
dr = r[1] - r[0]

# Momentum grid
k = 2 * np.pi * np.fft.fftfreq(N, d=dr)

# --------------------------
# Time Parameters
# --------------------------
dt = 0.005
total_steps = 4000
plot_interval = 20

# --------------------------
# Paczynski-Wiita Potential
# --------------------------
# Small regularization instead of hard clipping
eps = 0.01
V = -1.0 / (r - rs + eps)

# --------------------------
# Smooth Absorbing Layer
# --------------------------
W = np.zeros_like(r)

# Left absorber (near horizon)
absorb_width_left = 5.0

mask_left = r < (rs + absorb_width_left)

x_left = (
    (rs + absorb_width_left - r[mask_left])
    / absorb_width_left
)

W[mask_left] = 60.0 * x_left**2

# Right absorber (prevents FFT wrap-around)
absorb_width_right = 15.0

mask_right = r > (
    r_start + L - absorb_width_right
)

x_right = (
    r[mask_right]
    - (r_start + L - absorb_width_right)
) / absorb_width_right

W[mask_right] = 60.0 * x_right**2

# --------------------------
# Initial Wave Packet
# --------------------------
r0 = 30.0
sigma = 2.0
p0 = -2.5

psi = (
    np.exp(
        -(r - r0)**2 / (4 * sigma**2)
    )
    * np.exp(1j * p0 * r)
)

# Normalize
norm0 = np.sum(np.abs(psi)**2) * dr
psi /= np.sqrt(norm0)

print(
    f"Initial probability: "
    f"{np.sum(np.abs(psi)**2)*dr:.12f}"
)

# --------------------------
# Split Operator Factors
# --------------------------
V_eff = V - 1j * W

U_V = np.exp(
    -1j * V_eff * dt / 2
)

kinetic = 0.5 * k**2

U_K = np.exp(
    -1j * kinetic * dt
)

# --------------------------
# Visualization Setup
# --------------------------
plt.ion()

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(10, 8),
    gridspec_kw={"height_ratios": [3, 1]}
)

# Probability density
density_line, = ax1.plot(
    r,
    np.abs(psi)**2,
    lw=2,
    color="blue",
    label=r"$|\psi|^2$"
)

# Potential profile
ax1.plot(
    r,
    0.05 * V,
    "r--",
    alpha=0.6,
    label="0.05 × Potential"
)

ax1.axvline(
    rs,
    color="black",
    linestyle=":",
    label="Event Horizon"
)

# Keeping your original graph settings
ax1.set_xlim(0, 60)
ax1.set_ylim(-0.3, 0.4)

ax1.set_ylabel("Amplitude")
ax1.set_title(
    "Quantum Particle Approaching a Black Hole (Corrected FFT)"
)

ax1.legend()

# Probability history
prob_line, = ax2.plot(
    [],
    [],
    color="darkgreen",
    lw=2
)

ax2.set_xlim(
    0,
    total_steps * dt
)

ax2.set_ylim(
    0,
    1.05
)

ax2.set_xlabel("Time")
ax2.set_ylabel("Total Probability")

# --------------------------
# Storage
# --------------------------
times = []
probs = []

# --------------------------
# Main Evolution Loop
# --------------------------
for step in range(total_steps):

    # First half potential step
    psi *= U_V

    # Momentum space
    psi_k = np.fft.fft(psi)

    # Kinetic step
    psi_k *= U_K

    # Back to position space
    psi = np.fft.ifft(psi_k)

    # Second half potential step
    psi *= U_V

    # Surviving probability
    P = np.sum(
        np.abs(psi)**2
    ) * dr

    if np.isnan(P):
        raise RuntimeError(
            "Simulation became unstable."
        )

    times.append(step * dt)
    probs.append(P)

    # Visualization update
    if step % plot_interval == 0:

        density_line.set_ydata(
            np.abs(psi)**2
        )

        prob_line.set_data(
            times,
            probs
        )

        ax2.set_xlim(
            0,
            max(1, times[-1])
        )

        fig.canvas.draw_idle()
        plt.pause(0.001)

# --------------------------
# Finish
# --------------------------
plt.ioff()
plt.show()

print(
    f"\nFinal surviving probability: "
    f"{P:.6f}"
)

print(
    f"Captured probability: "
    f"{1-P:.6f}"
)
