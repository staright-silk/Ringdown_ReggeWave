import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Schwarzschild Regge-Wheeler Wave Evolution
# ============================================================

# ----------------------------
# Physical Parameters
# ----------------------------
M = 1.0
l = 2

# ----------------------------
# Tortoise Coordinate Grid
# ----------------------------
rstar_min = -100.0
rstar_max = 200.0

N = 4000

rstar = np.linspace(rstar_min, rstar_max, N)
drs = rstar[1] - rstar[0]

# ----------------------------
# Schwarzschild Tortoise Coordinate
# ----------------------------
def tortoise(r):
    return r + 2*M*np.log(r/(2*M)-1)

# ----------------------------
# Fast Newton Inversion
# ----------------------------
def invert_tortoise(rs):

    if rs > 0:
        r = rs + 2*M
    else:
        r = 2*M + np.exp(rs/(2*M))

    for _ in range(20):

        f = (
            r
            + 2*M*np.log(r/(2*M)-1)
            - rs
        )

        fp = (
            1
            + 2*M/(r-2*M)
        )

        dr = f/fp
        r -= dr

        if abs(dr) < 1e-12:
            break

    return r

print("Computing r(r*) mapping...")

r = np.array([invert_tortoise(x) for x in rstar])

# ----------------------------
# Regge-Wheeler Potential
# ----------------------------
f = 1 - 2*M/r

V = f * (
    l*(l+1)/r**2
    + 2*M/r**3
)

# ----------------------------
# Absorbing Layers
# ----------------------------
W = np.zeros_like(rstar)

width = 20.0
strength = 0.02

left = rstar < rstar_min + width
x = (rstar_min + width - rstar[left]) / width
W[left] = strength*x**2

right = rstar > rstar_max - width
x = (rstar[right] - (rstar_max-width)) / width
W[right] = strength*x**2

# ----------------------------
# Initial Gaussian Packet
# ----------------------------
x0 = 80.0
sigma = 8.0
k0 = -0.5

psi0 = (
    np.exp(
        -(rstar-x0)**2/(2*sigma**2)
    )
    * np.cos(k0*rstar)
)

pi0 = (
    -k0
    * np.exp(
        -(rstar-x0)**2/(2*sigma**2)
    )
    * np.sin(k0*rstar)
)

# ----------------------------
# Stable Time Step
# ----------------------------
dt = 0.25*drs

psi_old = psi0 - dt*pi0
psi = psi0.copy()

# ----------------------------
# Visualization
# ----------------------------
plt.ion()

fig, ax = plt.subplots(figsize=(10,6))

line, = ax.plot(
    rstar,
    psi,
    lw=1.5,
    color="blue",
    label=r"$\Psi$"
)

ax.plot(
    rstar,
    5*V/np.max(V),
    "r--",
    alpha=0.7,
    label="Scaled Regge-Wheeler Potential"
)

ax.set_xlim(-100,150)
ax.set_ylim(-1.5,1.5)

ax.set_xlabel(r"$r_*$")
ax.set_ylabel(r"$\Psi$")
ax.set_title("Regge-Wheeler Wave Scattering")

ax.legend()

# ----------------------------
# Evolution Parameters
# ----------------------------
steps = 8000

# ----------------------------
# Main Evolution Loop
# ----------------------------
for n in range(steps):

    # ------------------------
    # 4th-order Laplacian
    # ------------------------
    lap = np.zeros_like(psi)

    lap[2:-2] = (
        -psi[4:]
        +16*psi[3:-1]
        -30*psi[2:-2]
        +16*psi[1:-3]
        -psi[:-4]
    )/(12*drs**2)

    # Edge treatment
    lap[0]  = lap[2]
    lap[1]  = lap[2]
    lap[-1] = lap[-3]
    lap[-2] = lap[-3]

    # ------------------------
    # Wave Equation Update
    # ------------------------
    psi_new = (
        2*psi
        - psi_old
        + dt**2*(lap - V*psi)
    )

    # ------------------------
    # Absorber
    # ------------------------
    psi_new *= np.exp(-W)

    # Optional hard edge damping
    psi_new[0] *= 0.95
    psi_new[-1] *= 0.95

    # Advance
    psi_old = psi
    psi = psi_new

    # ------------------------
    # Visualization
    # ------------------------
    if n % 20 == 0:

        line.set_ydata(psi)

        ax.set_title(
            f"Regge-Wheeler Evolution   t = {n*dt:.2f}"
        )

        plt.pause(0.001)

plt.ioff()
plt.show()

# ----------------------------
# Final Diagnostics
# ----------------------------
energy_density = (
    0.5*np.gradient(psi, drs)**2
    + 0.5*V*psi**2
)

energy = np.trapz(
    energy_density,
    rstar
)

print()
print("Simulation finished.")
print(f"Approximate final energy: {energy:.6e}")
