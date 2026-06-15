import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ============================================================
# Schwarzschild Regge-Wheeler Wave Evolution
# ============================================================

# ----------------------------
# Physical parameters
# ----------------------------
M = 1.0
l = 2

# ----------------------------
# Tortoise coordinate domain
# ----------------------------
rstar_min = -100.0
rstar_max = 200.0

N = 4000

rstar = np.linspace(rstar_min, rstar_max, N)
drs = rstar[1] - rstar[0]

# ----------------------------
# Convert r* -> r numerically
# r* = r + 2M ln(r/(2M)-1)
# ----------------------------

def tortoise(r):
    return r + 2*M*np.log(r/(2*M)-1)

def invert_tortoise(rs):
    r_h = 2*M*(1+1e-10)

    if rs < 20:
        upper = 50
    else:
        upper = rs + 50

    return brentq(
        lambda r: tortoise(r)-rs,
        r_h,
        upper
    )

print("Computing r(r*) map...")
r = np.array([invert_tortoise(x) for x in rstar])

# ----------------------------
# Regge-Wheeler potential
# ----------------------------
f = 1 - 2*M/r

V = f * (
    l*(l+1)/r**2 +
    2*M/r**3
)

# ----------------------------
# Absorbing layers
# ----------------------------
W = np.zeros_like(rstar)

width = 20

left = rstar < rstar_min + width
x = (rstar_min + width - rstar[left]) / width
W[left] = 0.02*x**2

right = rstar > rstar_max - width
x = (rstar[right] - (rstar_max-width)) / width
W[right] = 0.02*x**2

# ----------------------------
# Initial Gaussian packet
# ----------------------------
x0 = 80.0
sigma = 8.0
k0 = -0.5

psi0 = np.exp(
    -(rstar-x0)**2/(2*sigma**2)
) * np.cos(k0*rstar)

# Initial time derivative
pi0 = (
    -k0 *
    np.exp(-(rstar-x0)**2/(2*sigma**2))
    * np.sin(k0*rstar)
)

# ----------------------------
# Time stepping
# ----------------------------
dt = 0.4*drs

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
    lw=1
)

ax.plot(
    rstar,
    10*V/np.max(V),
    'r--',
    alpha=0.6,
    label="Potential"
)

ax.set_xlim(-100,150)
ax.set_ylim(-1.5,1.5)

ax.set_xlabel(r"$r_*$")
ax.set_ylabel(r"$\Psi$")
ax.legend()

# ----------------------------
# Evolution
# ----------------------------
steps = 8000

for n in range(steps):

    lap = (
        np.roll(psi,-1)
        -2*psi
        +np.roll(psi,1)
    )/drs**2

    psi_new = (
        2*psi
        - psi_old
        + dt**2*(lap - V*psi)
    )

    # absorber
    psi_new *= np.exp(-W)

    psi_old = psi
    psi = psi_new

    if n % 20 == 0:

        line.set_ydata(psi)

        ax.set_title(
            f"t = {n*dt:.2f}"
        )

        plt.pause(0.001)

plt.ioff()
plt.show()
