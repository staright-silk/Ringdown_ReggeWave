"""
Stage 1 Regge–Wheeler Solver (Laptop-Friendly Research Style)

Features
--------
- Schwarzschild Regge–Wheeler potential
- Tortoise-coordinate grid
- 4th-order spatial finite differences
- CFL-controlled leapfrog evolution
- Absorbing boundary layers
- Energy diagnostic
- Horizon / infinity flux estimates
- Ringdown signal recording
- CSV export
- Optional live animation

Requirements:
    numpy, scipy, matplotlib, pandas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# =========================
# Physical Parameters
# =========================
M = 1.0
ELL = 2

# =========================
# Grid
# =========================
RSTAR_MIN = -100.0
RSTAR_MAX = 250.0
N = 3000

rstar = np.linspace(RSTAR_MIN, RSTAR_MAX, N)
dx = rstar[1] - rstar[0]

# =========================
# Coordinate Mapping
# =========================
def tortoise(r):
    return r + 2*M*np.log(r/(2*M)-1)

def invert_tortoise(rs):
    r_h = 2*M*(1+1e-12)
    upper = max(50.0, rs + 100.0)
    return brentq(lambda r: tortoise(r)-rs, r_h, upper)

print("Building r(r*) mapping...")
r = np.array([invert_tortoise(x) for x in rstar])

# =========================
# Regge-Wheeler Potential
# =========================
f = 1 - 2*M/r
V = f * (ELL*(ELL+1)/r**2 + 2*M/r**3)

# =========================
# Absorbing Layer
# =========================
W = np.zeros_like(rstar)

width = 25.0
strength = 0.02

left = rstar < (RSTAR_MIN + width)
x = (RSTAR_MIN + width - rstar[left])/width
W[left] = strength*x**3

right = rstar > (RSTAR_MAX - width)
x = (rstar[right] - (RSTAR_MAX-width))/width
W[right] = strength*x**3

# =========================
# Initial Data
# =========================
x0 = 80.0
sigma = 10.0
k0 = -0.6

psi0 = np.exp(-(rstar-x0)**2/(2*sigma**2))*np.cos(k0*rstar)
pi0 = -k0*np.exp(-(rstar-x0)**2/(2*sigma**2))*np.sin(k0*rstar)

# =========================
# CFL Time Step
# =========================
CFL = 0.3
dt = CFL*dx

psi_old = psi0 - dt*pi0
psi = psi0.copy()

# =========================
# Diagnostics
# =========================
times = []
energies = []
flux_horizon = []
flux_infinity = []
ringdown = []

obs_index = np.argmin(np.abs(rstar - 50.0))
horizon_index = 20
infinity_index = N-21

# =========================
# 4th Order Laplacian
# =========================
def lap4(u, dx):
    out = np.zeros_like(u)

    out[2:-2] = (
        -u[4:] + 16*u[3:-1]
        -30*u[2:-2]
        +16*u[1:-3]
        -u[:-4]
    )/(12*dx*dx)

    out[:2] = (np.roll(u,-1)[:2]-2*u[:2]+np.roll(u,1)[:2])/dx**2
    out[-2:] = (np.roll(u,-1)[-2:]-2*u[-2:]+np.roll(u,1)[-2:])/dx**2

    return out

# =========================
# Animation
# =========================
ANIMATE = True

if ANIMATE:
    plt.ion()
    fig, ax = plt.subplots(figsize=(10,5))
    line, = ax.plot(rstar, psi, lw=1)
    ax.plot(rstar, 5*V/np.max(V), "--")
    ax.set_xlim(-100, 150)
    ax.set_ylim(-1.5, 1.5)

# =========================
# Evolution
# =========================
STEPS = 12000

for n in range(STEPS):

    lap = lap4(psi, dx)

    psi_new = (
        2*psi
        - psi_old
        + dt**2*(lap - V*psi)
    )

    psi_new *= np.exp(-W)

    pi = (psi_new - psi_old)/(2*dt)

    grad = np.gradient(psi, dx)

    energy_density = 0.5*(pi**2 + grad**2 + V*psi**2)
    energy = np.trapz(energy_density, rstar)

    fluxL = -(pi[horizon_index]*grad[horizon_index])
    fluxR = +(pi[infinity_index]*grad[infinity_index])

    times.append(n*dt)
    energies.append(energy)
    flux_horizon.append(fluxL)
    flux_infinity.append(fluxR)
    ringdown.append(psi[obs_index])

    psi_old = psi
    psi = psi_new

    if ANIMATE and n % 20 == 0:
        line.set_ydata(psi)
        ax.set_title(f"t = {n*dt:.2f}")
        plt.pause(0.001)

if ANIMATE:
    plt.ioff()
    plt.show()

# =========================
# Export Data
# =========================
df = pd.DataFrame({
    "time": times,
    "energy": energies,
    "flux_horizon": flux_horizon,
    "flux_infinity": flux_infinity,
    "ringdown": ringdown
})

df.to_csv("regge_wheeler_output.csv", index=False)

print("Saved: regge_wheeler_output.csv")
print("Done.")
