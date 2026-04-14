"""
Tidal Basin Shallow Water Equation Solver

1D numerical simulation of tidal propagation in a shallow basin or estuary.

Solves the depth-averaged shallow water equations on a staggered grid using
the explicit finite-difference scheme of Speer (1984), as described in
Friedrichs (2011), Chapter 3 in Contemporary Issues in Estuarine Physics.

The model simulates how an M2 tidal forcing at the seaward boundary propagates
into a basin, generating tidal distortion through nonlinear friction. Harmonic
analysis of the output extracts the amplitudes and phases of M2, M4, and M6
tidal constituents as a function of position, revealing how tidal asymmetry
develops along the basin.

Continuity:   B * dZ/dt + dQ/dx = 0
Momentum:     dQ/dt + d/dx(Q^2/A) = -gA * dZ/dx - Cd * Q|Q| * P / A^2

where:
    Z  = water surface elevation [m]
    Q  = discharge [m^3/s]
    A  = cross-sectional area [m^2]
    B  = channel width [m]
    P  = wetted perimeter [m]
    Cd = drag coefficient [-]

Numerical scheme

Staggered grid (Speer 1984): Z points are offset half a grid step (deltaX/2)
to the right of Q points.

    Q1  Z1  Q2  Z2  ...  ZN  QN+1

Explicit forward-in-time, centred-in-space differencing.
Stability requires Courant number C = sqrt(gH) * deltaT / deltaX < 1.

Outputs:
    water_level_velocity.png   - M2 amplitude of Z and U vs position
    spatial_snapshot.png       - Z and U across basin at a single timestep
    time_series.png            - Z and U over time at the seaward boundary
    harmonic_amplitudes.png    - M2, M4, M6 amplitudes of Z and U vs position
    tidal_phases.png           - M2, M4, M6 phases of Z and U vs position
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from harmfit import harmfit

OUT_DIR = "./outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Parameters

deltaT  = 30       
deltaX  = 500      
Lbasin  = 2e4      
B0      = 1e3      
H0      = 3       
M2amp   = 0.5   
discharge = 0      
Cd      = 5e-3  

# Tidal periods
Tm2 = 12 * 3600 + 25 * 60

# Angular frequencies for harmonic analysis (M2, M4, M6)
wn = np.array([2 * np.pi / Tm2,
               4 * np.pi / Tm2,
               6 * np.pi / Tm2])

# Grid and initialisation

time = np.arange(0, 10 * Tm2 + deltaT, deltaT)   
Nt   = len(time)

x  = np.arange(0, Lbasin + deltaX, deltaX)
Nx = len(x)

B = np.ones(Nx) * B0    
H = np.ones(Nx) * H0    

# State arrays: Z on half-grid, Q on full grid
Z = np.zeros((Nx - 1, Nt))
Q = np.zeros((Nx, Nt))

# Cross-sectional area and wetted perimeter at Q points
A = np.outer(B * H, np.ones(Nt))
P = B.reshape(-1, 1) * np.ones((1, Nt))

# Force arrays for diagnostics
Inertia = np.zeros((Nx, Nt))
PG      = np.zeros((Nx, Nt))
Fric    = np.zeros((Nx, Nt))

# Boundary conditions
Z[0, :] = M2amp * np.sin(2 * np.pi * time / Tm2)
Q[Nx - 1, :] = -discharge                       

# Courant number check
courant = np.sqrt(9.81 * np.max(H)) * deltaT / deltaX
print(f"Courant number = {courant:.3f}  ({'stable' if courant < 1 else 'UNSTABLE — reduce deltaT'})")

# Time integration  (Speer 1984 staggered-grid scheme)

print(f"Running simulation: {Nt} timesteps...")

for pt in range(1, Nt):
    for px in range(2, Nx):
        # Continuity: update water level Z
        Z[px - 1, pt] = (Z[px - 1, pt - 1]
                         - (deltaT / (0.5 * (B[px - 1] + B[px])))
                         * (Q[px, pt - 1] - Q[px - 1, pt - 1]) / deltaX)

        # Update cross-sectional area and wetted perimeter
        A[px - 1, pt] = B[px - 1] * (H[px] + 0.5 * Z[px - 1, pt] + 0.5 * Z[px - 2, pt])
        P[px - 1, pt] = B[px - 1] + 2 * H[px - 1] + Z[px - 1, pt] + Z[px - 2, pt]

        # Momentum: update discharge Q
        # Store force terms for diagnostics
        Inertia[px - 1, pt] = (Q[px - 1, pt] - Q[px - 1, pt - 1]) / deltaT
        PG[px - 1, pt]      = -9.81 * A[px - 1, pt] * (Z[px - 1, pt] - Z[px - 2, pt]) / deltaX
        Fric[px - 1, pt]    = (-Cd * abs(Q[px - 1, pt - 1]) * Q[px - 1, pt - 1]
                                * P[px - 1, pt - 1] / A[px - 1, pt - 1] ** 2)

        Q[px - 1, pt] = (Q[px - 1, pt - 1]
                         - 9.81 * A[px - 1, pt] * (deltaT / deltaX)
                         * (Z[px - 1, pt] - Z[px - 2, pt])
                         - Cd * deltaT * abs(Q[px - 1, pt - 1]) * Q[px - 1, pt - 1]
                         * P[px - 1, pt - 1] / A[px - 1, pt - 1] ** 2)

    # Seaward boundary: Q consistent with Z
    Q[0, pt] = Q[1, pt] + B[0] * deltaX * (Z[0, pt] - Z[0, pt - 1]) / deltaT

print("Simulation complete.")

# Flow velocity
U = Q / A

# Harmonic analysis  (last tidal period only)

print("Running harmonic analysis...")
Nsteps = math.floor(Tm2 / deltaT)

# Output arrays
Z0, ZM2, ZM4, ZM6           = [np.zeros(Nx - 1) for _ in range(4)]
phaseZM2, phaseZM4, phaseZM6 = [np.zeros(Nx - 1) for _ in range(3)]
U0, UM2, UM4, UM6           = [np.zeros(Nx) for _ in range(4)]
phaseUM2, phaseUM4, phaseUM6 = [np.zeros(Nx) for _ in range(3)]


def fit_harmonics(signal, time, wn):
    
    # Fit harmonic constituents to a time series using least-squares optimisation.

    invariant = {"timesteps": time, "wn": wn}
    a0_in = np.mean(signal)
    coefin = np.concatenate(([a0_in], np.ones(len(wn)), np.ones(len(wn))))
    popt, _ = optimize.curve_fit(harmfit, invariant, signal, coefin, maxfev=10000)
    mean_fit = popt[0]
    Cn = popt[1:len(wn) + 1]
    Dn = popt[len(wn) + 1:]
    amplitudes = np.sqrt(Cn ** 2 + Dn ** 2)
    phases = np.arctan2(Dn, Cn)
    return amplitudes, phases, mean_fit


# Analyse water levels
for px in range(Nx - 1):
    amps, phases, mean = fit_harmonics(Z[px, :], time, wn)
    Z0[px] = mean
    ZM2[px], ZM4[px], ZM6[px] = amps
    phaseZM2[px], phaseZM4[px], phaseZM6[px] = phases

# Analyse flow velocities
for px in range(Nx):
    amps, phases, mean = fit_harmonics(U[px, :], time, wn)
    U0[px] = mean
    UM2[px], UM4[px], UM6[px] = amps
    phaseUM2[px], phaseUM4[px], phaseUM6[px] = phases

print("Harmonic analysis complete.")

# Plots

def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=200)
    print(f"Saved: {path}")


# 1. M2 amplitude of water level and velocity vs position
plt.figure(figsize=(8, 4))
plt.plot(x[1:] / 1e3, ZM2, color="blue", label="Water level M2 amplitude [m]")
plt.plot(x / 1e3,     UM2, color="red",  label="Tidal velocity M2 amplitude [m/s]")
plt.xlabel("Distance from mouth [km]")
plt.ylabel("Amplitude [m] / [m/s]")
plt.title("M2 tidal amplitude along the basin")
plt.legend()
plt.tight_layout()
savefig("water_level_velocity.png")
plt.show()

# 2. Spatial snapshot at t=20
plt.figure(figsize=(8, 4))
plt.plot(x[1:] / 1e3, Z[:, 20], color="blue", label="Water level [m]")
plt.plot(x / 1e3,     U[:, 20], color="red",  label="Flow velocity [m/s]")
plt.xlabel("Distance from mouth [km]")
plt.ylabel("Water level [m] / Velocity [m/s]")
plt.title("Water level and velocity along the basin (t = 20 steps)")
plt.legend()
plt.tight_layout()
savefig("spatial_snapshot.png")
plt.show()

# 3. Time series at seaward boundary
plt.figure(figsize=(10, 4))
plt.plot(time / 3600, Z[0, :], color="blue", label="Water level [m]")
plt.plot(time / 3600, U[0, :], color="red",  label="Flow velocity [m/s]")
plt.xlabel("Time [hours]")
plt.ylabel("Water level [m] / Velocity [m/s]")
plt.title("Water level and velocity at the seaward boundary")
plt.legend()
plt.tight_layout()
savefig("time_series.png")
plt.show()

# 4. M2, M4, M6 amplitudes
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (amp2, amp4, amp6, label) in zip(axes, [
    (ZM2, ZM4, ZM6, "Water level amplitude [m]"),
    (UM2, UM4, UM6, "Velocity amplitude [m/s]"),
]):
    xplot = x[1:] / 1e3 if len(amp2) == Nx - 1 else x / 1e3
    ax.plot(xplot, amp2, label="M2")
    ax.plot(xplot, amp4, label="M4")
    ax.plot(xplot, amp6, label="M6")
    ax.set_xlabel("Distance from mouth [km]")
    ax.set_ylabel(label)
    ax.legend()

axes[0].set_title("Water level harmonics")
axes[1].set_title("Velocity harmonics")
plt.suptitle("M2, M4, M6 amplitudes along the basin")
plt.tight_layout()
savefig("harmonic_amplitudes.png")
plt.show()

# 5. Tidal phases
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (ph2, ph4, ph6, label) in zip(axes, [
    (phaseZM2, phaseZM4, phaseZM6, "Phase [rad]"),
    (phaseUM2, phaseUM4, phaseUM6, "Phase [rad]"),
]):
    xplot = x[1:] / 1e3 if len(ph2) == Nx - 1 else x / 1e3
    ax.plot(xplot, ph2, label="M2")
    ax.plot(xplot, ph4, label="M4")
    ax.plot(xplot, ph6, label="M6")
    ax.set_xlabel("Distance from mouth [km]")
    ax.set_ylabel(label)
    ax.legend()

axes[0].set_title("Water level phases")
axes[1].set_title("Velocity phases")
plt.suptitle("M2, M4, M6 phases along the basin")
plt.tight_layout()
savefig("tidal_phases.png")
plt.show()

print(f"\nAll outputs saved in: {os.path.abspath(OUT_DIR)}")
