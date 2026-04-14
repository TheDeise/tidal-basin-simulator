# Tidal Basin Shallow Water Equation Solver

A 1D numerical model of tidal wave propagation in a shallow basin or estuary, solving the depth-averaged shallow water equations on a staggered grid. Harmonic analysis of the output reveals how tidal distortion develops along the basin.

---

## Background

When a tidal wave propagates from the open sea into a shallow basin, nonlinear processes — primarily friction and changes in cross-sectional area with the tide — distort the originally sinusoidal M2 wave. Energy is transferred to higher harmonics (M4, M6), producing asymmetry between flood and ebb. This asymmetry has important implications for sediment transport and estuarine morphology.

This model simulates that process by solving the 1D shallow water equations for a rectangular basin forced by an M2 tide at its seaward boundary. Harmonic analysis of the simulated water levels and velocities extracts the M2, M4, and M6 tidal constituents as a function of position, quantifying how distortion develops landward.

---

## Method

**Governing equations** — depth-averaged continuity and momentum:

```
Continuity:  B * dZ/dt + dQ/dx = 0
Momentum:    dQ/dt = -gA * dZ/dx - Cd * Q|Q| * P / A^2
```

**Numerical scheme** — staggered grid (Speer 1984), explicit forward-in-time differencing. Z (water level) and Q (discharge) points are offset by half a grid step:

```
Q1  Z1  Q2  Z2  ...  ZN  QN+1
```

Stability requires the Courant number `C = sqrt(gH) * deltaT / deltaX < 1`.

**Harmonic analysis** — after 10 tidal cycles (to reach quasi-steady state), `scipy.optimize.curve_fit` fits M2, M4, and M6 sinusoids to the simulated time series at each grid point, extracting amplitudes and phases.

---

## Results

The model saves five figures to `./outputs/`:

| File | Description |
|---|---|
| `water_level_velocity.png` | M2 amplitude of water level and velocity vs distance |
| `spatial_snapshot.png` | Water level and velocity along the basin at one timestep |
| `time_series.png` | Water level and velocity over time at the seaward boundary |
| `harmonic_amplitudes.png` | M2, M4, M6 amplitudes of Z and U vs distance |
| `tidal_phases.png` | M2, M4, M6 phases of Z and U vs distance |

---

## Usage

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run:**
```bash
python main.py
```

The simulation runs 10 M2 tidal cycles (~5 days) at 30-second timesteps, taking around 1–2 minutes.

**Key parameters** (edit at the top of `main.py`):

| Parameter | Default | Description |
|---|---|---|
| `deltaT` | 30 s | Time step — reduce if Courant > 1 |
| `deltaX` | 500 m | Spatial step |
| `Lbasin` | 20,000 m | Basin length |
| `H0` | 10 m | Mean water depth |
| `M2amp` | 1.0 m | Tidal forcing amplitude |
| `Cd` | 2.5 × 10⁻³ | Drag coefficient |

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Array operations and grid setup |
| `scipy` | Harmonic analysis via `curve_fit` |
| `matplotlib` | Plotting |

---

## References

- Speer, P.E. (1984). *Tidal distortion in shallow estuaries*. PhD thesis, Woods Hole Oceanographic Institution.
- Friedrichs, C.T. (2011). Tidal flat morphodynamics. In *Contemporary Issues in Estuarine Physics*, Cambridge University Press.

---

## Project context

Developed as part of the MSc Earth Sciences programme at Utrecht University (Morphodynamics of Tidal Systems, 2025).
