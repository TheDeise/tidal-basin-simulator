# Tidal basin simulator — 1D shallow water equations

A numerical model of tidal wave propagation in a shallow closed basin, based on the staggered-grid finite-difference scheme from Speer (1984). The model shows how an M2 tidal signal distorts as it propagates landward, generating M4 and M6 harmonics through nonlinear friction.

## Background

When a tidal wave enters a shallow estuary or basin, friction and changes in cross-sectional area with the tide distort the signal — energy transfers from the dominant M2 constituent to higher harmonics (M4, M6). This asymmetry between flood and ebb has direct implications for sediment transport direction. The model reproduces this process and extracts harmonic amplitudes and phases along the basin.

## Running it

```bash
pip install numpy scipy matplotlib
python main.py
```

Runs 10 M2 tidal cycles (~5 days) at 30-second timesteps, takes about 1–2 minutes. Outputs go to `./outputs/` and include time series, spatial snapshots, and plots of M2/M4/M6 amplitudes and phases along the basin.

The script checks the Courant number on startup — if it's ≥ 1 the scheme will be unstable, so reduce `deltaT`.

## Parameters

Set at the top of `main.py`: basin length (`Lbasin`), depth (`H0`), tidal amplitude (`M2amp`), drag coefficient (`Cd`), and grid resolution (`deltaX`, `deltaT`).

## Dependencies

numpy, scipy, matplotlib

## References

Speer, P.E. (1984). Tidal distortion in shallow estuaries. PhD thesis, WHOI.  
Friedrichs, C.T. (2011). In *Contemporary Issues in Estuarine Physics*, Cambridge University Press.

## Context

Written as part of the MSc Earth Sciences programme at Utrecht University, 2025.
