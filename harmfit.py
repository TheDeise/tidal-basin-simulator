"""
harmfit.py
----------
Harmonic fitting function for tidal analysis.

Fits a mean plus a sum of sine and cosine terms to a time series:

    h(t) = a0 + sum_k [ Cn_k * sin(wn_k * t) + Dn_k * cos(wn_k * t) ]

Designed to work with scipy.optimize.curve_fit via a dictionary of
independent variables (timesteps and angular frequencies).
"""

import numpy as np


def harmfit(invariant, *args):
    """
    Evaluate a harmonic model at given timesteps.

    Parameters
    ----------
    invariant : dict
        Must contain:
            'timesteps' : array-like, shape (Nt,) — time vector [s or hours]
            'wn'        : array-like, shape (Nf,) — angular frequencies [rad/s or rad/hr]
    *args : float or array
        Model parameters in the order [a0, Cn_1, ..., Cn_Nf, Dn_1, ..., Dn_Nf].
        When called via scipy.optimize.curve_fit, args is a single array.

    Returns
    -------
    h : ndarray, shape (Nt,)
        Modelled time series.
    """
    if len(args) == 1:
        args = tuple(args[0])

    t  = np.asarray(invariant["timesteps"])
    wn = np.asarray(invariant["wn"])

    a0 = args[0]
    rest = np.array(args[1:])
    Cn, Dn = np.split(rest, 2)

    h = np.full(len(t), a0, dtype=float)
    for k in range(len(Cn)):
        h += Cn[k] * np.sin(wn[k] * t) + Dn[k] * np.cos(wn[k] * t)

    return h
