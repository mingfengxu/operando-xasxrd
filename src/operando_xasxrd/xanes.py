"""Scalar descriptors of normalised K-edge XANES, computed the same way for every spectrum.

* ``edge_half``  energy where normalised mu(E) first crosses 0.5 on the rising edge
                 (linear interpolation; robust, insensitive to white-line shape)
* ``white_line`` maximum of mu(E) above the edge, refined by a quadratic fit of
                 the points around the maximum (sub-grid precision)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def edge_half(energy: np.ndarray, mu: np.ndarray, level: float = 0.5,
              window=(None, None)) -> float:
    e, m = _window(energy, mu, window)
    above = np.nonzero(m >= level)[0]
    if len(above) == 0 or above[0] == 0:
        return np.nan
    i = above[0]
    return float(e[i - 1] + (level - m[i - 1]) * (e[i] - e[i - 1]) / (m[i] - m[i - 1]))


def white_line(energy: np.ndarray, mu: np.ndarray, window=(None, None), half_width: float = 2.0) -> float:
    e, m = _window(energy, mu, window)
    if len(e) < 3:
        return np.nan
    i = int(np.argmax(m))
    sel = np.abs(e - e[i]) <= half_width
    if sel.sum() < 3:
        return float(e[i])
    c2, c1, _ = np.polyfit(e[sel] - e[i], m[sel], 2)
    if c2 >= 0:
        return float(e[i])
    return float(e[i] - c1 / (2 * c2))


def _window(energy, mu, window):
    e = np.asarray(energy, float)
    m = np.asarray(mu, float)
    lo, hi = window
    sel = np.ones_like(e, bool)
    if lo is not None:
        sel &= e >= lo
    if hi is not None:
        sel &= e <= hi
    return e[sel], m[sel]


def describe(spectra: pd.DataFrame, edge_window=(8330, 8360), wl_window=(8340, 8365)) -> pd.DataFrame:
    """Descriptors for every column of an ATHENA multi-column export (Ni K-edge defaults)."""
    e = spectra.index.to_numpy()
    rows = []
    for name in spectra.columns:
        mu = spectra[name].to_numpy()
        rows.append({"name": name,
                     "edge_half": edge_half(e, mu, window=edge_window),
                     "white_line": white_line(e, mu, window=wl_window)})
    return pd.DataFrame(rows)
