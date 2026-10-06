"""Put XRD patterns, XAS spectra and the electrochemistry on one clock.

In a semi-simultaneous experiment the cell is moved between techniques, so XRD
and XAS of the same cell are never taken at the same instant. Every scan is
mapped to the electrochemical state at the middle of its acquisition:
Na content, voltage, and which segment of the protocol it belongs to
(rest, constant-current charge/discharge, constant-voltage hold).
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .io import Scan

# EC-Lab "mode": 1 = constant current, 2 = constant voltage, 3 = rest
_MODE = {1: "CC", 2: "CV", 3: "rest"}


def segments(echem: pd.DataFrame) -> np.ndarray:
    """Human-readable protocol segment for every echem row."""
    mode = echem["mode"].astype(int).map(_MODE).to_numpy()
    ox = echem["ox/red"].astype(int).to_numpy()
    seg = np.where(mode == "rest", "rest",
                   np.where(ox == 1, "charge", "discharge"))
    seg = np.where(mode == "CV", np.char.add(seg.astype(str), "-CV"), seg)
    return seg.astype(object)


def state_at(echem: pd.DataFrame, t: Sequence[float], x0: float | None = None) -> pd.DataFrame:
    """Electrochemical state at times ``t`` (epoch s).

    ``dx`` is Na extracted per formula unit since the first echem point
    (``x[0] - x``); ``x`` is ``x0 - dx`` when the true initial Na content
    ``x0`` is known (e.g. from ICP), otherwise EC-Lab's own ``x``.
    """
    t = np.asarray(t, float)
    te = echem["t"].to_numpy()
    outside = (t < te[0]) | (t > te[-1])
    xe = echem["x"].to_numpy()
    dx = np.interp(t, te, xe[0] - xe)
    seg = segments(echem)
    idx = np.clip(np.searchsorted(te, t, side="right") - 1, 0, len(te) - 1)
    hc = (echem["half cycle"].to_numpy()[idx] if "half cycle" in echem else np.zeros(len(t), int)).astype(float)
    V = np.interp(t, te, echem["Ewe/V"].to_numpy())
    dx[outside], V[outside], hc[outside] = np.nan, np.nan, np.nan
    segment = np.where(outside, np.where(t < te[0], "before echem", "after echem"), seg[idx])
    return pd.DataFrame({
        "t": t,
        "hours": (t - te[0]) / 3600.0,
        "dx": dx,
        "x": (x0 if x0 is not None else xe[0]) - dx,
        "V": V,
        "segment": segment,
        "half_cycle": hc,
    })


def align_scans(echem: pd.DataFrame, scans: Iterable[Scan], modality: str,
                x0: float | None = None) -> pd.DataFrame:
    scans = list(scans)
    df = state_at(echem, [s.mid for s in scans], x0)
    df.insert(0, "scan", np.arange(1, len(scans) + 1))
    df.insert(0, "modality", modality)
    df["duration_s"] = [np.nan if s.end is None else s.end - s.start for s in scans]
    return df


def compare_with_table(aligned: pd.DataFrame, table_x: Sequence[float]) -> pd.DataFrame:
    """Check a hand-made x column against the time-stamp alignment.

    Returns the aligned rows next to the table values, the residual after
    removing a constant offset (different initial-x conventions) and a flag
    for points whose protocol segment is not a charge step.
    """
    n = len(table_x)
    a = aligned.iloc[:n].copy()
    a["x_table"] = np.asarray(table_x, float)
    off = np.median(a["x"] - a["x_table"])
    a["resid"] = a["x"] - a["x_table"] - off
    a["flag"] = np.where(a["segment"].str.startswith("discharge"), "on discharge",
                         np.where(np.abs(a["resid"]) > 0.01, "x mismatch", ""))
    a.attrs["offset"] = float(off)
    return a
