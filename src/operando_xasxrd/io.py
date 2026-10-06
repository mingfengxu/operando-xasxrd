"""Readers for the files an ALBA NOTOS operando XRD/XAS experiment leaves behind.

* EC-Lab exports (``.mpt`` with the full column set, ``.txt`` with time/Ewe only)
* SPEC-style ``.dat`` files (one or many ``#S`` scans; start/end time in the header)
* ATHENA multi-column exports (normalised mu(E), white-line tables)
* FullProf sequential-refinement summaries collected in Excel
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_ECLAB_TIME = "%m/%d/%Y %H:%M:%S.%f"


def _epoch(series) -> np.ndarray:
    """datetime-like -> seconds since 1970 (float)."""
    return pd.to_datetime(series).values.astype("datetime64[ms]").astype(np.int64) / 1000.0


def read_eclab(path) -> pd.DataFrame:
    """Read an EC-Lab ASCII export (``.mpt`` or ``.txt``, with or without header block).

    Adds ``t`` (epoch seconds) parsed from the absolute ``time/s`` column.
    """
    lines = Path(path).read_text(encoding="latin-1").splitlines()
    if lines and lines[0].startswith("EC-Lab"):
        n = int(re.search(r"Nb header lines\s*:\s*(\d+)", lines[1]).group(1))
        lines = lines[n - 1:]
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep="\t", encoding="latin-1")
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    df["t"] = _epoch(pd.to_datetime(df["time/s"], format=_ECLAB_TIME))
    return df


def merge_reexport(full: pd.DataFrame, original: pd.DataFrame, tol: float = 1e-6) -> pd.DataFrame:
    """Give a re-processed EC-Lab export the time stamps of the original recording.

    Re-exporting a ``.mpr`` after adding a derived column (e.g. ``x``) can
    re-date the absolute time column to the export date. The rows are the same,
    so the original ``.txt`` (time, Ewe) is used to restore the clock; the
    potentials must agree row by row, otherwise a ``ValueError`` is raised.
    """
    if len(full) != len(original):
        raise ValueError(f"row count differs: {len(full)} vs {len(original)}")
    dev = np.abs(full["Ewe/V"].to_numpy() - original["Ewe/V"].to_numpy()).max()
    if dev > tol:
        raise ValueError(f"Ewe differs by up to {dev:.2e} V; not the same recording")
    out = full.copy()
    shift = original["t"].to_numpy() - full["t"].to_numpy()
    if np.ptp(shift) > 2.0:   # exports round fractional seconds differently
        raise ValueError("time offset is not constant between the two exports")
    out["t"] = original["t"].to_numpy()
    out.attrs["clock_shift_s"] = float(np.median(shift))
    return out


@dataclass
class Scan:
    number: int
    start: float          # epoch s
    end: Optional[float]  # epoch s (None if not recorded)
    command: str

    @property
    def mid(self) -> float:
        return self.start if self.end is None else 0.5 * (self.start + self.end)


def _spec_time(s: str) -> float:
    # naive local beamline time, treated like the (also naive) EC-Lab clock
    t = datetime.strptime(" ".join(s.split()), "%a %b %d %H:%M:%S %Y")
    return (t - datetime(1970, 1, 1)).total_seconds()


def read_spec_scans(path) -> List[Scan]:
    """Scan numbers with start/end times from a SPEC/Sardana ``.dat`` file.

    Only header lines are parsed, so this is fast even for large files.
    """
    scans: List[Scan] = []
    cur = None
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            if not line.startswith("#"):
                continue
            if line.startswith("#S "):
                parts = line.split(maxsplit=2)
                cur = {"number": int(parts[1]), "command": parts[2].strip() if len(parts) > 2 else "",
                       "start": None, "end": None}
                scans.append(cur)
            elif cur is not None and line.startswith("#D ") and cur["start"] is None:
                cur["start"] = _spec_time(line[3:])
            elif cur is not None and line.startswith("#C Acquisition ended at"):
                cur["end"] = _spec_time(line.split(" at ", 1)[1])
    return [Scan(s["number"], s["start"], s["end"], s["command"]) for s in scans if s["start"] is not None]


def read_athena_columns(path) -> pd.DataFrame:
    """ATHENA multi-column export -> DataFrame indexed by energy, one column per group."""
    names, rows = [], []
    for line in Path(path).read_text(encoding="latin-1").splitlines():
        m = re.match(r"#\s*Column\.(\d+):\s*(.+)", line)
        if m:
            names.append(m.group(2).strip())
        elif line.strip() and not line.startswith("#"):
            rows.append([float(v) for v in line.split()])
    arr = np.asarray(rows)
    return pd.DataFrame(arr[:, 1:], index=pd.Index(arr[:, 0], name="energy"), columns=names[1:arr.shape[1]])


def read_athena_table(path) -> pd.Series:
    """Two-column ATHENA report (e.g. white-line positions): group name -> value."""
    out = {}
    for line in Path(path).read_text(encoding="latin-1").splitlines():
        m = re.match(r'\s*"([^"]+)"\s+([-+0-9.eE]+)', line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return pd.Series(out)


def scan_index(name: str) -> Optional[int]:
    """'31NNMO_0_NiX_0012.xy' -> 12 (None for references such as 'NiOnew_0001.xy')."""
    m = re.search(r"_[A-Z][a-z]?X_(\d+)", name)
    return int(m.group(1)) if m else None


def read_refinement_summary(path, sheet: str = "sum") -> pd.DataFrame:
    """FullProf sequential results collected in an Excel 'sum' sheet.

    Expected layout: index | file name | a | esd | c | esd | ... ; empty rows
    (patterns that were not refined) are returned as NaN.
    """
    df = pd.read_excel(path, sheet_name=sheet, header=0)
    cols = list(df.columns)
    out = pd.DataFrame({
        "pattern": df[cols[0]],
        "file": df[cols[1]],
        "a": pd.to_numeric(df[cols[2]], errors="coerce"),
        "a_esd": pd.to_numeric(df[cols[3]], errors="coerce"),
        "c": pd.to_numeric(df[cols[4]], errors="coerce"),
        "c_esd": pd.to_numeric(df[cols[5]], errors="coerce"),
    })
    out = out[out["pattern"].notna()].copy()
    out["pattern"] = out["pattern"].astype(int)
    return out.reset_index(drop=True)
