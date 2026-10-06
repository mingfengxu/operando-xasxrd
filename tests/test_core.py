import numpy as np
import pandas as pd
import pytest

from operando_xasxrd import (align_scans, compare_with_table, edge_half, fit_hinge, fit_hinge_batch,
                             merge_reexport, read_athena_columns, read_eclab, read_spec_scans, scan_index,
                             white_line, Scan)


def _echem(t0=0.0):
    t = t0 + np.arange(0, 10 * 3600, 60.0)
    n = len(t)
    half = n // 2
    x = np.r_[np.linspace(0.8, 0.3, half), np.linspace(0.3, 0.7, n - half)]
    return pd.DataFrame({"t": t, "x": x, "Ewe/V": 3 + (0.8 - x),
                         "mode": 1, "ox/red": np.r_[np.ones(half, int), np.zeros(n - half, int)],
                         "half cycle": np.r_[np.zeros(half, int), np.ones(n - half, int)]})


def test_hinge_recovers_break():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 0.5, 14)
    y = 8343 + np.where(x < 0.27, 10 * x, 2.7 + 0.5 * (x - 0.27)) + rng.normal(0, 0.03, x.size)
    r = fit_hinge(x, y, n_boot=300)
    assert abs(r.xb - 0.27) < 0.02
    assert r.xb_ci[0] <= r.xb <= r.xb_ci[1]
    assert r.delta_bic > 6
    assert abs(r.k1 - 10) < 1.0


def test_hinge_batch_shape_and_straight_line():
    x = np.linspace(0, 1, 12)
    Y = np.stack([2 * x + 1, -x + 3])
    out = fit_hinge_batch(x, Y)
    assert out["xb"].shape == (2,)
    assert np.all(out["sse"] < 1e-6)


def test_alignment_segments_and_x():
    e = _echem()
    scans = [Scan(i, t, t + 120, "") for i, t in enumerate([600.0, 2 * 3600.0, 8 * 3600.0, 11 * 3600.0], 1)]
    a = align_scans(e, scans, "XAS")
    assert list(a["segment"][:3]) == ["charge", "charge", "discharge"]
    assert a["segment"].iloc[3] == "after echem" and np.isnan(a["dx"].iloc[3])
    assert a["dx"].iloc[0] == pytest.approx(0.5 * 660 / (5 * 3600), abs=2e-3)


def test_compare_with_table_flags_discharge():
    e = _echem()
    scans = [Scan(i, t, t, "") for i, t in enumerate([1800.0, 3600.0, 5400.0, 7 * 3600.0], 1)]
    a = align_scans(e, scans, "XAS")
    table = (a["x"] - 0.1).to_numpy().copy()
    table[3] = table[2] - 0.05          # extrapolated as if still charging
    chk = compare_with_table(a, table)
    assert chk.attrs["offset"] == pytest.approx(0.1)
    assert list(chk["flag"]) == ["", "", "", "on discharge"]


def test_merge_reexport(tmp_path):
    rows = "\n".join(f"07/06/2023 17:38:{s:02d}.5000\t{2.8 + s * 1e-3:.7E}" for s in range(5))
    (tmp_path / "o.txt").write_text("time/s\tEwe/V\t\n" + rows.replace("\n", "\t\n") + "\t\n")
    rows2 = "\n".join(f"09/26/2023 11:28:{s:02d}.50\t{2.8 + s * 1e-3:.7E}\t{0.8 - s * 1e-3}" for s in range(5))
    (tmp_path / "f.mpt").write_text("time/s\tEwe/V\tx\n" + rows2 + "\n")
    m = merge_reexport(read_eclab(tmp_path / "f.mpt"), read_eclab(tmp_path / "o.txt"))
    assert m["t"].iloc[0] == read_eclab(tmp_path / "o.txt")["t"].iloc[0]
    assert m.attrs["clock_shift_s"] < -80 * 86400
    bad = read_eclab(tmp_path / "o.txt").assign(**{"Ewe/V": 0.0})
    with pytest.raises(ValueError):
        merge_reexport(read_eclab(tmp_path / "f.mpt"), bad)


def test_spec_and_athena_readers(tmp_path):
    (tmp_path / "s.dat").write_text(
        "#S 1 qExafsc 8240\n#D Thu Jul 06 18:09:57 2023\n#L e mu\n1 2\n#C Acquisition ended at Thu Jul  6 18:12:36 2023\n"
        "#S 2 qExafsc 8240\n#D Thu Jul 06 18:47:31 2023\n1 2\n")
    s = read_spec_scans(tmp_path / "s.dat")
    assert [x.number for x in s] == [1, 2]
    assert s[0].end - s[0].start == 159 and s[1].end is None
    (tmp_path / "n.txt").write_text("# Column.1: energy eV\n# Column.2: c_NiX_0001.xy\n# Column.3: NiO_0001.xy\n"
                                    "8330 0.0 0.0\n8340 1.0 0.9\n")
    d = read_athena_columns(tmp_path / "n.txt")
    assert list(d.columns) == ["c_NiX_0001.xy", "NiO_0001.xy"]
    assert scan_index("31NNMO_0_NiX_0012.xy") == 12 and scan_index("NiOnew_0001.xy") is None


def test_edge_descriptors():
    e = np.arange(8320, 8380, 0.3)
    mu = 1 / (1 + np.exp(-(e - 8345) / 1.5)) + 0.4 * np.exp(-0.5 * ((e - 8352.3) / 2.5) ** 2)
    assert edge_half(e, mu) == pytest.approx(8345, abs=0.15)
    assert white_line(e, mu) == pytest.approx(8352.6, abs=0.4)


def test_threshold_onset_and_step_dependence():
    from operando_xasxrd import threshold_onset
    x = np.linspace(0, 0.5, 11)                       # step 0.05
    y = np.where(x < 0.27, 10 * x, 2.7 + 1.5 * (x - 0.27))
    xo, i = threshold_onset(x, y, tol=0.1)
    assert xo == pytest.approx(0.35) and i == 7       # first point after which steps are small
    x2 = np.linspace(0, 0.5, 6)                       # coarser sampling -> later onset
    y2 = np.where(x2 < 0.27, 10 * x2, 2.7 + 1.5 * (x2 - 0.27))
    assert np.isnan(threshold_onset(x2, y2, tol=0.1)[0])
