"""Semi-simultaneous operando XRD + Ni K-edge XANES of Li-substituted P2 cathodes (ALBA NOTOS, 2023).

    python examples/afm_li_p2.py --root ~/Work/operando_XRD_XANES --out results

``--root`` is the experiment folder as it was left after the beamtime
(EC-Lab exports, SPEC .dat files, ATHENA exports, FullProf summaries); the
relative paths are listed in CELLS below. Outputs: one aligned table per cell,
a break-point summary (CSV/JSON) and an overview figure.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from operando_xasxrd import (align_scans, threshold_onset, compare_with_table, describe, fit_hinge, merge_reexport,
                             read_athena_columns, read_eclab, read_refinement_summary, read_spec_scans,
                             scan_index)

A = "NOTOS Analysing/"
CELLS = {
    "NM":    dict(run="NNMO_0", tag="31NNMO_0", sr="SR0", xas="0",
                  mpt=A + "XRD/NNMO_0/31NNMO_0_C01_Ix.mpt", txt=A + "XRD/NNMO_0/31NNMO_0_C01.txt"),
    "LNM-2": dict(run="NLNMO_2", tag="32NLNMO_2", sr="SR2", xas="2",
                  mpt=A + "Echem/32NLNMO_2_C02.mpt", txt=A + "XRD/NLNMO_2/32NLNMO_2_C02.txt"),
    "LNM-3": dict(run="NLNMO_3", tag="13NLNMO_3", sr="SR3", xas="3",
                  mpt=A + "Echem/NLNMO_3_C03.mpt", txt=A + "XRD/NLNMO_3/NLNMO_3_C03.txt"),
    "LNM-4": dict(run="NLNMO_4", tag="14NLNMO_4", sr="SR4", xas="4",
                  mpt=A + "Echem/NLNMO_4_C04.mpt", txt=A + "XRD/NLNMO_4/NLNMO_4_C04.txt"),
}
# Compositions from Rietveld refinement (P63/mmc), Table S11 of the paper's SI.
COMPOSITION = {
    "NM":    dict(Na=0.71,  Li=0.0,   Ni=0.334, Mn=0.666),
    "LNM-2": dict(Na=0.745, Li=0.164, Ni=0.238, Mn=0.599),
    "LNM-3": dict(Na=0.748, Li=0.189, Ni=0.172, Mn=0.638),
    "LNM-4": dict(Na=0.75,  Li=0.222, Ni=0.081, Mn=0.697),
}
# Optional: an x column made by hand for the XAS scans (one value per scan, from scan 1),
# checked against the time-stamp alignment. Put it in a CSV with columns cell,scan,x.
SEGMENTS_CC = ("charge",)
SEGMENTS_CC_CV = ("charge", "charge-CV")


def load_cell(root, c, x0=None):
    p = lambda rel: os.path.join(root, rel)
    echem = merge_reexport(read_eclab(p(c["mpt"])), read_eclab(p(c["txt"])))
    xas = align_scans(echem, read_spec_scans(p(f"{A}XRD/{c['run']}/IS{c['tag']}xas/{c['tag']}_NiX.dat")), "XAS", x0)
    frames = sorted(glob.glob(p(f"{A}XRD/{c['run']}/IS{c['tag']}pd/{c['tag']}_*.dat")))
    xrd = align_scans(echem, [read_spec_scans(f)[0] for f in frames], "XRD", x0)
    d = describe(read_athena_columns(p(f"{A}XAS/Ni XAS/{c['xas']}_nor.txt")))
    d["scan"] = d["name"].map(scan_index)
    xas = xas.merge(d.dropna(subset=["scan"]).astype({"scan": int})[["scan", "edge_half", "white_line"]],
                    on="scan", how="left")
    ref = read_refinement_summary(p(f"SR/{c['sr']}/{c['sr']}-sum.xlsx"))
    xrd = xrd.merge(ref[["pattern", "a", "a_esd", "c", "c_esd"]].rename(columns={"pattern": "scan"}),
                    on="scan", how="left")
    return echem, xas, xrd


def first_charge(df, include_cv=False):
    """First-charge scans; by default constant-current only (the 4.5 V CV-hold scan is left out)."""
    seg = SEGMENTS_CC_CV if include_cv else SEGMENTS_CC
    return df[df["segment"].isin(seg) & (df["half_cycle"] == 0)]


def branch_gap(ch, dis, col, n=100):
    """Mean and max difference charge - discharge of ``col`` at equal Δx (overlap range only)."""
    c = ch.sort_values("dx")
    d = dis.sort_values("dx")
    lo, hi = max(c["dx"].min(), d["dx"].min()), min(c["dx"].max(), d["dx"].max())
    g = np.linspace(lo, hi, n)
    gap = np.interp(g, c["dx"], c[col]) - np.interp(g, d["dx"], d[col])
    return {"dx_range": f"{lo:.2f}-{hi:.2f}", "mean_gap": float(gap.mean()), "max_gap": float(gap.max()),
            "dx_at_max": float(g[np.argmax(gap)])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--table", help="optional CSV (cell,scan,x) with a hand-made x column to check")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--include-cv", action="store_true", help="also fit the scan(s) taken during the 4.5 V hold")
    ap.add_argument("--hysteresis", action="store_true", help="also plot discharge and report the Ni-edge gap")
    ap.add_argument("--threshold", type=float, default=0.1, help="eV per step for the SI threshold rule")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    table = pd.read_csv(a.table) if a.table else None

    obs = [("edge_half", "XAS", "Ni K-edge E$_{1/2}$ (eV)"), ("white_line", "XAS", "Ni white line (eV)"),
           ("a", "XRD", "a (Å)"), ("c", "XRD", "c (Å)")]
    fig, ax = plt.subplots(len(obs) + 1, len(CELLS), figsize=(4.2 * len(CELLS), 14), sharex="col")
    rows, hyst = [], []
    for j, (name, c) in enumerate(CELLS.items()):
        comp = COMPOSITION[name]
        echem, xas, xrd = load_cell(a.root, c, x0=comp["Na"])
        pd.concat([xas, xrd]).to_csv(os.path.join(a.out, f"aligned_{name}.csv"), index=False)
        print(f"{name}: clock shift of re-exported echem {echem.attrs['clock_shift_s'] / 86400:.1f} d; "
              f"{len(xas)} XAS, {len(xrd)} XRD scans; segments XAS: "
              + ", ".join(f"{k} {v}" for k, v in xas['segment'].value_counts().items()))
        if table is not None and name in set(table["cell"]):
            chk = compare_with_table(xas, table.loc[table["cell"] == name].sort_values("scan")["x"])
            bad = chk[chk["flag"] != ""]
            print(f"  hand-made x: constant offset {chk.attrs['offset']:+.3f}; flagged scans: "
                  + ("; ".join(f"#{r.scan} {r.flag} (V={r.V:.2f})" for r in bad.itertuples()) or "none"))

        e_dx = echem["x"].iloc[0] - echem["x"]
        ax[0, j].plot(e_dx, echem["Ewe/V"], "k-", lw=0.7)
        ax[0, j].set_title(name)
        ax[0, j].set_ylabel("V vs Na$^+$/Na")
        for i, (col, mod, label) in enumerate(obs, start=1):
            src = xas if mod == "XAS" else xrd
            ch = first_charge(src, a.include_cv).dropna(subset=[col])
            dis = src[src["segment"].str.startswith("discharge")].dropna(subset=[col])
            r = fit_hinge(ch["dx"].to_numpy(), ch[col].to_numpy(), n_boot=a.n_boot)
            v_b = float(np.interp(r.xb, ch["dx"], ch["V"]))
            rows.append({"cell": name, "observable": col, "modality": mod, "n": r.n,
                         "dx_break": r.xb, "dx_ci_lo": r.xb_ci[0], "dx_ci_hi": r.xb_ci[1], "V_break": v_b,
                         "slope_before": r.k1, "slope_after": r.k2, "slope_ratio": r.k2 / r.k1,
                         "delta_bic": r.delta_bic, "Ni_per_fu": comp["Ni"]})
            if mod == "XAS":
                xo, io_ = threshold_onset(ch["dx"], ch[col], tol=a.threshold)
                rows[-1].update(dx_threshold=xo, V_threshold=np.nan if io_ is None else float(ch["V"].iloc[io_]))
            if col == "edge_half" and a.hysteresis:
                hyst.append({"cell": name, **branch_gap(ch, dis, col)})
            axx = ax[i, j]
            axx.plot(ch["dx"], ch[col], "o", color="C3", ms=4, label="charge")
            if a.hysteresis:
                axx.plot(dis["dx"], dis[col], "o", mfc="none", color="C0", ms=4, label="discharge")
            if mod == "XAS" and np.isfinite(rows[-1]["dx_threshold"]):
                axx.axvline(rows[-1]["dx_threshold"], color="C2", lw=0.8, ls=":", label="SI threshold rule")
            xx = np.linspace(ch["dx"].min(), ch["dx"].max(), 200)
            axx.plot(xx, r.predict(xx), "-", color="C3", lw=1)
            axx.axvspan(*r.xb_ci, color="0.85", zorder=0)
            axx.axvline(r.xb, color="0.4", lw=0.8, ls="--")
            axx.set_ylabel(label)
        ax[-1, j].set_xlabel("Na extracted, Δx")
    ax[1, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "breakpoints.png"), dpi=120)

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(a.out, "breakpoints.csv"), index=False)
    with open(os.path.join(a.out, "breakpoints.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    pd.set_option("display.width", 160)
    print(summary.round(4).to_string(index=False))
    if not a.hysteresis:
        return
    h = pd.DataFrame(hyst)
    h.to_csv(os.path.join(a.out, "edge_hysteresis.csv"), index=False)
    print("\nNi edge hysteresis (charge minus discharge at equal Δx):")
    print(h.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
