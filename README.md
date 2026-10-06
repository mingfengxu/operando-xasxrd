# operando-xasxrd

Align **semi-simultaneous operando XRD, XAS and electrochemistry** on one clock, extract
spectral/structural descriptors the same way for every scan, and locate where each
observable changes regime, with **bootstrap uncertainties fitted in batch with PyTorch**.

Written for an ALBA NOTOS beamtime (2023) on Li-substituted P2-type Na-ion cathodes:
four operando cells measured in turn, each cycle giving one XRD pattern and
one Ni K-edge XANES spectrum per cell every ~40 min (20–33 of each per cell).
Published in *Adv. Funct. Mater.* 36 (2025) 2425499.

## Why

In a semi-simultaneous experiment XRD and XAS of one cell are never taken at the same
moment, and the files live on different clocks and in different formats:

| source | format | time information |
|---|---|---|
| potentiostat | EC-Lab `.mpt` / `.txt` | absolute time per point (a re-processed export may be **re-dated**) |
| XRD, XAS | SPEC/Sardana `.dat` | `#D` start and "Acquisition ended" lines per scan |
| XANES | ATHENA multi-column export | none (scan number in the group name) |
| lattice parameters | FullProf sequential refinement, Excel summary | pattern number |

Hand-built tables (scan → Na content) are error-prone. This package rebuilds them from
the raw time stamps and checks them.

## What it does

1. **Readers** for the four formats (`io.py`). `merge_reexport` restores the original clock
   of a re-processed EC-Lab export (the one with the `x` column was re-dated by 82 days)
   after checking that the potential matches row by row.
2. **Alignment** (`align.py`): every scan is mapped to the electrochemical state at the middle
   of its acquisition — Na extracted Δx, voltage, half cycle and protocol segment (rest, CC
   charge/discharge, CV hold, before/after the echem record). `compare_with_table` checks a
   hand-made x column.
3. **XANES descriptors** (`xanes.py`): edge position at half step (E½) and white-line maximum
   with sub-grid quadratic refinement. These are the same for every spectrum, so there are no manual picks.
4. **Break-point regression in PyTorch** (`breakpoint.py`): a smooth two-slope model
   `y = b + k1(x−xb) + (k2−k1)·softplus(β(x−xb))/β`. It starts from an exact grid search
   and is refined by Adam. A wild residual bootstrap gives the confidence interval of xb. All
   2000 replicates are one `(B, 4)` parameter tensor optimised in a single autograd graph,
   so they run in seconds and are GPU-ready. ΔBIC tests the hinge against a straight line.

## Results (first charge, 2.0–4.5 V, C/10)

```bash
pip install -e ".[dev]"
python examples/afm_li_p2.py --root ~/Work/operando_XRD_XANES --out results [--table xas_x.csv]
#   --include-cv   also fit the scan taken during the 4.5 V hold (default: constant-current charge only)
#   --hysteresis   also plot discharge and report the charge/discharge gap of the Ni edge
```

Only the **constant-current part of the first charge** is fitted by default. LNM-2/3/4 were held
at 4.5 V for 40 min, and one scan per cell fell into that hold. Including it moves the Ni-edge
and *a* breaks by ≤ 0.004. It changes only LNM-2 *c* (0.34 → 0.41), because *c* drops sharply
(11.23 → 11.06 Å) during the hold, the onset of interlayer collapse at 4.5 V.

Compositions (Rietveld, SI Table S11) set the absolute x in the aligned tables; plots use Δx.

| cell | composition | Ni per f.u. |
|---|---|---|
| NM | Na0.71Ni0.334Mn0.666O2 | 0.334 |
| LNM-2 | Na0.745Li0.164Ni0.238Mn0.599O2 | 0.238 |
| LNM-3 | Na0.748Li0.189Ni0.172Mn0.638O2 | 0.172 |
| LNM-4 | Na0.75Li0.222Ni0.081Mn0.697O2 | 0.081 |

**Break points, first charge (CC), Δx with 95 % bootstrap interval**

| cell | Ni E½ | V | Ni white line | *a* | *c* | Ni E½ slope after/before |
|---|---|---|---|---|---|---|
| NM    | 0.35 (0.28–0.41) | 3.95 | 0.24 (0.22–0.27) | 0.31 (0.29–0.34) | 0.27 (0.26–0.28) | 0.35 |
| LNM-2 | 0.37 (0.34–0.39) | 4.09 | 0.24 (0.20–0.29) | 0.28 (0.25–0.30) | 0.34 (0.32–0.36) | 0.33 |
| LNM-3 | 0.29 (0.27–0.31) | 4.24 | 0.25 (0.23–0.27) | 0.24 (0.21–0.26) | 0.25 (0.24–0.27) | 0.07 |
| LNM-4 | 0.17 (0.16–0.17) | 3.62 | 0.15 (0.14–0.16) | 0.15 (0.14–0.16) | 0.16 (0.15–0.17) | 0.01 |

ΔBIC > 20 for every fit, so a break is strongly preferred over a straight line.

* The break moves to smaller Δx as the Ni content falls (NM/LNM-2 → LNM-3 → LNM-4).
* In **LNM-3 and LNM-4** the Ni edge stops shifting (slope 1–7 % of the initial value) at the
  same Δx where *a* stops contracting. About 0.2–0.3 more Na per formula unit is then extracted
  without further Ni oxidation, which points to oxygen redox. Two independent probes, XAS and
  XRD, agree within their bootstrap intervals.
* In **NM and LNM-2** the edge keeps shifting after the break (about a third of the initial
  rate). In NM the frozen *a*/*c* reflect the two-phase P2–O2 region (only the P2 phase is
  refined), not the end of Ni redox.

**Comparison with the criterion used in the paper (SI Fig. S18).** The paper counts Ni redox as
finished once the white-line change stays within ±0.1 eV for two consecutive points.
`threshold_onset` implements that rule. On the same automatically extracted white line and the
same Δx axis:

| cell | SI threshold rule | hinge, white line | hinge, E½ |
|---|---|---|---|
| NM    | 0.38 | 0.24 | 0.35 |
| LNM-2 | 0.52 | 0.24 | 0.37 |
| LNM-3 | 0.37 | 0.25 | 0.29 |
| LNM-4 | 0.24 | 0.15 | 0.17 |

Both methods give the same ordering, but the threshold rule comes out later. It returns the first
point already inside the band, not the intersection of the two regimes. Its tolerance is per
step, so it implicitly demands a slope below 0.1 eV / (step in Δx), about 2.4 eV per unit Δx here.
The answer therefore changes with the scan interval, and a slow residual drift, as in LNM-2,
postpones it a lot. The hinge fit does not depend on the step and comes with an uncertainty.

**Data checks found by the alignment** (against the hand-made x table behind Fig. S18):

* For LNM-3/4 the table uses the refined Na content as x0, so it matches the potentiostat to
  within 0.01, except for the first one or two points. For NM it is offset by 0.07 and for LNM-2 by 0.02.
* The last point of LNM-3 and of LNM-4 in that table, plotted as the high-voltage end of charge,
  was recorded about 27 min **into the discharge** (3.75 / 3.86 V). The charge ended, the 40 min
  hold and a 5 min rest followed, and the next scan came after the current had reversed. In
  Fig. S18 these are the last points (−0.05 and −0.17 eV). The −0.17 eV point, the only one
  clearly outside the ±0.1 eV band, is the Ni edge starting to come back on discharge.
* ATHENA's automatic white-line pick failed for the first two NM spectra (8346.5 eV instead
  of ≈8351.3 eV). The descriptor in `xanes.py` gives 8351.35 eV.

## Limitations

* Δx is Na extracted according to the potentiostat (it includes any side-reaction current).
  Absolute x uses the refined initial Na content.
* The reference spectra (NiO, LiNiO₂) in the four ATHENA projects are not on a common energy
  scale (E½ of NiO differs by up to 1 eV between files). Only shifts within one cell are
  used, not absolute oxidation states.
* A two-slope model is a summary, not a mechanism. With 11–15 points per branch the
  intervals reflect noise around that model, not model error.
* Lattice parameters come from the existing FullProf sequential refinement. The package does
  not refine patterns.

## Layout

```
src/operando_xasxrd/  io.py  align.py  xanes.py  breakpoint.py
examples/afm_li_p2.py  tests/  .github/workflows/ci.yml
```

Raw data are not included.
