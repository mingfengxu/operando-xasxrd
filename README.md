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
```

| cell | Ni E½ break Δx (95 % CI) | V | a break Δx | c break Δx | Ni-edge slope after/before | Ni-edge hysteresis, mean (max) |
|---|---|---|---|---|---|---|
| NM    | 0.35 (0.28–0.41) | 3.95 | 0.31 (0.29–0.34) | 0.27 | 0.35 | 0.14 eV (0.44) |
| LNM-2 | 0.38 (0.35–0.40) | 4.10 | 0.28 (0.25–0.31) | 0.41 | 0.30 | 0.51 eV (0.75) |
| LNM-3 | 0.29 (0.28–0.31) | 4.24 | 0.24 (0.21–0.26) | 0.26 | 0.05 | 1.27 eV (2.10) |
| LNM-4 | 0.17 (0.16–0.17) | 3.62 | 0.15 (0.14–0.16) | 0.16 | 0.02 | 1.81 eV (3.18) |

ΔBIC > 20 for every fit, so a break is strongly preferred over a straight line.

* In **LNM-3 and LNM-4** the Ni edge stops shifting (slope drops to 2–5 %) at the same Δx
  where the in-plane parameter *a* stops contracting. About 0.2–0.35 more Na per formula unit
  is then extracted without further Ni oxidation, which points to oxygen redox. Two independent
  probes, XAS and XRD, agree within their bootstrap intervals.
* In **NM and LNM-2** the edge keeps shifting after the break (30–35 % of the initial rate).
  In NM the frozen *a*/*c* reflect the two-phase P2–O2 region (only the P2 phase is refined),
  not the end of Ni redox. The two modalities therefore disagree there for a structural
  reason, and the comparison makes that visible.
* The charge/discharge gap of the Ni edge at equal Δx grows monotonically with Li content
  (0.14 → 1.81 eV). On discharge, Ni is reduced only after a large part of the Na is
  re-inserted, which is consistent with the voltage hysteresis of the Li-rich compositions.

**Data checks found by the alignment** (against the hand-made x table used for the paper figure):

* The x values differ from the potentiostat by a constant per cell (initial-x convention).
* The last XAS point of LNM-3 and of LNM-4 was recorded on **discharge** (3.75 / 3.86 V),
  not at the end of charge.
* The last LNM-2 point lies in the 4.5 V CV hold, not on the CC branch.
* ATHENA's automatic white-line pick failed for the first two NM spectra (8346.5 eV instead
  of ≈8351.3 eV). The descriptor in `xanes.py` gives 8351.35 eV.

## Limitations

* Δx is Na extracted according to the potentiostat. Absolute x needs the initial composition
  (`x0` in `state_at`).
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
