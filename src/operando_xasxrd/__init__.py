"""Align semi-simultaneous operando XRD / XAS / electrochemistry and quantify where they change regime."""
from .align import align_scans, compare_with_table, segments, state_at
from .breakpoint import HingeResult, fit_hinge, fit_hinge_batch
from .io import (Scan, merge_reexport, read_athena_columns, read_athena_table, read_eclab,
                 read_refinement_summary, read_spec_scans, scan_index)
from .xanes import describe, edge_half, white_line

__version__ = "0.1.0"
