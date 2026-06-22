"""Sparse-recovery localization (COMP5340 contribution).

Reformulates AutoTS traffic-sign localization as sparse outlier recovery and
provides convex (L1-SOR), greedy (OMP/CoSaMP/SP) and uncertainty-aware (USPA)
aggregators alongside the paper's NSAL and classic baselines. See the proposal
report ``2026_06_18_compressed_sensing_autots_proposal_report_18.md``.
"""

from . import aggregators, data, metrics  # noqa: F401
from .aggregators import ALL_METHODS, K_MIN, AggResult, aggregate  # noqa: F401
