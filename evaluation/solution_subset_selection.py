"""Solution Subset Selection (SSS) post-processing.

sss_type 0 — greedy hypervolume contribution (default)
sss_type 1 — fallback: return first *subset_size* solutions as-is
"""
from typing import List
import numpy as np

try:
    from pymoo.indicators.hv import HV as _HV
    _PYMOO = True
except ImportError:
    _PYMOO = False


def search_solution_subset(sss_type: int, subset_size: int, solutions: list) -> list:
    if len(solutions) <= subset_size:
        return solutions
    if sss_type == 0 and _PYMOO:
        return _greedy_hv(solutions, subset_size)
    # Fallback: return solutions with highest mono_objective_score
    return sorted(solutions, key=lambda s: s.mono_objective_score, reverse=True)[:subset_size]


def _greedy_hv(solutions: list, k: int) -> list:
    """Iteratively add the solution with the greatest marginal HV contribution."""
    ref = np.array([1.1, 1.1])
    hv_ind = _HV(ref_point=ref)

    def _pts(subset):
        return np.array([[s.total_cost, 1.0 - s.total_satisfaction] for s in subset])

    selected: list = []
    remaining = solutions[:]

    while len(selected) < k and remaining:
        best_sol, best_hv = None, -1.0
        for sol in remaining:
            candidate = selected + [sol]
            hv = float(hv_ind(_pts(candidate)))
            if hv > best_hv:
                best_hv = hv
                best_sol = sol
        selected.append(best_sol)
        remaining.remove(best_sol)

    return selected
