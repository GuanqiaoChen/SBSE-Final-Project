"""Quality metrics used by the UCLM_SIMD executer (2-objective: cost ↓, satisfaction ↑)."""
from typing import List
import numpy as np

try:
    from pymoo.indicators.hv import HV as _HV
    _PYMOO = True
except ImportError:
    _PYMOO = False


def calculate_hypervolume(population: list, ref_x: float = 1.1, ref_y: float = 1.1) -> float:
    """2-D hypervolume.  Points are (total_cost, 1−total_satisfaction) so both axes are minimised."""
    if not population or not _PYMOO:
        return float('nan')
    points = np.array(
        [[s.total_cost, 1.0 - s.total_satisfaction] for s in population]
    )
    ind = _HV(ref_point=np.array([ref_x, ref_y]))
    return float(ind(points))


def calculate_spread(population: list) -> float:
    """Extent of the approximation front (L2 norm of per-objective range)."""
    if len(population) < 2:
        return 0.0
    costs = [s.total_cost for s in population]
    sats  = [s.total_satisfaction for s in population]
    dc = max(costs) - min(costs)
    ds = max(sats)  - min(sats)
    return float(np.sqrt(dc ** 2 + ds ** 2))


def calculate_numSolutions(population: list) -> int:
    return len(population)


def calculate_spacing(population: list) -> float:
    """Spacing metric — lower means more uniform distribution."""
    n = len(population)
    if n < 2:
        return 0.0
    pts = np.array([[s.total_cost, s.total_satisfaction] for s in population])
    min_dists = []
    for i in range(n):
        diffs = pts - pts[i]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        dists[i] = np.inf
        min_dists.append(float(np.min(dists)))
    d_bar = np.mean(min_dists)
    return float(np.sqrt(np.sum((d_bar - np.array(min_dists)) ** 2) / (n - 1)))


def calculate_bestAvgValue(population: list) -> float:
    if not population:
        return 0.0
    return float(max(s.mono_objective_score for s in population))


def calculate_avgValue(population: list) -> float:
    if not population:
        return 0.0
    return float(np.mean([s.mono_objective_score for s in population]))


def calculate_mean_bits_per_sol(population: list) -> float:
    if not population:
        return 0.0
    return float(np.mean([int(np.sum(s.selected)) for s in population]))
