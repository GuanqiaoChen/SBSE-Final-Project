"""Quality metrics for Pareto fronts."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from common.solution import Solution

try:
    from pymoo.indicators.hv import HV as _HV

    _PYMOO = True
except ImportError:
    _PYMOO = False


def objective_matrix(population: List[Solution]) -> np.ndarray:
    """Return (n, 3) objectives using the all-minimise convention."""
    return np.array([s.objectives_min() for s in population])


def objective_matrix_2d(population: list) -> np.ndarray:
    """Return shared cost/satisfaction objectives using all-minimise convention."""
    return np.array([_shared_objectives_min(s) for s in population])


def _shared_objectives_min(solution) -> Tuple[float, float]:
    if hasattr(solution, "cost"):
        cost = solution.cost
        satisfaction = solution.satisfaction
    else:
        cost = solution.total_cost
        satisfaction = solution.total_satisfaction
    return float(cost), -float(satisfaction)


def hypervolume(
    population: List[Solution],
    ref_point: Optional[np.ndarray] = None,
) -> float:
    """3-D hypervolume. Requires pymoo; returns NaN if unavailable."""
    if not population:
        return 0.0
    if not _PYMOO:
        return float("nan")
    if ref_point is None:
        ref_point = np.array([1.1, 1.1, 1.1])
    ind = _HV(ref_point=ref_point)
    return float(ind(objective_matrix(population)))


def hypervolume_2d(
    population: list,
    ref_point: Optional[np.ndarray] = None,
) -> float:
    """2-D hypervolume on cost and satisfaction. Requires pymoo."""
    if not population:
        return 0.0
    if not _PYMOO:
        return float("nan")
    if ref_point is None:
        ref_point = np.array([1.1, 0.1])
    ind = _HV(ref_point=ref_point)
    return float(ind(objective_matrix_2d(population)))


def spread(population: List[Solution]) -> float:
    """Extent of the 3-D front."""
    if len(population) < 2:
        return 0.0
    mat = objective_matrix(population)
    ranges = np.max(mat, axis=0) - np.min(mat, axis=0)
    return float(np.linalg.norm(ranges))


def spread_2d(population: list) -> float:
    """Extent of the shared 2-D front."""
    if len(population) < 2:
        return 0.0
    mat = objective_matrix_2d(population)
    ranges = np.max(mat, axis=0) - np.min(mat, axis=0)
    return float(np.linalg.norm(ranges))


def spacing(population: List[Solution]) -> float:
    """Spacing metric: uniformity of solution distribution (lower is better)."""
    n = len(population)
    if n < 2:
        return 0.0
    mat = objective_matrix(population)
    return _spacing_from_matrix(mat)


def spacing_2d(population: list) -> float:
    """Spacing metric on the shared 2-D front."""
    n = len(population)
    if n < 2:
        return 0.0
    mat = objective_matrix_2d(population)
    return _spacing_from_matrix(mat)


def _spacing_from_matrix(mat: np.ndarray) -> float:
    n = len(mat)
    min_dists = []
    for i in range(n):
        diffs = mat - mat[i]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        dists[i] = np.inf
        min_dists.append(float(np.min(dists)))
    d_bar = np.mean(min_dists)
    return float(np.sqrt(np.sum((d_bar - np.array(min_dists)) ** 2) / (n - 1)))


def igd(
    population: List[Solution],
    reference_front: List[Solution],
) -> float:
    """Inverted Generational Distance from reference_front to population."""
    if not population or not reference_front:
        return float("nan")
    approx = objective_matrix(population)
    ref = objective_matrix(reference_front)
    total = sum(
        float(np.min(np.sqrt(np.sum((approx - r) ** 2, axis=1))))
        for r in ref
    )
    return total / len(ref)


def summary(
    population: List[Solution],
    ref_point: Optional[np.ndarray] = None,
) -> dict:
    return {
        "num_solutions": len(population),
        "hypervolume": hypervolume(population, ref_point),
        "spread": spread(population),
        "spacing": spacing(population),
    }


def summary_2d(population: list, ref_point: Optional[np.ndarray] = None) -> dict:
    return {
        "num_solutions": len(population),
        "hypervolume": hypervolume_2d(population, ref_point),
        "spread": spread_2d(population),
        "spacing": spacing_2d(population),
    }
