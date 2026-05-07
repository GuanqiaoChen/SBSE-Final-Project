"""Shared genetic operators (crossover and mutation) for NSGA-II and SPEA-2."""
from __future__ import annotations

import random
from typing import List, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from common.solution import Solution


def apply_crossover(population: List[Solution], prob: float) -> List[Solution]:
    """One-point crossover applied to consecutive pairs."""
    new_pop: List[Solution] = []
    i = 0
    while i < len(population):
        if i == len(population) - 1:
            new_pop.append(population[i].clone())
            i += 1
        else:
            if random.random() < prob:
                c1, c2 = _one_point(population[i], population[i + 1])
            else:
                c1, c2 = population[i].clone(), population[i + 1].clone()
            new_pop.extend([c1, c2])
            i += 2
    return new_pop


def _one_point(p1: Solution, p2: Solution) -> Tuple[Solution, Solution]:
    from common.solution import Solution

    n = len(p1.selected)
    pt = random.randint(1, n - 1)
    g1 = np.concatenate([p1.selected[:pt], p2.selected[pt:]])
    g2 = np.concatenate([p2.selected[:pt], p1.selected[pt:]])
    _ensure_nonempty(g1)
    _ensure_nonempty(g2)
    return Solution(p1.dataset, selected=g1), Solution(p1.dataset, selected=g2)


def apply_mutation(population: List[Solution], prob: float) -> List[Solution]:
    """Bit-flip mutation where each gene flips independently with probability *prob*."""
    from common.solution import Solution

    new_pop: List[Solution] = []
    for ind in population:
        new_sel = ind.selected.copy()
        for i in range(len(new_sel)):
            if random.random() < prob:
                new_sel[i] ^= 1
        _ensure_nonempty(new_sel)
        sol = Solution(ind.dataset, selected=new_sel)
        sol.rank = ind.rank
        sol.crowding_distance = ind.crowding_distance
        new_pop.append(sol)
    return new_pop


def _ensure_nonempty(arr: np.ndarray) -> None:
    """Guarantee at least one gene is selected (avoids empty solutions)."""
    if not np.any(arr):
        arr[random.randint(0, len(arr) - 1)] = 1
