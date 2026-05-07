"""UCLM-compatible 2-objective Solution (cost ↓, satisfaction ↑).

This class matches the interface expected by the UCLM_SIMD algorithm code.
Our own 3-objective Solution lives in common/solution.py.
"""
from __future__ import annotations

from typing import List
import numpy as np


class Solution:

    def __init__(self, dataset, _ignored=None, selected=None, uniform: bool = False):
        self.dataset = dataset

        if selected is not None:
            # Accept both a full binary array and a np.where()-style tuple
            if isinstance(selected, tuple):
                arr = np.zeros(dataset.num_requirements, dtype=np.int8)
                if len(selected[0]):
                    arr[selected[0]] = 1
                self.selected = arr
            else:
                self.selected = np.asarray(selected, dtype=np.int8).flatten()
        elif uniform:
            # Generate with default int dtype to match UCLM RNG sequence, then cast
            self.selected = np.random.randint(0, 2, dataset.num_requirements).astype(np.int8)
        else:
            self.selected = np.zeros(dataset.num_requirements, dtype=np.int8)

        # 2-objective values
        self.total_cost: float = 0.0
        self.total_satisfaction: float = 0.0
        self.mono_objective_score: float = 0.0

        # NSGA-II bookkeeping
        self.rank: int = 0
        self.crowding_distance: float = 0.0
        self.domination_count: int = 0
        self.dominated_solutions: List[Solution] = []

        self.evaluate()

    def evaluate(self) -> None:
        ds = self.dataset

        if not np.any(self.selected):
            self.total_cost = 0.0
            self.total_satisfaction = 0.0
            self.mono_objective_score = 0.0
            return

        self.total_cost = float(np.dot(self.selected, ds.costs)) / ds.total_cost

        per_st = np.array(
            [
                float(np.dot(self.selected, ds.priorities[s]))
                / ds.max_sat_per_stakeholder[s]
                for s in range(ds.num_stakeholders)
            ]
        )
        self.total_satisfaction = (
            float(np.dot(ds.importances, per_st)) / ds.total_importance
        )
        self.mono_objective_score = self.total_satisfaction - self.total_cost

    def dominates(self, other: Solution) -> bool:
        """2-objective Pareto dominance: cost ↓, satisfaction ↑."""
        if self.total_cost > other.total_cost:
            return False
        if self.total_satisfaction < other.total_satisfaction:
            return False
        return (
            self.total_cost < other.total_cost
            or self.total_satisfaction > other.total_satisfaction
        )

    def set_bit(self, index: int, value: int) -> None:
        self.selected[index] = value
        self.evaluate()

    def correct_dependencies(self) -> None:
        if not self.dataset.dependencies:
            return
        changed = True
        while changed:
            changed = False
            for i, deps in enumerate(self.dataset.dependencies):
                if self.selected[i] == 1:
                    for dep in deps:
                        if self.selected[dep] == 0:
                            self.selected[dep] = 1
                            changed = True
        self.evaluate()
