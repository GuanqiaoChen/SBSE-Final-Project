from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np


class Solution:
    """Binary-encoded solution for the 3-objective MONRP.

    Objectives (all normalised to roughly [0, 1]):
        f1  cost              — minimise
        f2  satisfaction      — maximise (stored as-is; dominance handles direction)
        f3  fairness_variance — minimise (variance of per-stakeholder satisfaction)
    """

    def __init__(
        self,
        dataset,
        selected: Optional[np.ndarray] = None,
        uniform: bool = False,
    ) -> None:
        self.dataset = dataset

        if selected is not None:
            self.selected = np.asarray(selected, dtype=np.int8).flatten()
        elif uniform:
            while True:
                self.selected = np.random.randint(
                    0, 2, dataset.num_requirements, dtype=np.int8
                )
                if np.any(self.selected):
                    break
        else:
            self.selected = np.zeros(dataset.num_requirements, dtype=np.int8)

        # Objective values — populated by evaluate()
        self.cost: float = 0.0
        self.satisfaction: float = 0.0
        self.fairness_variance: float = 0.0

        # NSGA-II bookkeeping
        self.rank: int = 0
        self.crowding_distance: float = 0.0
        self.domination_count: int = 0
        self.dominated_solutions: List[Solution] = []

        # SPEA-2 bookkeeping
        self.strength: int = 0
        self.raw_fitness: float = 0.0
        self.density: float = 0.0
        self.spea2_fitness: float = 0.0

        self.evaluate()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> None:
        ds = self.dataset

        if not np.any(self.selected):
            self.cost = 0.0
            self.satisfaction = 0.0
            self.fairness_variance = 0.0
            return

        self.cost = float(np.dot(self.selected, ds.costs)) / ds.total_cost

        per_st = np.array(
            [
                float(np.dot(self.selected, ds.priorities[s]))
                / ds.max_sat_per_stakeholder[s]
                for s in range(ds.num_stakeholders)
            ]
        )
        self.satisfaction = float(np.dot(ds.importances, per_st)) / ds.total_importance
        self.fairness_variance = float(np.var(per_st))

    # ------------------------------------------------------------------
    # Pareto dominance (3-objective)
    # ------------------------------------------------------------------

    def dominates(self, other: Solution) -> bool:
        """Return True if self Pareto-dominates other."""
        # No worse in all three objectives …
        if self.cost > other.cost:
            return False
        if self.satisfaction < other.satisfaction:
            return False
        if self.fairness_variance > other.fairness_variance:
            return False
        # … and strictly better in at least one
        return (
            self.cost < other.cost
            or self.satisfaction > other.satisfaction
            or self.fairness_variance < other.fairness_variance
        )

    # ------------------------------------------------------------------
    # Mutation helper
    # ------------------------------------------------------------------

    def set_bit(self, index: int, value: int) -> None:
        self.selected[index] = value
        self.evaluate()

    # ------------------------------------------------------------------
    # Dependency repair (forward-chaining)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def objectives_min(self) -> Tuple[float, float, float]:
        """All objectives expressed as 'minimise': (cost, −satisfaction, fairness_variance)."""
        return (self.cost, -self.satisfaction, self.fairness_variance)

    def clone(self) -> Solution:
        s = Solution.__new__(Solution)
        s.dataset = self.dataset
        s.selected = self.selected.copy()
        s.cost = self.cost
        s.satisfaction = self.satisfaction
        s.fairness_variance = self.fairness_variance
        s.rank = self.rank
        s.crowding_distance = self.crowding_distance
        s.domination_count = 0
        s.dominated_solutions = []
        s.strength = self.strength
        s.raw_fitness = self.raw_fitness
        s.density = self.density
        s.spea2_fitness = self.spea2_fitness
        return s

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Solution(cost={self.cost:.3f}, sat={self.satisfaction:.3f}, "
            f"var={self.fairness_variance:.4f})"
        )
