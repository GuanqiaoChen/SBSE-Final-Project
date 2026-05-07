"""NSGA-II implementation for 3-objective MONRP (from scratch).

Objectives:
    f1  normalised development cost       — minimise
    f2  weighted customer satisfaction    — maximise
    f3  variance of per-stakeholder sat.  — minimise (fairness)

Reference: Deb, Pratap, Agarwal, Meyarivan (2002).
"""
from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from common.dataset import Dataset
from common.solution import Solution
from common.operators import apply_crossover, apply_mutation


class NSGA2:

    def __init__(
        self,
        dataset: Dataset,
        pop_size: int = 100,
        max_generations: int = 250,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        selection_candidates: int = 2,
        tackle_dependencies: bool = False,
        seed: int = None,
    ) -> None:
        self.dataset = dataset
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.selection_candidates = selection_candidates
        self.tackle_dependencies = tackle_dependencies

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        start = time.time()

        population = self._init_population()
        fronts = self._fast_nondominated_sort(population)
        for front in fronts:
            self._crowding_distance(front)
        offspring = self._breed(population)

        for _gen in range(self.max_generations):
            combined = population + offspring
            fronts = self._fast_nondominated_sort(combined)

            new_pop: List[Solution] = []
            fi = 0
            while fi < len(fronts) and len(new_pop) + len(fronts[fi]) <= self.pop_size:
                self._crowding_distance(fronts[fi])
                new_pop.extend(fronts[fi])
                fi += 1

            if len(new_pop) < self.pop_size and fi < len(fronts):
                self._crowding_distance(fronts[fi])
                fronts[fi].sort(key=lambda s: s.crowding_distance, reverse=True)
                new_pop.extend(fronts[fi][: self.pop_size - len(new_pop)])

            population = new_pop

            if self.tackle_dependencies:
                for sol in population:
                    sol.correct_dependencies()

            offspring = self._breed(population)

        final_fronts = self._fast_nondominated_sort(population)
        self._crowding_distance(final_fronts[0])

        return {
            "population": final_fronts[0],
            "time": time.time() - start,
            "num_generations": self.max_generations,
        }

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_population(self) -> List[Solution]:
        pop: List[Solution] = []
        while len(pop) < self.pop_size:
            pop.append(Solution(self.dataset, uniform=True))
        return pop

    # ------------------------------------------------------------------
    # Fast non-dominated sort  (Deb et al., 2002)
    # ------------------------------------------------------------------

    def _fast_nondominated_sort(
        self, population: List[Solution]
    ) -> List[List[Solution]]:
        fronts: List[List[Solution]] = [[]]

        for ind in population:
            ind.domination_count = 0
            ind.dominated_solutions = []
            for other in population:
                if ind.dominates(other):
                    ind.dominated_solutions.append(other)
                elif other.dominates(ind):
                    ind.domination_count += 1
            if ind.domination_count == 0:
                ind.rank = 0
                fronts[0].append(ind)

        i = 0
        while fronts[i]:
            next_front: List[Solution] = []
            for ind in fronts[i]:
                for dominated in ind.dominated_solutions:
                    dominated.domination_count -= 1
                    if dominated.domination_count == 0:
                        dominated.rank = i + 1
                        next_front.append(dominated)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]  # last element is always an empty sentinel

    # ------------------------------------------------------------------
    # Crowding distance (extended to 3 objectives)
    # ------------------------------------------------------------------

    def _crowding_distance(self, front: List[Solution]) -> None:
        n = len(front)
        if n == 0:
            return
        if n <= 2:
            for sol in front:
                sol.crowding_distance = float('inf')
            return

        for sol in front:
            sol.crowding_distance = 0.0

        # For each objective (expressed as 'minimise')
        for get_val in (
            lambda s: s.cost,
            lambda s: -s.satisfaction,       # negate so lower = better
            lambda s: s.fairness_variance,
        ):
            sorted_f = sorted(front, key=get_val)
            sorted_f[0].crowding_distance = float('inf')
            sorted_f[-1].crowding_distance = float('inf')
            vals = [get_val(s) for s in sorted_f]
            scale = vals[-1] - vals[0] or 1.0
            for i in range(1, n - 1):
                sorted_f[i].crowding_distance += (vals[i + 1] - vals[i - 1]) / scale

    # ------------------------------------------------------------------
    # Tournament selection (crowding operator)
    # ------------------------------------------------------------------

    def _crowding_operator(self, a: Solution, b: Solution) -> int:
        """Return 1 if a is preferred over b, else -1."""
        if a.rank < b.rank:
            return 1
        if a.rank > b.rank:
            return -1
        return 1 if a.crowding_distance > b.crowding_distance else -1

    def _tournament_select(self, population: List[Solution]) -> List[Solution]:
        selected: List[Solution] = []
        for _ in range(len(population)):
            candidates = random.choices(population, k=self.selection_candidates)
            best = candidates[0]
            for c in candidates[1:]:
                if self._crowding_operator(c, best) == 1:
                    best = c
            selected.append(best.clone())
        return selected

    # ------------------------------------------------------------------
    # Offspring generation
    # ------------------------------------------------------------------

    def _breed(self, population: List[Solution]) -> List[Solution]:
        mating_pool = self._tournament_select(population)
        offspring = apply_crossover(mating_pool, self.crossover_prob)
        offspring = apply_mutation(offspring, self.mutation_prob)
        return offspring
