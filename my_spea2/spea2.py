"""SPEA-2 implementation for 3-objective MONRP (from scratch).

Objectives:
    f1  normalised development cost       — minimise
    f2  weighted customer satisfaction    — maximise
    f3  variance of per-stakeholder sat.  — minimise (fairness)

Reference: Zitzler, Laumanns, Thiele (2001).
"""
from __future__ import annotations

import random
import time
from typing import Any, Dict, List

import numpy as np

from common.dataset import Dataset
from common.solution import Solution
from common.operators import apply_crossover, apply_mutation


class SPEA2:

    def __init__(
        self,
        dataset: Dataset,
        pop_size: int = 100,
        archive_size: int = 100,
        max_generations: int = 250,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        tackle_dependencies: bool = False,
        seed: int = None,
    ) -> None:
        self.dataset = dataset
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.max_generations = max_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
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
        archive: List[Solution] = []

        for gen in range(self.max_generations):
            combined = population + archive
            self._assign_fitness(combined)
            archive = self._environmental_selection(combined)

            if self.tackle_dependencies:
                for sol in archive:
                    sol.correct_dependencies()

            if gen == self.max_generations - 1:
                break

            mating_pool = self._mating_selection(archive + population)
            offspring = apply_crossover(mating_pool, self.crossover_prob)
            offspring = apply_mutation(offspring, self.mutation_prob)
            population = offspring[: self.pop_size]
            # top up if crossover shortened the list
            while len(population) < self.pop_size:
                population.append(Solution(self.dataset, uniform=True))

        nd = [s for s in archive if s.raw_fitness < 1]
        if not nd:
            nd = sorted(archive, key=lambda s: s.spea2_fitness)

        return {
            "population": nd,
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
    # Fitness assignment (Zitzler et al., 2001, Section 3.1)
    # ------------------------------------------------------------------

    def _assign_fitness(self, combined: List[Solution]) -> None:
        n = len(combined)

        # Strength S(i) = |{j ∈ P̃ : i ≻ j}|
        for ind in combined:
            ind.strength = sum(1 for other in combined if ind.dominates(other))

        # Raw fitness R(i) = Σ_{j ≻ i} S(j)
        for ind in combined:
            ind.raw_fitness = float(
                sum(other.strength for other in combined if other.dominates(ind))
            )

        # Density D(i) via k-th nearest neighbour in objective space
        k = max(1, int(np.sqrt(n)))
        obj_mat = np.array([s.objectives_min() for s in combined])  # (n, 3)

        for idx, sol in enumerate(combined):
            dists = np.sqrt(np.sum((obj_mat - obj_mat[idx]) ** 2, axis=1))
            sorted_dists = np.sort(dists)
            sigma_k = sorted_dists[k] if k < n else sorted_dists[-1]
            sol.density = 1.0 / (sigma_k + 2.0)
            sol.spea2_fitness = sol.raw_fitness + sol.density

    # ------------------------------------------------------------------
    # Environmental selection
    # ------------------------------------------------------------------

    def _environmental_selection(
        self, combined: List[Solution]
    ) -> List[Solution]:
        nd = [s for s in combined if s.raw_fitness < 1]

        if len(nd) < self.archive_size:
            dominated = sorted(
                [s for s in combined if s.raw_fitness >= 1],
                key=lambda s: s.spea2_fitness,
            )
            nd.extend(dominated[: self.archive_size - len(nd)])
            return nd

        if len(nd) > self.archive_size:
            return self._truncate(nd, self.archive_size)

        return nd

    def _truncate(
        self, solutions: List[Solution], size: int
    ) -> List[Solution]:
        """Remove crowded solutions one by one via lexicographic distance comparison."""
        current = solutions[:]
        obj_mat = np.array([s.objectives_min() for s in current])  # (n, 3)

        while len(current) > size:
            n = len(current)
            # Pairwise Euclidean distances in objective space
            diff = obj_mat[:, np.newaxis, :] - obj_mat[np.newaxis, :, :]
            dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))          # (n, n)
            np.fill_diagonal(dist_mat, np.inf)

            # Sort each row ascending; last entry is the inf diagonal — drop it
            sorted_dists = np.sort(dist_mat, axis=1)[:, : n - 1]   # (n, n-1)

            # Lexicographic minimum: find the row with the smallest distance sequence
            remaining = np.arange(n)
            for col in range(n - 1):
                if len(remaining) == 1:
                    break
                col_vals = sorted_dists[remaining, col]
                min_val = float(np.min(col_vals))
                mask = col_vals == min_val
                remaining = remaining[mask]

            remove_idx = int(remaining[0])
            current.pop(remove_idx)
            obj_mat = np.delete(obj_mat, remove_idx, axis=0)

        return current

    # ------------------------------------------------------------------
    # Mating selection (binary tournament on SPEA-2 fitness)
    # ------------------------------------------------------------------

    def _mating_selection(self, pool: List[Solution]) -> List[Solution]:
        selected: List[Solution] = []
        for _ in range(self.pop_size):
            a, b = random.sample(pool, min(2, len(pool)))
            winner = a if a.spea2_fitness <= b.spea2_fitness else b
            selected.append(winner.clone())
        return selected
