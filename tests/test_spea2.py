"""Unit tests for SPEA-2."""
import numpy as np
import pytest

from common.dataset import Dataset
from common.solution import Solution
from my_spea2.spea2 import SPEA2


@pytest.fixture
def ds():
    return Dataset("a1")


@pytest.fixture
def alg(ds):
    return SPEA2(ds, pop_size=20, archive_size=20, max_generations=5, seed=0)


def test_assign_fitness_sets_all_fields(alg, ds):
    pop = [Solution(ds, uniform=True) for _ in range(10)]
    alg._assign_fitness(pop)
    for sol in pop:
        assert sol.strength >= 0
        assert sol.raw_fitness >= 0
        assert sol.density > 0
        assert sol.spea2_fitness >= 0


def test_nondominated_raw_fitness_less_than_1(alg, ds):
    # Create one clearly dominated and one dominant solution
    dominant = Solution(ds, selected=np.zeros(ds.num_requirements, dtype=np.int8))
    dominated = Solution(ds, selected=np.zeros(ds.num_requirements, dtype=np.int8))
    dominant.cost, dominant.satisfaction, dominant.fairness_variance = 0.1, 0.9, 0.001
    dominated.cost, dominated.satisfaction, dominated.fairness_variance = 0.9, 0.1, 0.1
    alg._assign_fitness([dominant, dominated])
    # dominant is not dominated by anyone → raw_fitness == 0 < 1
    assert dominant.raw_fitness == 0.0


def test_environmental_selection_size(alg, ds):
    pop = [Solution(ds, uniform=True) for _ in range(30)]
    alg._assign_fitness(pop)
    archive = alg._environmental_selection(pop)
    assert len(archive) <= alg.archive_size


def test_truncate_exact_size(alg, ds):
    solutions = [Solution(ds, uniform=True) for _ in range(25)]
    result = alg._truncate(solutions, 15)
    assert len(result) == 15


def test_run_returns_nonempty(alg):
    result = alg.run()
    assert len(result["population"]) > 0


def test_run_pareto_nondominated(ds):
    alg = SPEA2(ds, pop_size=30, archive_size=30, max_generations=10, seed=7)
    front = alg.run()["population"]
    for a in front:
        for b in front:
            if a is not b:
                assert not b.dominates(a)


def test_run_timing(alg):
    assert alg.run()["time"] > 0


def test_mating_selection_size(alg, ds):
    pool = [Solution(ds, uniform=True) for _ in range(30)]
    alg._assign_fitness(pool)
    selected = alg._mating_selection(pool)
    assert len(selected) == alg.pop_size
