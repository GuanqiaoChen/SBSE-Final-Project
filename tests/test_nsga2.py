"""Unit tests for NSGA-II."""
import numpy as np
import pytest

from common.dataset import Dataset
from common.solution import Solution
from my_nsgaii.nsga2 import NSGA2


@pytest.fixture
def ds():
    return Dataset("a1")


@pytest.fixture
def alg(ds):
    return NSGA2(ds, pop_size=20, max_generations=5, seed=0)


def test_init_population_size(alg):
    pop = alg._init_population()
    assert len(pop) == 20


def test_init_population_all_valid(alg):
    pop = alg._init_population()
    for sol in pop:
        assert np.any(sol.selected)


def test_fast_nondominated_sort_front0_nondominated(alg):
    pop = alg._init_population()
    fronts = alg._fast_nondominated_sort(pop)
    f0 = fronts[0]
    # No solution in front 0 should be dominated by any other solution in front 0
    for a in f0:
        for b in f0:
            if a is not b:
                assert not b.dominates(a), "front-0 solution is dominated"


def test_fast_nondominated_sort_covers_all(alg):
    pop = alg._init_population()
    fronts = alg._fast_nondominated_sort(pop)
    total = sum(len(f) for f in fronts)
    assert total == len(pop)


def test_crowding_distance_boundary_inf(alg):
    pop = alg._init_population()
    fronts = alg._fast_nondominated_sort(pop)
    f0 = fronts[0]
    if len(f0) >= 3:
        alg._crowding_distance(f0)
        sorted_f = sorted(f0, key=lambda s: s.cost)
        assert sorted_f[0].crowding_distance == float('inf') or sorted_f[-1].crowding_distance == float('inf')


def test_crowding_distance_small_front(alg, ds):
    front = [Solution(ds, uniform=True), Solution(ds, uniform=True)]
    alg._crowding_distance(front)
    for sol in front:
        assert sol.crowding_distance == float('inf')


def test_run_returns_nonempty_pareto(alg):
    result = alg.run()
    assert "population" in result
    assert len(result["population"]) > 0


def test_run_pareto_nondominated(ds):
    alg = NSGA2(ds, pop_size=30, max_generations=10, seed=42)
    front = alg.run()["population"]
    for a in front:
        for b in front:
            if a is not b:
                assert not b.dominates(a)


def test_run_timing(alg):
    result = alg.run()
    assert result["time"] > 0


def test_tournament_select_size(alg):
    pop = alg._init_population()
    fronts = alg._fast_nondominated_sort(pop)
    for f in fronts:
        alg._crowding_distance(f)
    selected = alg._tournament_select(pop)
    assert len(selected) == len(pop)
