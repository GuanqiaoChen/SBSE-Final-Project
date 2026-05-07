"""Unit tests for the Solution class."""
import numpy as np
import pytest

from common.dataset import Dataset
from common.solution import Solution


@pytest.fixture
def ds():
    return Dataset("a1")


def test_evaluate_zero_selection(ds):
    sol = Solution(ds, selected=np.zeros(ds.num_requirements, dtype=np.int8))
    assert sol.cost == 0.0
    assert sol.satisfaction == 0.0
    assert sol.fairness_variance == 0.0


def test_evaluate_full_selection(ds):
    sol = Solution(ds, selected=np.ones(ds.num_requirements, dtype=np.int8))
    assert 0.0 < sol.cost <= 1.0
    assert 0.0 < sol.satisfaction <= 1.0
    assert sol.fairness_variance >= 0.0


def test_evaluate_normalised_bounds(ds):
    for _ in range(20):
        sol = Solution(ds, uniform=True)
        assert 0.0 <= sol.cost <= 1.0, "cost out of range"
        assert 0.0 <= sol.satisfaction <= 1.0, "satisfaction out of range"
        assert sol.fairness_variance >= 0.0, "variance negative"


def test_dominates_basic(ds):
    # Build two solutions with controlled objectives
    a = Solution(ds, selected=np.zeros(ds.num_requirements, dtype=np.int8))
    b = Solution(ds, selected=np.zeros(ds.num_requirements, dtype=np.int8))

    # Force objective values directly
    a.cost, a.satisfaction, a.fairness_variance = 0.3, 0.7, 0.01
    b.cost, b.satisfaction, b.fairness_variance = 0.5, 0.6, 0.02

    assert a.dominates(b), "a should dominate b"
    assert not b.dominates(a), "b should not dominate a"


def test_dominates_equal(ds):
    a = Solution(ds, selected=np.zeros(ds.num_requirements, dtype=np.int8))
    b = Solution(ds, selected=np.zeros(ds.num_requirements, dtype=np.int8))
    a.cost = b.cost = 0.5
    a.satisfaction = b.satisfaction = 0.5
    a.fairness_variance = b.fairness_variance = 0.05
    assert not a.dominates(b)
    assert not b.dominates(a)


def test_clone_independence(ds):
    sol = Solution(ds, uniform=True)
    clone = sol.clone()
    clone.selected[0] ^= 1
    assert not np.array_equal(sol.selected, clone.selected)


def test_uniform_nonempty(ds):
    for _ in range(50):
        sol = Solution(ds, uniform=True)
        assert np.any(sol.selected), "uniform solution must select at least one requirement"


def test_objectives_min_negates_satisfaction(ds):
    sol = Solution(ds, uniform=True)
    c, neg_sat, fv = sol.objectives_min()
    assert c == sol.cost
    assert neg_sat == -sol.satisfaction
    assert fv == sol.fairness_variance
