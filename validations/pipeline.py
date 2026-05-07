"""Experiment pipeline: run NSGA-II and SPEA-2 on benchmark datasets and compare."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from common.dataset import Dataset
from my_nsgaii.nsga2 import NSGA2
from my_spea2.spea2 import SPEA2
from UCLM_SIMD.genetic.nsgaii.nsgaii_algorithm import NSGAIIAlgorithm as UCLMNSGA2
import validations.metrics as metrics


def run_experiment(
    dataset_name: str,
    num_runs: int = 5,
    pop_size: int = 100,
    max_generations: int = 250,
    crossover_prob: float = 0.9,
    mutation_prob: float = 0.1,
    archive_size: Optional[int] = None,
    tackle_dependencies: bool = False,
    base_seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run both algorithms and collect aggregate quality metrics."""
    dataset = Dataset(dataset_name)
    if archive_size is None:
        archive_size = pop_size

    if verbose:
        print(f"\n{'=' * 70}")
        print(
            f"Dataset: {dataset_name}  "
            f"({dataset.num_requirements} requirements, "
            f"{dataset.num_stakeholders} stakeholders)"
        )
        print(f"{'=' * 70}")

    results: Dict[str, Any] = {
        "dataset": dataset_name,
        "config": {
            "pop_size": pop_size,
            "max_generations": max_generations,
            "crossover_prob": crossover_prob,
            "mutation_prob": mutation_prob,
            "metric_space": "2D shared cost/satisfaction",
        },
        "nsga2": [],
        "spea2": [],
        "uclm_nsga2": [],
    }

    for run in range(num_runs):
        seed = base_seed + run

        alg_n = NSGA2(
            dataset=dataset,
            pop_size=pop_size,
            max_generations=max_generations,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            tackle_dependencies=tackle_dependencies,
            seed=seed,
        )
        res_n = alg_n.run()
        m_n = metrics.summary_2d(res_n["population"])
        m_n["time"] = res_n["time"]
        results["nsga2"].append(m_n)

        alg_s = SPEA2(
            dataset=dataset,
            pop_size=pop_size,
            archive_size=archive_size,
            max_generations=max_generations,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            tackle_dependencies=tackle_dependencies,
            seed=seed,
        )
        res_s = alg_s.run()
        m_s = metrics.summary_2d(res_s["population"])
        m_s["time"] = res_s["time"]
        results["spea2"].append(m_s)

        alg_u = UCLMNSGA2(
            execs=1,
            dataset_name=dataset_name,
            random_seed=seed,
            population_length=pop_size,
            max_generations=max_generations,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            tackle_dependencies=tackle_dependencies,
        )
        res_u = alg_u.run()
        m_u = metrics.summary_2d(res_u["population"])
        m_u["time"] = res_u["time"]
        results["uclm_nsga2"].append(m_u)

        if verbose:
            print(
                f"  run {run + 1}/{num_runs} | "
                f"NSGA-II  HV={m_n['hypervolume']:.4f}  "
                f"#sol={m_n['num_solutions']:3d}  t={m_n['time']:.1f}s | "
                f"SPEA-2   HV={m_s['hypervolume']:.4f}  "
                f"#sol={m_s['num_solutions']:3d}  t={m_s['time']:.1f}s | "
                f"UCLM NSGA-II  HV={m_u['hypervolume']:.4f}  "
                f"#sol={m_u['num_solutions']:3d}  t={m_u['time']:.1f}s"
            )

    results["nsga2_summary"] = _aggregate(results["nsga2"])
    results["spea2_summary"] = _aggregate(results["spea2"])
    results["uclm_nsga2_summary"] = _aggregate(results["uclm_nsga2"])

    if verbose:
        _print_summary(results)

    return results


def run_all(dataset_names: List[str], **kwargs) -> List[Dict[str, Any]]:
    return [run_experiment(name, **kwargs) for name in dataset_names]


def _aggregate(runs: List[Dict]) -> Dict:
    agg: Dict[str, float] = {}
    for key in runs[0]:
        vals = [r[key] for r in runs]
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))
    return agg


def _print_summary(results: Dict) -> None:
    print(
        f"\n  {'Metric':<22} {'NSGA-II mean +/- std':>24} "
        f"{'SPEA-2 mean +/- std':>24} {'UCLM NSGA-II mean +/- std':>30}"
    )
    print(f"  {'-' * 104}")
    for key in ["hypervolume", "spread", "spacing", "num_solutions", "time"]:
        n = results["nsga2_summary"]
        s = results["spea2_summary"]
        u = results["uclm_nsga2_summary"]
        nm = n.get(f"{key}_mean", float("nan"))
        ns = n.get(f"{key}_std", float("nan"))
        sm = s.get(f"{key}_mean", float("nan"))
        ss = s.get(f"{key}_std", float("nan"))
        um = u.get(f"{key}_mean", float("nan"))
        us = u.get(f"{key}_std", float("nan"))
        print(
            f"  {key:<22} {nm:>10.4f} +/- {ns:<10.4f}   "
            f"{sm:>10.4f} +/- {ss:<10.4f}   {um:>10.4f} +/- {us:<10.4f}"
        )
    print()
