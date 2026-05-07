"""SBSE Final Project: MONRP algorithm comparison."""
import argparse
import json
import os

from common.dataset import Dataset
from my_nsgaii.nsga2 import NSGA2
from my_spea2.spea2 import SPEA2
from UCLM_SIMD.genetic.nsgaii.nsgaii_algorithm import NSGAIIAlgorithm as UCLMNSGA2
from validations.pipeline import run_experiment
from validations.plots import (
    hv_time_scatter,
    metric_bars,
    metric_boxplots,
    normalized_radar,
    pareto_2d,
    pareto_3d,
    presentation_notes,
    ranking_heatmap,
    win_count_bars,
)


ALL_DATASETS = ["a1", "a2", "a3", "a4", "c1", "c2", "c3", "c4", "d1", "d2", "d3", "d4"]
SMALL_DATASETS = ["a1", "a2", "a3", "a4"]


def _parse():
    parser = argparse.ArgumentParser(
        description="MONRP comparison: NSGA-II, SPEA-2, and UCLM NSGA-II"
    )
    parser.add_argument("--dataset", default="a1", help="Dataset name [default: a1]")
    parser.add_argument("--all", action="store_true", help="Run on all 12 datasets")
    parser.add_argument("--small", action="store_true", help="Run on the 4 small (a*) datasets")
    parser.add_argument("--runs", type=int, default=5, help="Independent runs [5]")
    parser.add_argument("--pop", type=int, default=100, help="Population size [100]")
    parser.add_argument("--gen", type=int, default=250, help="Max generations [250]")
    parser.add_argument("--cx", type=float, default=0.9, help="Crossover probability [0.9]")
    parser.add_argument("--mut", type=float, default=0.1, help="Mutation probability [0.1]")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed [42]")
    parser.add_argument("--deps", action="store_true", help="Repair dependency violations")
    parser.add_argument("--plot", action="store_true", help="Show interactive plots")
    parser.add_argument(
        "--save-plots",
        dest="save_plots",
        default=None,
        help="Directory to save comparison plots as PNG files",
    )
    parser.add_argument("--output", default=None, help="Save aggregate results to JSON file")
    return parser.parse_args()


def _visualise(dataset_name, args):
    """Generate Pareto-front plots for one dataset."""
    dataset = Dataset(dataset_name)
    n_alg = NSGA2(
        dataset=dataset,
        pop_size=args.pop,
        max_generations=args.gen,
        crossover_prob=args.cx,
        mutation_prob=args.mut,
        tackle_dependencies=args.deps,
        seed=args.seed,
    )
    s_alg = SPEA2(
        dataset=dataset,
        pop_size=args.pop,
        archive_size=args.pop,
        max_generations=args.gen,
        crossover_prob=args.cx,
        mutation_prob=args.mut,
        tackle_dependencies=args.deps,
        seed=args.seed,
    )
    u_alg = UCLMNSGA2(
        execs=1,
        dataset_name=dataset_name,
        random_seed=args.seed,
        population_length=args.pop,
        max_generations=args.gen,
        crossover_prob=args.cx,
        mutation_prob=args.mut,
        tackle_dependencies=args.deps,
    )

    fronts = {
        "nsga2": n_alg.run()["population"],
        "spea2": s_alg.run()["population"],
        "uclm_nsga2": u_alg.run()["population"],
    }

    save_dir = args.save_plots
    pareto_2d(
        fronts,
        title=f"Cost/Satisfaction Pareto Front - {dataset_name}",
        save_path=os.path.join(save_dir, f"{dataset_name}_2d.png") if save_dir else None,
    )
    pareto_3d(
        fronts,
        title=f"3-D Pareto Front - {dataset_name}",
        save_path=os.path.join(save_dir, f"{dataset_name}_3d.png") if save_dir else None,
    )


def main():
    args = _parse()

    if args.all:
        datasets = ALL_DATASETS
    elif args.small:
        datasets = SMALL_DATASETS
    else:
        datasets = [args.dataset]

    shared = dict(
        num_runs=args.runs,
        pop_size=args.pop,
        max_generations=args.gen,
        crossover_prob=args.cx,
        mutation_prob=args.mut,
        tackle_dependencies=args.deps,
        base_seed=args.seed,
        verbose=True,
    )

    all_results = []
    for dataset_name in datasets:
        result = run_experiment(dataset_name, **shared)
        all_results.append(result)

        if args.plot or args.save_plots:
            _visualise(dataset_name, args)

    if len(all_results) > 1 and (args.plot or args.save_plots):
        save_dir = args.save_plots
        for metric in ("hypervolume", "spread", "spacing", "time"):
            metric_bars(
                all_results,
                metric=metric,
                save_path=os.path.join(save_dir, f"compare_{metric}.png") if save_dir else None,
            )

    if args.plot or args.save_plots:
        save_dir = args.save_plots
        for metric in ("hypervolume", "spread", "spacing", "time"):
            metric_boxplots(
                all_results,
                metric=metric,
                save_path=os.path.join(save_dir, f"boxplot_{metric}.png") if save_dir else None,
            )
        ranking_heatmap(
            all_results,
            save_path=os.path.join(save_dir, "rank_heatmap.png") if save_dir else None,
        )
        hv_time_scatter(
            all_results,
            save_path=os.path.join(save_dir, "hypervolume_vs_time.png") if save_dir else None,
        )
        normalized_radar(
            all_results,
            save_path=os.path.join(save_dir, "normalized_profile.png") if save_dir else None,
        )
        win_count_bars(
            all_results,
            save_path=os.path.join(save_dir, "win_count.png") if save_dir else None,
        )
        if save_dir:
            presentation_notes(
                all_results,
                save_path=os.path.join(save_dir, "speaker_notes.md"),
            )

    if args.output:
        serialisable = [
            {
                "dataset": r["dataset"],
                "config": r["config"],
                "nsga2_summary": r["nsga2_summary"],
                "spea2_summary": r["spea2_summary"],
                "uclm_nsga2_summary": r["uclm_nsga2_summary"],
            }
            for r in all_results
        ]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2)
        print(f"\nResults saved -> {args.output}")


if __name__ == "__main__":
    main()
