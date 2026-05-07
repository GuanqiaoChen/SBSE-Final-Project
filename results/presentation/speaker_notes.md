# Speaker Notes

## Slide: Experimental Setup
- We compare three algorithms on the shared MONRP objectives: cost and satisfaction.
- Datasets included in this run: a1.
- Hypervolume and spread are interpreted as higher is better; spacing and runtime are lower is better.
- The custom NSGA-II and SPEA-2 also optimize fairness variance, but UCLM NSGA-II is 2-objective, so the main comparison uses the shared 2D space.

## Slide: 2D Pareto Front
- This plot shows the tradeoff between implementation cost and stakeholder satisfaction.
- Points farther toward low cost and high satisfaction are preferable.
- A wider front means the algorithm gives decision makers more alternatives rather than a single answer.

## Slide: 3D Pareto Front
- This slide is only for the custom NSGA-II and SPEA-2, because they include the fairness objective.
- The third axis shows fairness variance; lower variance means satisfaction is distributed more evenly across stakeholders.
- Use this slide to explain the extension beyond the baseline UCLM formulation.

## Slide: Metric Bar Charts
- Hypervolume summarizes both convergence and coverage of the Pareto front.
- Spread shows how much of the tradeoff surface is covered.
- Spacing shows how evenly solutions are distributed; lower spacing is better.
- Runtime compares computational cost under the same population, generation, and seed settings.

## Slide: Metric Boxplots
- Boxplots show stability across independent runs.
- A high median with a compact box is desirable for hypervolume.
- For spacing and runtime, lower medians and compact boxes indicate more reliable behavior.

## Slide: Rank Heatmap
- Each cell is the rank for one algorithm on one dataset and metric.
- Rank 1 is the best for that row.
- This gives a compact overview of consistency instead of focusing on a single dataset.

## Slide: Hypervolume vs Runtime
- This plot shows the quality-efficiency tradeoff.
- The ideal region is upper-left: high hypervolume with low runtime.
- It helps justify whether a quality improvement is worth the extra computational time.

## Slide: Normalized Profile Radar
- Each axis is normalized so the best observed algorithm scores near the outside.
- This is a high-level summary, not a replacement for the raw metric charts.
- Use it to explain overall strengths and weaknesses quickly.

## Slide: Win Count
- This counts how often each algorithm has the best mean metric value across datasets.
- It is useful as a closing summary, but should be interpreted together with the metric magnitudes.
