# `scripts/`

Jupyter notebooks. The `tutorial_*.ipynb` series is a numbered, hands-on
walkthrough of the package meant to be read in order; `benchmarking.ipynb` is
an ad-hoc scratch notebook, not part of the series.

## Tutorial series

1. **[`tutorial.ipynb`](tutorial.ipynb)** — Getting started: building a
   Hamiltonian, constructing an initial state, and running naive time
   evolution.
2. **[`tutorial_evolution_methods.ipynb`](tutorial_evolution_methods.ipynb)**
   — the available time-evolution solvers (Krylov, randomized Krylov) and an
   introductory accuracy comparison between them.
3. **[`tutorial_benchmarking.ipynb`](tutorial_benchmarking.ipynb)** — how to
   benchmark runtime and memory scaling across qubit counts and methods using
   the `Benchmark` class.
4. **[`tutorial_parameter_study.ipynb`](tutorial_parameter_study.ipynb)** —
   how to run a `parameter_study` sweep across `krylov_dim`, sketch ratio, and
   evolution time.

## Other notebooks

- **[`tutorial_visualizations.ipynb`](tutorial_visualizations.ipynb)** — for
  after the tutorials: extra, less structured ways to visualize run results.
- **[`benchmarking.ipynb`](benchmarking.ipynb)** — scratch notebook used for
  ad-hoc benchmarking experiments; not part of the numbered series.

## Related

- The library code these notebooks exercise lives in [`src/`](../src/) and
  [`utils/`](../utils/).
