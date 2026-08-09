# `tests/`

Pytest suite for `randquantkry`, split into unit and integration tests. Shared
fixtures (small Hamiltonians and initial states, sized so exact
diagonalization is cheap enough to serve as ground truth) live in
`conftest.py`.

```bash
uv run pytest
```

## `unit/`

One test module per `src/` module, exercising it in isolation:

- **`test_hamiltonian.py`** — `build_hamiltonian`, `build_initial_state`,
  `reconstruct_hamiltonian`, `hash_terms`: correct dimensions, Hermiticity,
  input validation, and roundtrip fidelity.
- **`test_evolution.py`** — every `time_evolve_*` solver: agreement with
  `scipy.linalg.expm` / exact diagonalization, norm preservation, seed
  determinism, and behavior at early Krylov breakdown or insufficient memory.
- **`test_trotter.py`** — `time_evolve_trotter`: norm preservation,
  convergence to the exact result as step count grows, and that first- vs
  second-order Trotter behave as expected (approximate, not exact; higher
  order more accurate at fixed step count).
- **`test_benchmark.py`** — the `Benchmark` class: method validation and the
  `krylov_dim` / `num_random_vectors` auto-resolution/clamping logic.
- **`test_post_processing.py`** / **`test_obersvables.py`** — the plotting
  data-prep helpers (`_filtered_xy`, `_sort_labels`, `_df_to_results`), kept
  free of matplotlib so they run headlessly.

## `integration/`

- **`test_parameter_study_pipeline.py`** — runs `Benchmark.parameter_study`
  end-to-end and checks the resulting DataFrame: expected columns, qubit/dim
  combinations correctly skipped, and `rel_error` present whenever exact
  diagonalization is feasible.

## Related

- Code under test lives in [`src/`](../src/) and [`utils/`](../utils/).
