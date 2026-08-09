# `src/`

Core library code for `randquantkry`: Hamiltonian construction, time-evolution
solvers, benchmarking, and result visualization.

## Modules

- **`hamiltonian.py`** — build a `SparsePauliOp` Hamiltonian from a list of
  Pauli terms (`build_hamiltonian`), construct an initial computational-basis
  state (`build_initial_state`), and reconstruct/validate a Hamiltonian from
  its terms (`reconstruct_hamiltonian`).

- **`evolution.py`** — the time-evolution solvers, all sharing the signature
  `(hamiltonian, initial_state, time, ...) -> evolved_state`:
  - `time_evolve_naive` — exact diagonalization (ground truth, feasible up to
    ~14 qubits).
  - `time_evolve_krylov` — deterministic Lanczos with full reorthogonalization.
  - `time_evolve_randomized_krylov` — sketched Krylov (fast SRTT/DCT sketch,
    truncated reorthogonalization, periodic sketch-QR correction).
  - `time_evolve_randomized_krylov_gaussian` — the same algorithm with a dense
    Gaussian sketch, kept as an explicit comparison baseline.
  - `time_evolve_randomized_krylov_gramschmidt` — variant using Gram-Schmidt
    reorthogonalization.

- **`trotter.py`** — `time_evolve_trotter`: first/second-order Trotter-Suzuki
  product formula (via Qiskit), representing what an actual quantum computer
  would run.

- **`benchmark.py`** — the `Benchmark` class: runtime/memory sweeps across
  system size and evolution method, plus `verify_consistency` (state norm ≈ 1,
  no NaN/Inf), `validate_accuracy` (comparison against exact diagonalization or
  a self-convergence proxy beyond the exact-diagonalization qubit limit), and
  `parameter_study` (a `pandas`-based sweep across `n_qubits`, `krylov_dim`,
  sketch-to-Krylov-dimension ratio, and evolution time).

- **`experiment_benchmark.py`** — higher-level experiment drivers
  (`experiment_benchmark_V2`, `sweep_num_random_vectors`) built on top of
  `benchmark.py` for specific sketch-dimension sweeps.

- **`post_processing.py`** — turns the plain dicts/DataFrames produced by
  `Benchmark` into styled matplotlib figures (`plot_runtime`, `plot_memory`,
  `plot_parameter_study`). Deliberately kept separate from `benchmark.py`:
  computation is expensive and stable, plotting is cheap and gets tweaked
  constantly, and `benchmark.py` stays free of any matplotlib dependency so it
  can be tested headlessly in CI.

- **`observables.py`** — inspection and plotting helpers for quantum states
  and operators: `print_quantum_state`, `display_quantum_state_latex`,
  `plot_state_histogram`, `plot_hamiltonian_heatmap`, `plot_operator_sparsity`,
  `plot_operator_subblock`, `plot_unitary_matrix`.

## Related

- Shared infrastructure (logging, reproducible run I/O) lives in
  [`utils/`](../utils/).
- Runnable tutorials/walkthroughs of this code live in
  [`scripts/`](../scripts/).
- Tests live in [`tests/`](../tests/).
