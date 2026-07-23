# randquantkry

**A RandNLA-accelerated Krylov subspace solver for quantum time evolution.**

Simulating quantum dynamics classically hits an exponential wall: the state vector of an
*n*-qubit system lives in a 2ⁿ-dimensional space, and exact methods scale accordingly. This
project explores how far that wall can be pushed back using **randomized numerical linear
algebra (RandLA)** combined with **Krylov subspace methods** and, more specifically, *why*
that reach is bounded by the same physics that governs the problem in the first place:
[Lieb–Robinson bounds](https://en.wikipedia.org/wiki/Lieb%E2%80%93Robinson_bounds) limit how
fast information spreads through a local Hamiltonian, which in turn sets how large a Krylov
subspace is actually needed for a given system size and simulation time.

Rather than diagonalizing a dense N×N matrix (`naive`) or paying full O(m²N) reorthogonalization
costs (`krylov`), this package builds Krylov subspaces using **sparse matrix-vector products**
and a **fast randomized sketch** (Güttel & Schweitzer's sketched-FOM, with a truncated
reorthogonalization made numerically safe by a periodic sketch-based correction). The result
is a method that reaches the same accuracy at a fraction of the memory and, once system size
and Krylov dimension are large enough, a fraction of the runtime too.

## Key features

- **Sparse Hamiltonian construction** (`hamiltonian.py`) — build arbitrary Pauli-string
  Hamiltonians from a compact term list, with full input validation.
- **Five time-evolution methods** (`evolution.py`, `trotter.py`):
  - `naive` — exact diagonalization (ground truth, feasible only up to ~14 qubits)
  - `krylov` — deterministic Lanczos, full reorthogonalization
  - `randomized_krylov` — sketched Krylov (fast SRTT/DCT sketch, truncated reorthogonalization,
    sketch-QR correction)
  - `randomized_krylov_gaussian` — the same algorithm with a dense Gaussian sketch, kept as an
    explicit comparison baseline (see *Known limitations* below)
  - `trotter` — first/second-order Trotter-Suzuki product formula (via Qiskit), representing
    what an actual quantum computer would run
- **Benchmark suite** (`benchmark.py`, class `QuantumBenchmark`) — runtime and memory sweeps
  across system size, plus a verification/validation split:
  - `verify_consistency` — internal sanity checks (state norm ≈ 1, no NaN/Inf) that don't need
    ground truth
  - `validate_accuracy` — comparison against exact diagonalization or a self-convergence proxy
    beyond the exact-diagonalization qubit limit
- **Parameter study** (`parameter_study.py`) — a tidy `pandas`-based sweep across `n_qubits`,
  `krylov_dim`, sketch-to-Krylov-dimension ratio, and evolution time, feeding directly into the
  Lieb–Robinson analysis (minimal Krylov dimension `m*` needed for a target accuracy, as a
  function of system size and time).
- **Crash-safe, reproducible runs** (`utils/run_io.py`) — every parameter-study run writes its
  configuration, package versions, and results incrementally to disk, so a crash partway through
  a long sweep doesn't lose what was already computed.
- **CI/CD** — GitLab pipeline with a fast import sanity check, the full `pytest` suite with
  coverage, and longer verification/convergence studies gated to `main`. Testable locally before
  every push with [`gitlab-ci-local`](https://github.com/firecow/gitlab-ci-local).

## Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible dependency
management.

```bash
git clone <repo-url>
cd randquantkry
uv sync --extra dev
```

That's it — `uv` reads `pyproject.toml` and the committed `uv.lock`, and installs an
environment that exactly matches what CI runs.

If you'd rather use plain `pip`:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Quick start

### Running the benchmark suite from the command line

```bash
uv run python -m scripts.benchmarking \
    --n-qubits 6 8 10 12 \
    --krylov-dim 20 \
    --num-random-vectors 41 \
    --time 0.5 \
    --methods krylov randomized_krylov
```

`Benchmark` takes a single reference `hamiltonian`/`initial_state`; the script builds one at
your first `--n-qubits` value and resizes it internally (via `reconstruct_hamiltonian`) for
every other size in the sweep. See `scripts/benchmarking.py --help` for all options.

### Running a parameter study

```bash
uv run python -m scripts.parameter_studying \
    --n-qubits 8 10 12 \
    --krylov-dim 10 20 40 \
    --s-ratio 1.5 2.0 3.0 \
    --time 0.5 1.0 \
    --methods krylov randomized_krylov \
    --output-dir results
```

Sweeps `n_qubits × krylov_dim × sketch-to-Krylov-dimension ratio × time`, writing results
**incrementally** to `results/run_<timestamp>/results.jsonl` as each point is computed (see
`utils/run_io.py`) — if the sweep is interrupted or crashes partway through, everything
computed so far is still on disk and loadable, along with a `metadata.json` recording the
exact configuration and package versions used, for reproducibility.

### Interactive walkthrough

For a guided, notebook-style introduction to the package — building a Hamiltonian, running each
evolution method, comparing accuracy and speed — see **`scripts/tutorial.ipynb`**. It's the
recommended starting point if you're new to the codebase, since it walks through the reasoning
behind each method rather than just the API.

## Project structure

```
randquantkry/
├── src/randquantkry/
│   ├── hamiltonian.py       # Pauli-string Hamiltonian construction + resizing
│   ├── evolution.py         # naive, krylov, randomized_krylov(_gaussian)
│   ├── trotter.py           # Trotter-Suzuki product formula (via Qiskit)
│   ├── benchmark.py         # Benchmark: runtime/memory sweeps, verify/validate, parameter_study
│   ├── plotting.py          # consistent styling across all result plots
│   └── utils/
│       ├── logging_config.py     # structured console + JSON-lines logging
│       ├── memory_estimate.py    # analytical memory prediction before running
│       ├── convergence_check.py  # self-convergence ground-truth proxy
│       └── run_io.py             # crash-safe, reproducible run persistence
├── scripts/
│   ├── benchmarking.py       # CLI: runtime/memory benchmark
│   ├── parameter_studying.py # CLI: multi-factor parameter study
│   └── tutorial.ipynb        # interactive walkthrough
├── tests/                     # pytest suite (hamiltonian, evolution, benchmark, verification)
├── results/                    # timestamped parameter-study output (gitignored, see run_io.py)
├── pyproject.toml
├── uv.lock
└── .gitlab-ci.yml
```

## Testing

```bash
uv run pytest --cov=randquantkry
```

To test the CI pipeline itself locally before pushing (catches YAML/config issues without
waiting for a real pipeline run):

```bash
gitlab-ci-local              # full pipeline
gitlab-ci-local unit-tests   # a single job
```

## Known limitations

Being upfront about these, since they came out of real debugging during development:

- **`trotter` is not ground truth.** It's a genuine approximation method (converging to the
  exact result as `num_steps` grows), included specifically because it's the algorithm real
  quantum hardware runs — not because it's exact. `validate_accuracy` will flag this explicitly
  when Trotter is used as the comparison baseline above the exact-diagonalization qubit limit.
- **`num_random_vectors` must exceed `krylov_dim`** (roughly `2*krylov_dim + 1`) for the
  sketch-QR correction to be well-defined, and can never exceed `2**n_qubits`. Both `krylov_dim`
  and `num_random_vectors` are automatically clamped with a logged warning if a sweep would
  otherwise request something outside this range.
- **`randomized_krylov_gaussian` is a deliberate baseline, not a recommended method.** It uses a
  dense Gaussian sketch instead of the fast SRTT/DCT sketch — theoretically similar accuracy,
  but with a genuinely worse memory footprint (O(s·N) for the sketch matrix itself) and, once
  `H` is sparse, no real speed advantage over deterministic Krylov. Kept for comparison, not
  for production use.
- **Memory grows with both `n_qubits` and `krylov_dim`.** Use
  `utils.memory_estimate.estimate_memory_gb` to check feasibility *before* running a large sweep,
  not after.

## References

This implementation is built directly on:

- Güttel, S., & Schweitzer, M. (2023). *Randomized sketching for Krylov approximations of
  large-scale matrix functions.* SIAM Journal on Matrix Analysis and Applications.
- Balabanov, O., & Grigori, L. (2022). *Randomized Gram-Schmidt process with application to GMRES.*
- Martinsson, P.-G., & Tropp, J. A. (2020). *Randomized numerical linear algebra: Foundations
  and algorithms.* Acta Numerica.
- Tropp, J. A. (2011). *Improved analysis of the subsampled randomized Hadamard transform.*

## License

MIT — see `LICENSE`.