# RandQuantKry

![pipeline status](https://img.shields.io/badge/pipeline-passed-green)
![docs](https://img.shields.io/badge/docs-latest-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)

**A RandNLA-accelerated Krylov subspace solver for quantum time evolution.**

<p align="center">
  <img src="output/multi_k_psi_convergence_rotate.gif" width="600" alt="Krylov Convergence">
</p>

Simulating quantum dynamics classically hits an exponential wall: the state vector of an
*n*-qubit system lives in a 2ⁿ-dimensional space, and exact methods scale accordingly. This
project explores how far that wall can be pushed back using **randomized numerical linear
algebra (RandLA)** combined with **Krylov subspace methods**.

<p align="center">
  <img src="output/Memory20qs.png" alt="Memory vs. System Size" width="500">
</p>

Rather than diagonalizing a dense N×N matrix (`naive`) or paying full O(m²N) reorthogonalization
costs (`krylov`), this package builds Krylov subspaces using **sparse matrix-vector products**
and a **fast randomized sketch** (`randomized_krylov`) using Güttel & Schweitzer's sketched-FOM,
with a truncated reorthogonalization made numerically safe by a periodic sketch-based
correction.

## Key features

| Method | Description |
|---|---|
| `naive` | Exact diagonalization — ground truth, feasible only up to ~14 qubits |
| `krylov` | Deterministic Lanczos with full reorthogonalization |
| `randomized_krylov` | Sketched Krylov (fast SRTT/DCT sketch, truncated reorthogonalization, sketch-QR correction) |
| `randomized_krylov_gaussian` | Same algorithm with a dense Gaussian sketch — comparison baseline only |
| `trotter` | First/second-order Trotter-Suzuki product formula (via Qiskit) |

Plus a full **benchmark suite** (runtime/memory sweeps, `verify_consistency`, `validate_accuracy`,
`parameter_study`) and **reproducible run tracking** that logs configuration, package versions,
and results incrementally.


## Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible dependency
management.

```bash
git clone <repo-url>
cd randquantkry
uv sync --extra dev
```

If you prefer `pip`:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux / Mac
pip install -e ".[dev]"
```

## Where to start ?

### Interactive Walkthroughs & Tutorials

For hands-on, step-by-step introductions to the package, explore the interactive Jupyter notebooks in the **[`scripts`](/scripts/)** directory:

1. **[`scripts/tutorial.ipynb`](scripts/tutorial.ipynb)** - **Getting Started**: An overview covering Hamiltonian construction and naive quantum state evolution.
<p align="center">
  <img src="output/tutorial1.png" alt="Hamiltonian vs Unitary" width="500">
</p>

2. **[`scripts/tutorial_evolution_methods.ipynb`](scripts/tutorial_evolution_methods.ipynb)** - **Evolution Algorithms**: Dive into the available time-evolution solvers (Krylov, Randomized Krylov) and a introductory comparison of accuracy

3. **[`scripts/tutorial_benchmarking.ipynb`](scripts/tutorial_benchmarking.ipynb)** - **Benchmarking Workflows**: Shows how to benchmark runtime and memory scaling across different qubit sizes and methods.

<p align="center">
  <img src="output/Memory_vs_SystemSize.png" alt="Memory vs System Size" width="45%" style="display: inline-block;">
  <img src="output/Runtime_vs_SystemSize.png" alt="Runtime vs System Size" width="45%" style="display: inline-block;">
</p>

4. **[`scripts/tutorial_parameter_study.ipynb`](scripts/tutorial_parameter_study.ipynb)** - **Parameter Sweeps & Analysis**: How to run multi-variable parameter grids, save structured run artifacts, load JSONL logs, and plot post-processing metrics.

5. **[`scripts/tutorial_visualizations.ipynb`](scripts/tutorial_visualizations.ipynb)** - **Fun Visuals for Parameter Study**: How to create bubble plots and video convergence studies.

<table align="center">
  <tr>
    <td align="center" width="45%">
      <img src="output/Krylov_err_vs_time.png" alt="Bubble Plot Parameter Study Krylov" width="100%">
    </td>
    <td align="center" width="45%">
      <img src="output/multi_k_psi_convergence.gif" alt="Krylov Convergence Study" width="100%">
      <br><br>
      <img src="output/rel_error_convergence.gif" alt="Krylov Convergence Study Relative Error" width="100%">
    </td>
  </tr>
</table>

## Project structure

```
randquantkry/
├── src/
│   ├── hamiltonian.py       # Pauli-string Hamiltonian construction + resizing
│   ├── evolution.py         # naive, krylov, randomized_krylov(_gaussian)
│   ├── trotter.py           # Trotter-Suzuki product formula (via Qiskit)
│   ├── benchmark.py         # Benchmark: runtime/memory sweeps, verify/validate, parameter_study
│   └── plotting.py          # consistent styling across all result plots
├── utils/
│   ├── logging_config.py              # structured console + JSON-lines logging
│   ├── meta_metadata_file_generator.py # maps run_ids to metadata
│   └── reproducible_runs.py           # unique run_ids and output directories
├── scripts/                 # tutorial notebooks
├── tests/                   # pytest suite
├── results/                 # timestamped parameter-study output
└── docs/                    # this documentation
```

## References

- Güttel, S., & Schweitzer, M. (2023). *Randomized sketching for Krylov approximations of
  large-scale matrix functions.* SIAM Journal on Matrix Analysis and Applications.
- Balabanov, O., & Grigori, L. (2022). *Randomized Gram-Schmidt process with application to GMRES.*
- Martinsson, P.-G., & Tropp, J. A. (2020). *Randomized numerical linear algebra: Foundations
  and algorithms.* Acta Numerica.
- Tropp, J. A. (2011). *Improved analysis of the subsampled randomized Hadamard transform.*

## License

MIT — see [`LICENSE`](https://gitlab.com/<namespace>/randquantkry/-/blob/main/LICENSE).