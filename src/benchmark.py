##########################################################################
#
#             Benchmark Functions for Quantum Evolution
#
###########################################################################

# imports
import os
import threading
import tracemalloc
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import psutil
from qiskit.quantum_info import SparsePauliOp
from randquantkry.utils.logging_config import configure_logging, get_logger
from randquantkry.utils.reproducible_runs import append_result_row, start_run

# randquantkry imports
from .evolution import (
    time_evolve_krylov,
    time_evolve_naive,
    time_evolve_randomized_krylov,
    time_evolve_randomized_krylov_gaussian,
)
from .hamiltonian import build_initial_state, reconstruct_hamiltonian
from .trotter import time_evolve_trotter

log = get_logger(__name__)

ALL_METHODS = (
    "naive",
    "krylov",
    "randomized_krylov",
    "randomized_krylov_gaussian",
    "trotter",
)
MAX_NAIVE_QUBITS = 12  # adapt to your local system !!


class Benchmark:
    """
    Configurable benchmark suite comparing quantum time-evolution methods
    across a range of system sizes. Also includes a parameter study function
    for further analysis.

    Parameters
    ----------
    reconstruct_hamiltonian : callable(n_qubits, hamiltonian) -> SparsePauliOp
        Builds the Hamiltonian for a given system size. Using a factory
        (rather than a single fixed Hamiltonian) is what lets runtime/memory
        be swept over n_qubits.
    build_initial_state : callable(n_qubits, flipped_qubit) -> ndarray
        Builds the initial state for a given system size.
    time : float
        Evolution time, held fixed across the n_qubits sweep (pass a
        different QuantumBenchmark instance, or extend later, to also sweep
        time -- this is the natural hook for the Lieb-Robinson parameter
        study to plug in later).
    krylov_dim : int or callable(n_qubits) -> int
        Krylov dimension.
        an empirically-justified m*(n_qubits) from a convergence/parameter
        study, rather than one fixed guess.
    num_random_vectors : int, callable(krylov_dim) -> int, or None
        Sketch dimension for both randomized methods. If None, defaults to
        2*krylov_dim + 1 (the rule of thumb established during development:
        the sketch-QR correction requires num_random_vectors > krylov_dim).
    methods : list of str, optional
        Subset of ALL_METHODS to run. Defaults to all.
    trotter_kwargs : dict, optional
        Extra kwargs (num_steps, order) forwarded to time_evolve_trotter.
    seed : int, optional
        Seed forwarded to the randomized methods.
    """

    def __init__(
        self,
        hamiltonian: SparsePauliOp,
        initial_state,
        time: float,
        krylov_dim: int,
        num_random_vectors: int | None,
        methods=None,
        trotter_kwargs=None,
        seed=None,
        output_dir="results",
    ):
        self.H = hamiltonian
        self.initial_state = initial_state
        self.time = time
        self.krylov_dim = krylov_dim
        self.num_random_vectors = num_random_vectors
        self.methods = list(methods) if methods is not None else list(ALL_METHODS)
        unknown = set(self.methods) - set(ALL_METHODS)
        if unknown:
            raise ValueError(f"Unknown method(s) {sorted(unknown)}; valid options are {ALL_METHODS}")
        self.trotter_kwargs = trotter_kwargs or {"num_steps": 100, "order": 1}
        self.seed = seed
        self.output_dir = Path(f"../{output_dir}").resolve()

    def _resolve_krylov_dim(self, n_qubits: int):
        N = 2**n_qubits
        krylov_dim = self.krylov_dim(n_qubits) if callable(self.krylov_dim) else self.krylov_dim
        if krylov_dim >= N:
            log.warning(
                "krylov_dim exceeds or equals N=2**n_qubits; clamping to N-1 "
                "(a Krylov subspace can never exceed the full Hilbert space dimension)",
                extra={
                    "context": {
                        "requested_krylov_dim": krylov_dim,
                        "n_qubits": n_qubits,
                        "N": N,
                    }
                },
            )
            krylov_dim = max(1, N - 1)
        return krylov_dim

    def _resolve_num_random_vectors(self, krylov_dim: int, n_qubits: int):
        N = 2**n_qubits
        if self.num_random_vectors is None:
            number_rand_vectors = 2 * krylov_dim + 1
        else:
            number_rand_vectors = (
                self.num_random_vectors(krylov_dim) if callable(self.num_random_vectors) else self.num_random_vectors
            )
        if number_rand_vectors > N:
            log.warning(
                "num_random_vectors exceeds N=2**n_qubits; clamping to N "
                "(cannot sketch down to a dimension larger than the original space)",
                extra={
                    "context": {
                        "requested_num_random_vectors": number_rand_vectors,
                        "n_qubits": n_qubits,
                        "N": N,
                    }
                },
            )
            number_rand_vectors = N
        return number_rand_vectors

    def _adjust_run_nqubits(self, n_qubits: int):
        """Construct everything needed for one n_qubits run: Hamiltonian,
        initial state, and resolved krylov_dim/num_random_vectors."""
        hamiltonian = reconstruct_hamiltonian(n_qubits, self.H)
        initial_state = build_initial_state(n_qubits, flipped_qubit=0)
        krylov_dim = self._resolve_krylov_dim(n_qubits)
        number_rand_vectors = self._resolve_num_random_vectors(krylov_dim, n_qubits)
        return hamiltonian, initial_state, krylov_dim, number_rand_vectors

    def _run_method(
        self,
        method: list,
        hamiltonian: SparsePauliOp,
        initial_state,
        time: float,
        krylov_dim: int,
        number_rand_vectors: int | None,
    ):
        """Run a single method, returning (state, runtime_sec)"""
        t0 = perf_counter()
        if method == "naive":
            if hamiltonian.num_qubits <= MAX_NAIVE_QUBITS:
                _, state = time_evolve_naive(hamiltonian, initial_state, time)
            else:
                log.info(
                    "'naive' skipped...",
                    extra={"context": {"n_qubits": hamiltonian.num_qubits}},
                )
                raise ValueError(f"naive method requires n_qubits <= {MAX_NAIVE_QUBITS}, got {hamiltonian.num_qubits}")
        elif method == "krylov":
            state = time_evolve_krylov(hamiltonian, initial_state, time, krylov_dim)
        elif method == "randomized_krylov":
            state = time_evolve_randomized_krylov(
                hamiltonian,
                initial_state,
                time,
                krylov_dim,
                number_rand_vectors,
                seed=self.seed,
            )
        elif method == "randomized_krylov_gaussian":
            state = time_evolve_randomized_krylov_gaussian(
                hamiltonian,
                initial_state,
                time,
                krylov_dim,
                number_rand_vectors,
                seed=self.seed,
            )
        elif method == "trotter":
            state = time_evolve_trotter(hamiltonian, initial_state, time, **self.trotter_kwargs)
        else:
            raise ValueError(f"Unknown method {method!r}")
        runtime = perf_counter() - t0
        return state, runtime

    def calculate_rel_error(calc_state, exact_state):
        """
        Calculates relative error as $|| state - exact_state || / || exact_state||$ in the L2 norm.
        """
        exact_norm = np.linalg.norm(exact_state)
        return float(np.linalg.norm(calc_state - exact_state) / exact_norm)

    def find_converged_krylov_dim(
        evolve_fn,
        hamiltonian,
        initial_state,
        time,
        krylov_dims=None,
        tol=1e-6,
        **evolve_kwargs,
    ):
        """
        Find the smallest krylov_dim (from a candidate list) at which the
        evolved state stops changing beyond `tol`, WITHOUT needing exact ground
        truth. Use this beyond ~14 qubits, where time_evolve_naive is no longer
        feasible.

        Parameters
        ----------
        evolve_fn : callable
            time_evolve_krylov or time_evolve_randomized_krylov (or any function
            with the same (hamiltonian, initial_state, time, krylov_dim, ...) signature).
        krylov_dims : sorted list of int
            Increasing candidate values to try, e.g. [10, 20, 40, 80, 160].
        tol : float
            Convergence threshold on ||state_m - state_{m_prev}|| / ||state_m||.
        evolve_kwargs : dict
            Extra keyword arguments passed through to evolve_fn (e.g.
            num_random_vectors=..., trunc=..., seed=...).

        Returns
        -------
        dict with:
            "converged_at" : the krylov_dim where convergence was first detected
                            (or None if never converged within the given list),
            "diffs" : list of successive relative differences (for plotting),
            "states" : list of the evolved states at each krylov_dim tried.
        """
        states = []
        diffs = []
        converged_at = None

        for m in krylov_dims:
            state = evolve_fn(hamiltonian, initial_state, time, m, **evolve_kwargs)
            states.append(state)

            if len(states) > 1:
                diff = np.linalg.norm(states[-1] - states[-2]) / np.linalg.norm(states[-1])
                diffs.append(float(diff))
                if diff < tol and converged_at is None:
                    converged_at = m

        return {
            "converged_at": converged_at,
            "diffs": diffs,
            "krylov_dims_tried": list(krylov_dims),
            "states": states,
        }

    # verification: internal consistency, no ground truth needed
    def verify_consistency(self, methods=None):
        """Perform basic internal consistency checks across selected methods without requiring ground truth.

        Checks performed per method:
        - Absence of NaN or Infinite values in the evolved state vector.
        - Conservation of probability density (state norm ≈ 1.0).

        Parameters
        ----------
        methods : list of str, optional
            Subset of methods to verify. Defaults to `self.methods`.

        Returns
        -------
        dict
            A dictionary mapping method names to their check results, containing:
            - "state_norm": float or None
                The L2 norm of the evolved state vector.
            - "has_nan": bool
                True if NaNs or Infs were detected in the vector.
            - "passed": bool
                True if the state passed all sanity checks (no NaNs and norm ≈ 1.0).
        """
        results = {}
        for method in methods or self.methods:
            state, _ = self._run_method(
                method,
                self.H,
                self.initial_state,
                self.time,
                self.krylov_dim,
                self.num_random_vectors,
            )
            has_nan = bool(np.any(np.isnan(state)) or np.any(np.isinf(state)))
            if not has_nan:
                norm = float(np.linalg.norm(state))
                if np.isclose(norm, 1.0, atol=1e-6):
                    results[method] = {
                        "state_norm": norm,
                        "has_nan": has_nan,
                        "passed": True,
                    }
                else:
                    results[method] = {
                        "state_norm": norm,
                        "has_nan": has_nan,
                        "passed": False,
                    }

                log.info(
                    "verify_consistency: method checked",
                    extra={
                        "context": {
                            "method": method,
                            "n_qubits": self.H.num_qubits,
                            **{k: v for k, v in results[method].items()},
                        }
                    },
                )
            else:
                float("nan")

                results[method] = {
                    "state_norm": None,
                    "has_nan": has_nan,
                    "passed": False,
                }

            log.info(
                "verify_consistency: method checked",
                extra={
                    "context": {
                        "method": method,
                        "n_qubits": self.H.num_qubits,
                        **{k: v for k, v in results[method].items()},
                    }
                },
            )

        return results

    # Validation & Verification
    # validation: compare against ground truth, trotter and self-convergence
    def validate_accuracy(self, convergence_dims=None, tol=1e-6, methods=None):
        """Validate accuracy of quantum evolution methods against a baseline ground truth.

        The ground truth reference depends on system size:
        - Systems <= 10 qubits: Evaluated against full exact time evolution (`naive`).
        - Systems > 10 qubits: Evaluated against a self-convergence proxy state computed via
          a large-subspace Krylov method (`self_convergence`).

        Parameters
        ----------
        convergence_dims : list of int, default=[10, 20, 40, 80, 160]
            List of increasing Krylov subspace dimensions used to compute the
            self-convergence baseline when exact diagonalisation is unavailable.
        tol : float, optional
            Tolerance threshold passed to the convergence solver.
        methods : list of str, optional
            Subset of methods to validate. Defaults to `self.methods`.

        Returns
        -------
        dict
            A dictionary mapping method names to validation metrics:
            - "rel_error": float or None
                Relative error with respect to the ground truth reference.
            - "ground_truth": str
                Type of reference used ("exact" or "self_convergence").
            - "error": str, optional
                Error message string if execution failed.
        """

        if self.H.num_qubits <= 10:
            _, exact_state = time_evolve_naive(self.H, self.initial_state, self.time)  # real ground truth
            ground_truth_kind = "exact"
            exact_norm = np.linalg.norm(exact_state)

        # elif:
        #     trotter_state = time_evolve_trotter(self.H, self.initial_state, self.time, **self.trotter_kwargs)
        #     ground_truth_kind = "trotter"
        #     exact_norm = np.linalg.norm(trotter_state)

        else:
            conv = self.find_converged_krylov_dim(
                time_evolve_krylov,
                self.H,
                self.initial_state,
                self.time,
                convergence_dims,
                tol=tol,
            )
            exact_state = conv["states"][-1]  # best available proxy: largest converged run
            exact_norm = np.linalg.norm(exact_state)
            ground_truth_kind = "self_convergence"

        results = {}
        for method in methods or self.methods:
            if method == "naive" and ground_truth_kind == "exact":
                results[method] = {"rel_error": 0.0, "ground_truth": ground_truth_kind}
                continue
            try:
                state, _ = self._run_method(
                    method,
                    self.H,
                    self.initial_state,
                    self.time,
                    self.krylov_dim,
                    self.num_random_vectors,
                )
                rel_error = float(np.linalg.norm(state - exact_state) / exact_norm)
                results[method] = {
                    "rel_error": rel_error,
                    "ground_truth": ground_truth_kind,
                }
            except (
                Exception  # noqa: BLE001 —> one method failing shouldn't abort the whole validate_accuracy sweep
            ) as e:
                results[method] = {
                    "rel_error": None,
                    "ground_truth": ground_truth_kind,
                    "error": str(e),
                }

            log.info(
                "validate_accuracy: method checked",
                extra={
                    "context": {
                        "method": method,
                        "n_qubits": self.H.num_qubits,
                        **{k: v for k, v in results[method].items()},
                    }
                },
            )
        return results

    def estimate_memory_gb(self, krylov_dim: int, n_hamiltonian_terms: int, num_random_vectors: int | None):
        N = 2**self.H.num_qubits
        BYTES_PER_COMPLEX128 = 16
        sparse_H_gb = (n_hamiltonian_terms * N * (BYTES_PER_COMPLEX128 + 8)) / 1e9

        V_gb = ((krylov_dim + 1) * N * BYTES_PER_COMPLEX128) / 1e9

        result = {
            "n_qubits": self.H.num_qubits,
            "N": N,
            "sparse_H_gb": sparse_H_gb,
            "V_storage_gb": V_gb,
            "total_gb_deterministic": sparse_H_gb + V_gb,
        }

        if num_random_vectors is not None:
            s = num_random_vectors
            # SV, SAV: krylov_dim (+1) sketch vectors of length s each -- tiny
            # compared to V's O(N) vectors, since s << N.
            sketch_storage_gb = (2 * (krylov_dim + 1) * s * BYTES_PER_COMPLEX128) / 1e9
            result["sketch_storage_gb"] = sketch_storage_gb
            result["total_gb_randomized"] = sparse_H_gb + V_gb + sketch_storage_gb

        return result

    def _measure_with_rss(fn, *args, **kwargs):
        process = psutil.Process(os.getpid())
        peak = [process.memory_info().rss]
        stop = threading.Event()

        def sampler():
            while not stop.is_set():
                peak[0] = max(peak[0], process.memory_info().rss)

        t = threading.Thread(target=sampler)
        t.start()
        t0 = perf_counter()
        result = fn(*args, **kwargs)
        runtime = perf_counter() - t0
        stop.set()
        t.join()
        return result, runtime, peak[0]

    def benchmark(self, n_qubits_list, methods=None):
        """
        Combined runtime + memory sweep. Runs each (n_qubits, method) point ONCE,
        recording wall-clock runtime and theoretical memory always.

        Returns a single results dict compatible with BOTH plot_runtime(results)
        and plot_memory(results) -- no need to run the sweep twice to get both plots.

        NOTE: tracemalloc's own tracing overhead adds
        a small, real bias to the measured runtime_sec for that run. If you need
        a clean runtime number AND empirical memory, prefer two separate calls
        rather than reading runtime_sec from an empirical=True run.
        """
        methods = methods or self.methods
        seed_val = self.seed

        run_config = {
            "n_qubits_list": list(n_qubits_list),
            "time_list": [self.time],
            "methods": list(methods),
            "seed": seed_val,
        }

        run_dir = start_run(self.output_dir, run_config)
        configure_logging(json_path=run_dir / "run.log.jsonl", console_level="INFO")

        log.info(
            "benchmark: starting",
            extra={"context": {"n_qubits_list": list(n_qubits_list), "methods": methods}},
        )

        results = {
            method: {
                "n_qubits": [],
                "state": [],
                "runtime_sec": [],
                "memory_gb": [],
                "memory_rss": [],
            }
            for method in methods
        }

        for n_qubits in n_qubits_list:
            hamiltonian, initial_state, krylov_dim, number_rand_vectors = self._adjust_run_nqubits(n_qubits)

            for method in methods:
                results[method]["n_qubits"].append(n_qubits)

                # measures runtime, wrapped in tracemalloc
                measured_mem_gb = None
                try:
                    tracemalloc.start()
                    process = psutil.Process(os.getpid())
                    rss_baseline = process.memory_info().rss
                    peak_rss = [rss_baseline]
                    stop = threading.Event()

                    def sampler(stop=stop, peak_rss=peak_rss, process=process):
                        while not stop.is_set():
                            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

                    t = threading.Thread(target=sampler)
                    t.start()

                    state, runtime = self._run_method(
                        method,
                        hamiltonian,
                        initial_state,
                        self.time,
                        krylov_dim,
                        number_rand_vectors,
                    )

                    stop.set()
                    t.join()
                    _, tracemalloc_peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()

                    measured_mem_gb = tracemalloc_peak / 1e9
                    results[method]["memory_gb"].append(measured_mem_gb)
                    results[method]["memory_rss"].append(peak_rss[0] / 1e9)

                    results[method]["runtime_sec"].append(runtime)
                    results[method]["state"].append(state)
                    row = {
                        "n_qubits": n_qubits,
                        "method": method,
                        "time": self.time,
                        "krylov_dim": (
                            krylov_dim
                            if method
                            in (
                                "krylov",
                                "randomized_krylov",
                                "randomized_krylov_gaussian",
                            )
                            else None
                        ),
                        "num_random_vectors": (number_rand_vectors if "randomized" in method else None),
                        "runtime_sec": runtime,
                        "memory_gb": measured_mem_gb,
                        "memory_rss": peak_rss[0],
                        "evolved_state_norm": (float(np.linalg.norm(state)) if state is not None else None),
                    }
                    append_result_row(run_dir, row)

                    log.debug(
                        "benchmark: point complete",
                        extra={
                            "context": {
                                "method": method,
                                "n_qubits": n_qubits,
                                "runtime_sec": runtime,
                            }
                        },
                    )

                except (
                    Exception  # noqa: BLE001 —> one sweep point failing shouldn't abort the whole benchmark run
                ) as e:
                    log.warning(
                        "benchmark: method failed, recording None",
                        extra={
                            "context": {
                                "method": method,
                                "n_qubits": n_qubits,
                                "error": str(e),
                            }
                        },
                    )
                    results[method]["runtime_sec"].append(None)
                    results[method]["state"].append(None)
                    results[method]["memory_gb"].append(None)
                    results[method]["memory_rss"].append(None)
                    try:
                        tracemalloc.stop()
                    except RuntimeError:
                        log.debug("benchmark: tracemalloc.stop() called with no active trace")

                    # Record failed
                    error_row = {
                        "n_qubits": n_qubits,
                        "method": method,
                        "time": self.time,
                        "error": str(e),
                    }
                    append_result_row(run_dir, error_row)

        log.info("benchmark: complete", extra={"context": {"run_dir": str(run_dir)}})
        return results, run_dir

    def parameter_study(
        self,
        n_qubits_list: int,
        krylov_dim_list: list,
        s_ratio_list: list,
        time_list: list,
        methods=("krylov", "randomized_krylov"),
        seed=67,
        accuracy_convergence_dims=None,
    ):
        """
        Run a full parameter sweep comparing Krylov and randomized Krylov
        time evolution across system size, Krylov dimension, sketch ratio,
        and simulation time, recording runtime, peak memory, and accuracy
        for every combination.

        For each `n_qubits`, a Hamiltonian and initial state are
        reconstructed via `self._adjust_run_nqubits`, then for every
        `time` a ground-truth state is established (exact diagonalization
        if `n_qubits <= MAX_NAIVE_QUBITS`, self-convergence proxy if
        `accuracy_convergence_dims` is given, Trotterized evolution as a
        fallback if `"trotter"` is in `methods`, or no ground truth at all
        otherwise). Each requested method is then run at every
        `krylov_dim` (and, for `"randomized_krylov"`, every `s_ratio`),
        with peak memory measured via `tracemalloc` and relative error
        computed against the ground-truth state when available. Points
        where `krylov_dim` or the derived sketch size `s` would meet or
        exceed the Hilbert space dimension `N = 2**n_qubits` are skipped.

        Results are both accumulated in memory and appended incrementally
        to disk as they're produced, so a run that crashes partway through
        still leaves a usable partial result on disk.

        Parameters
        ----------
        n_qubits_list : list[int]
            System sizes to sweep over.
        krylov_dim_list : list[int]
            Krylov subspace dimensions to try at each system size.
        s_ratio_list : list[float]
            Ratios of sketch dimension to `krylov_dim` to try for the
            `"randomized_krylov"` method (`s = ceil(s_ratio * krylov_dim)`).
            Ignored if `"randomized_krylov"` is not in `methods`.
        time_list : list[float]
            Simulation times to sweep over.
        methods : tuple[str, ...], optional
            Which methods to run; any combination of `"krylov"` and
            `"randomized_krylov"`. Default is both.
        seed : int, optional
            Seed used for reproducibility of the run (e.g. randomized
            sketch generation). Default is 67.
        accuracy_convergence_dims : list[int] or None, optional
            If provided, and `n_qubits > MAX_NAIVE_QUBITS`, use a
            self-convergence check (see `find_converged_krylov_dim`) as
            the ground truth instead of exact diagonalization or Trotter.
            `ground_truth_kind = "self_convergence"` but does not yet
            compute an actual ground-truth state or `rel_error` for
            those points. Default is None.

        Returns
        -------
        pandas.DataFrame
            One row per (n_qubits, time, krylov_dim, method[, s_ratio])
            combination actually run, with columns: `n_qubits`,
            `krylov_dim`, `s_ratio`, `num_random_vectors`, `time`,
            `method`, `runtime_sec`, `memory_gb`, `rel_error`,
            `ground_truth`, `evolved_state_norm`. `rel_error` is `None`
            wherever no ground-truth state was available.
        run_dir : pathlib.Path
            Directory containing this run's persisted config, JSONL log,
            and incrementally-appended result rows.
        """
        # set up storing results
        run_config = {
            "n_qubits_list": list(n_qubits_list),
            "krylov_dim_list": list(krylov_dim_list),
            "s_ratio_list": list(s_ratio_list),
            "time_list": list(time_list),
            "methods": list(methods),
            "seed": seed,
        }

        run_dir = start_run(self.output_dir, run_config)

        configure_logging(json_path=run_dir / "run.log.jsonl", console_level="INFO")

        rows = []
        total_points = len(n_qubits_list) * len(krylov_dim_list) * len(time_list)
        log.info(
            "run_parameter_study: starting",
            extra={
                "context": {
                    "run_dir": str(run_dir),
                    "n_qubits_list": list(n_qubits_list),
                    "krylov_dim_list": list(krylov_dim_list),
                    "s_ratio_list": list(s_ratio_list),
                    "time_list": list(time_list),
                    "methods": list(methods),
                    "total_base_points": total_points,
                }
            },
        )

        for n_qubits in n_qubits_list:
            hamiltonian, initial_state, krylov_dim, number_rand_vectors = self._adjust_run_nqubits(n_qubits)
            N = 2**n_qubits

            for time in time_list:
                if n_qubits <= MAX_NAIVE_QUBITS:
                    _, exact_state = time_evolve_naive(hamiltonian, initial_state, time)
                    ground_truth_kind = "exact"
                    exact_norm = np.linalg.norm(exact_state)
                elif accuracy_convergence_dims is not None:
                    ground_truth_kind = "self_convergence"
                    conv = self.find_converged_krylov_dim(
                        time_evolve_krylov,
                        hamiltonian,
                        initial_state,
                        time,
                        accuracy_convergence_dims,
                        tol=1e-6,
                    )
                    exact_state = conv["states"][-1]  # best available proxy
                    exact_norm = np.linalg.norm(exact_state)

                elif "trotter" in methods:
                    num_steps, order = (
                        self.trotter_kwargs["num_steps"],
                        self.trotter_kwargs["order"],
                    )
                    exact_state = time_evolve_trotter(hamiltonian, initial_state, time, num_steps, order)
                    ground_truth_kind = "trotter"
                    exact_norm = np.linalg.norm(exact_state)

                else:
                    exact_state = None
                    ground_truth_kind = None

                for krylov_dim in krylov_dim_list:
                    if krylov_dim >= N:
                        log.warning(
                            "run_parameter_study: skipping krylov_dim >= N",
                            extra={
                                "context": {
                                    "n_qubits": n_qubits,
                                    "krylov_dim": krylov_dim,
                                    "N": N,
                                }
                            },
                        )
                        continue

                    if "krylov" in methods:
                        tracemalloc.start()
                        state, runtime = self._run_method(
                            "krylov",
                            hamiltonian,
                            initial_state,
                            time,
                            krylov_dim,
                            number_rand_vectors,
                        )
                        _, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        mem = peak / 1e9
                        rel_error = (
                            float(np.linalg.norm(state - exact_state) / exact_norm) if exact_state is not None else None
                        )
                        evolved_state_norm = float(np.linalg.norm(state))
                        row = {
                            "n_qubits": n_qubits,
                            "krylov_dim": krylov_dim,
                            "s_ratio": None,
                            "num_random_vectors": None,
                            "time": time,
                            "method": "krylov",
                            "runtime_sec": runtime,
                            "memory_gb": mem,
                            "rel_error": rel_error,
                            "ground_truth": ground_truth_kind,
                            "evolved_state_norm": evolved_state_norm,
                        }

                        rows.append(row)
                        append_result_row(run_dir, row)

                    if "randomized_krylov" in methods:
                        for s_ratio in s_ratio_list:
                            s = int(np.ceil(s_ratio * krylov_dim))
                            if s >= N:
                                continue
                            tracemalloc.start()
                            state, runtime = self._run_method(
                                "randomized_krylov",
                                hamiltonian,
                                initial_state,
                                time,
                                krylov_dim,
                                s,
                            )
                            _, peak = tracemalloc.get_traced_memory()
                            tracemalloc.stop()
                            mem = peak / 1e9
                            evolved_state_norm = float(np.linalg.norm(state))
                            rel_error = (
                                float(np.linalg.norm(state - exact_state) / exact_norm)
                                if exact_state is not None
                                else None
                            )
                            row = {
                                "n_qubits": n_qubits,
                                "krylov_dim": krylov_dim,
                                "s_ratio": s_ratio,
                                "time": time,
                                "method": "randomized_krylov",
                                "runtime_sec": runtime,
                                "memory_gb": mem,
                                "rel_error": rel_error,
                                "ground_truth": ground_truth_kind,
                                "evolved_state_norm": evolved_state_norm,
                            }

                            rows.append(row)
                            append_result_row(run_dir, row)

        log.info(
            "parameter_study: complete",
            extra={"context": {"n_rows": len(rows), "run_dir": str(run_dir)}},
        )
        return pd.DataFrame(rows), run_dir
