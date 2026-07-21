##########################################################################
#
#             Benchmark Functions for Quantum Evolution
#
###########################################################################

# imports

from time import perf_counter
import numpy as np
import tracemalloc
import pandas as pd
 
from .evolution import (
    time_evolve_naive,
    time_evolve_krylov,
    time_evolve_randomized_krylov,
    time_evolve_randomized_krylov_gaussian,
    MAX_NAIVE_QUBITS,
)

from .hamiltonian import build_hamiltonian, reconstruct_hamiltonian, build_initial_state
from .trotter import time_evolve_trotter
from randquantkry.utils.logging_config import get_logger, configure_logging
from randquantkry.utils.reproducible_runs import start_run, append_result_row

 
log = get_logger(__name__)

ALL_METHODS = ("naive", "krylov", "randomized_krylov", "randomized_krylov_gaussian", "trotter")

class Benchmark:
    """
    Configurable benchmark suite comparing quantum time-evolution methods
    across a range of system sizes.
 
    Parameters
    ----------
    hamiltonian_factory : callable(n_qubits) -> SparsePauliOp
        Builds the Hamiltonian for a given system size. Using a factory
        (rather than a single fixed Hamiltonian) is what lets runtime/memory
        be swept over n_qubits.
    initial_state_factory : callable(n_qubits) -> ndarray
        Builds the initial state for a given system size.
    time : float
        Evolution time, held fixed across the n_qubits sweep (pass a
        different QuantumBenchmark instance, or extend later, to also sweep
        time -- this is the natural hook for the Lieb-Robinson parameter
        study to plug in later).
    krylov_dim : int or callable(n_qubits) -> int
        Krylov dimension. A callable lets this depend on n_qubits -- e.g.
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

    def __init__(self, hamiltonian, initial_state, time,
                 krylov_dim, num_random_vectors=None, methods=None,
                 trotter_kwargs=None, seed=None, output_dir="results"):
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
        self.output_dir = output_dir

    def _resolve_krylov_dim(self, n_qubits):
        N = 2 ** n_qubits
        kdim = self.krylov_dim(n_qubits) if callable(self.krylov_dim) else self.krylov_dim
        if kdim >= N:
            log.warning(
                "krylov_dim exceeds or equals N=2**n_qubits; clamping to N-1 "
                "(a Krylov subspace can never exceed the full Hilbert space dimension)",
                extra={"context": {"requested_krylov_dim": kdim, "n_qubits": n_qubits, "N": N}},
            )
            kdim = max(1, N - 1)
        return kdim
 
    def _resolve_num_random_vectors(self, krylov_dim, n_qubits):
        N = 2 ** n_qubits
        if self.num_random_vectors is None:
            nrv = 2 * krylov_dim + 1
        else:
            nrv = self.num_random_vectors(krylov_dim) if callable(self.num_random_vectors) else self.num_random_vectors
        if nrv > N:
            log.warning(
                "num_random_vectors exceeds N=2**n_qubits; clamping to N "
                "(cannot sketch down to a dimension larger than the original space)",
                extra={"context": {"requested_num_random_vectors": nrv, "n_qubits": n_qubits, "N": N}},
            )
            nrv = N
        return nrv

    def _reconstruct_hamiltonian(self, n_qubits):
        """Construct everything needed for one n_qubits point: Hamiltonian,
        initial state, and resolved krylov_dim/num_random_vectors."""
        hamiltonian = reconstruct_hamiltonian(n_qubits, self.H)
        psi0 = build_initial_state(n_qubits, flipped_qubit=0)
        kdim = self._resolve_krylov_dim(n_qubits)
        nrv = self._resolve_num_random_vectors(kdim, n_qubits)
        return hamiltonian, psi0, kdim, nrv


    def _run_method(self, method, hamiltonian, initial_state, time, kdim, nrv=None):
        """Run a single method, returning (state, runtime_sec). Raises
        whatever the underlying method raises (e.g. ValueError above
        MAX_QUBITS_DENSE) -- callers decide how to handle that."""
        t0 = perf_counter()
        if method == "naive":
            if hamiltonian.num_qubits <= MAX_NAIVE_QUBITS:
                unitary, state = time_evolve_naive(hamiltonian, initial_state, time)
            else:
                log.info(" 'naive' skipped to prevent memory overload due to too many qubits: ", extra={"context": {
                        "n_qubits": self.H.num_qubits }} )
        elif method == "krylov":
            state = time_evolve_krylov(hamiltonian, initial_state, time, kdim)
        elif method == "randomized_krylov":
            state = time_evolve_randomized_krylov(hamiltonian, initial_state, time, kdim, nrv, seed=self.seed)
        elif method == "randomized_krylov_gaussian":
            state = time_evolve_randomized_krylov_gaussian(hamiltonian, initial_state, time, kdim, nrv, seed=self.seed)
        elif method == "trotter":
            state = time_evolve_trotter(hamiltonian, initial_state, time, **self.trotter_kwargs)
        else:
            raise ValueError(f"Unknown method {method!r}")
        runtime = perf_counter() - t0
        return state, runtime
      
    # verification: internal consistency, no ground truth needed
    def verify_consistency(self,  methods=None):
        # checks per method: state_norm ≈ 1, no NaN/Inf, Tri Hermitian (where applicable),
        # randomized_krylov -> krylov as num_random_vectors grows large
        results = {}
        for method in methods or self.methods:
            kdim = self._resolve_krylov_dim(self.H.num_qubits)
            nrv = self._resolve_num_random_vectors(kdim, self.H.num_qubits)
            state, runtime = self._run_method(method, self.H, self.initial_state, self.time, kdim, nrv)
            has_nan = bool(np.any(np.isnan(state)) or np.any(np.isinf(state)))
            if not has_nan:
                norm = float(np.linalg.norm(state))
                if np.isclose(norm, 1.0, atol=1e-6):
                    results[method] = {"state_norm": norm, "has_nan": has_nan, "passed": True}
                else:
                    results[method] = {"state_norm": norm, "has_nan": has_nan, "passed": False}
 
                log.info("verify_consistency: method checked", extra={"context": {
                    "method": method, "n_qubits": self.H.num_qubits, **{k: v for k, v in results[method].items()},
                }})
            else: 
                float("nan")

                results[method] = {"state_norm": None, "has_nan": has_nan, "passed": False}

            log.info("verify_consistency: method checked", extra={"context": {
                "method": method, "n_qubits": self.H.num_qubits, **{k: v for k, v in results[method].items()},
            }})

        return results

    # Validation & Verification
    # validation: compare against ground truth, trotter and self-convergence
    def validate_accuracy(self, methods=None):
        # rel_error vs exact (≤14 qubits) or self-convergence proxy beyond that

        if self.H.num_qubits <= 10:
            unitary, exact_state = time_evolve_naive(self.H, self.initial_state, self.time)   # real ground truth
            ground_truth_kind = "exact"
            exact_norm = np.linalg.norm(exact_state)
            
        else:
            # TODO: figure out convergence_dim things for now compare to trotter
            # conv = find_converged_krylov_dim(time_evolve_krylov, H, psi0, self.time, convergence_dims, tol=tol)
            # exact_state = conv["states"][-1]  # best available proxy: largest converged run
            # exact_norm = np.linalg.norm(exact_state)
            # ground_truth_kind = "self_convergence"

            exact_state = time_evolve_trotter(self.H, self.initial_state, self.time, **self.trotter_kwargs)
            ground_truth_kind = "trotter"
            exact_norm = np.linalg.norm(exact_state)
        
        results = {}
        for method in methods or self.methods:
            if method == "naive" and ground_truth_kind == "exact":
                results[method] = {"rel_error": 0.0, "ground_truth": ground_truth_kind}
                continue
            try:
                kdim = self._resolve_krylov_dim(self.H.num_qubits)
                nrv = self._resolve_num_random_vectors(kdim, self.H.num_qubits)
                state, runtime = self._run_method(method, self.H, self.initial_state, self.time, kdim, nrv)
                rel_error = float(np.linalg.norm(state - exact_state) / exact_norm)
                results[method] = {"rel_error": rel_error, "ground_truth": ground_truth_kind}
            except Exception as e:
                results[method] = {"rel_error": None, "ground_truth": ground_truth_kind, "error": str(e)}

            log.info("validate_accuracy: method checked", extra={"context": {
                            "method": method, "n_qubits": self.H.num_qubits, **{k: v for k, v in results[method].items()},
                        }})
        return results

    # Benchmark Comparison Plots
    def benchmark_runtime(self, n_qubits_list, methods):
        methods = methods or self.methods
        log.info("benchmark_runtime: starting", extra={"context": {
            "n_qubits_list": list(n_qubits_list), "methods": methods,
        }})
        results = { method: {"n_qubits": [], "runtime_sec": []} for method in methods }

        for n_qubits in n_qubits_list:
            hamiltonian, psi0, kdim, nrv = self._reconstruct_hamiltonian(n_qubits)
            for method in methods:
                results[method]["n_qubits"].append(n_qubits)
                try:
                    exact_state, runtime = self._run_method(method, hamiltonian, psi0, self.time, kdim)
                    results[method]["runtime_sec"].append(runtime)
                    log.debug("benchmark_runtime: point", extra={"context": {
                        "method": method, "n_qubits": n_qubits, "runtime_sec": runtime,
                    }})
                except Exception as e:
                    log.warning("benchmark_runtime: method failed, recording None", extra={"context": {
                        "method": method, "n_qubits": n_qubits, "error": str(e),
                    }})
                    results[method]["runtime_sec"].append(None)   # TODO: don't crash the sweep
        log.info("benchmark_runtime: complete")
        return results

    def estimate_memory_gb(self, krylov_dim, n_hamiltonian_terms, num_random_vectors=None):
        N = 2 ** self.H.num_qubits
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

    def benchmark_memory(self, n_qubits_list, methods, empirical=False):
        methods = methods or self.methods
        results = { method: {"n_qubits": [], "memory_gb": []} for method in methods }

        if empirical:
            for m in methods:
                results[m]["memory_gb_measured"] = []

        # theoretical calcualtes memory usage
        for n_qubits in n_qubits_list:
            hamiltonian, psi0, kdim, nrv = self._reconstruct_hamiltonian(n_qubits)
            n_terms = len(hamiltonian)
 
            for method in methods:
                results[method]["n_qubits"].append(n_qubits)
 
                if method == "naive":
                    dim = 2 ** n_qubits
                    est_gb = (dim ** 2 * 16) / 1e9  # dense N x N complex128
                elif method in ("krylov",):
                    est = self.estimate_memory_gb(kdim, n_terms)
                    est_gb = est["total_gb_deterministic"]
                elif method in ("randomized_krylov", "randomized_krylov_gaussian"):
                    est = self.estimate_memory_gb(kdim, n_terms, nrv)
                    est_gb = est["total_gb_randomized"]
                elif method == "trotter":
                    # statevector + circuit -- dominated by the same O(N) statevector
                    est_gb = (2 ** n_qubits * 16) / 1e9
                else:
                    est_gb = None
 
                results[method]["memory_gb"].append(est_gb)

                # Emperical measures peak memory usage
                if empirical:
                    try:
                        tracemalloc.start()
                        self._run_method(method, hamiltonian, psi0, self.time, kdim, nrv)
                        _, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        results[method]["memory_gb_measured"].append(peak / 1e9)
                    except Exception as e:
                        log.warning("benchmark_memory: empirical measurement failed", extra={"context": {
                            "method": method, "n_qubits": n_qubits, "error": str(e),
                        }})
                        results[method]["memory_gb_measured"].append(None)
 
        log.info("benchmark_memory: complete")
        return results
        

    def benchmark(self, n_qubits_list, methods=None, empirical_memory=False):
        """
        Combined runtime + memory sweep. Runs each (n_qubits, method) point ONCE,
        recording wall-clock runtime and theoretical memory always, plus
        empirically-measured peak memory if empirical_memory=True.

        Returns a single results dict compatible with BOTH plot_runtime(results)
        and plot_memory(results) -- no need to run the sweep twice to get both plots.

        NOTE: when empirical_memory=True, tracemalloc's own tracing overhead adds
        a small, real bias to the measured runtime_sec for that run. If you need
        a clean runtime number AND empirical memory, prefer two separate calls
        (empirical_memory=False for timing, True for memory) rather than reading
        runtime_sec from an empirical=True run.
        """
        methods = methods or self.methods
        log.info("benchmark: starting", extra={"context": {
            "n_qubits_list": list(n_qubits_list), "methods": methods,
            "empirical_memory": empirical_memory,
        }})

        results = {method: {"n_qubits": [], "state": [], "runtime_sec": [], "memory_gb": []} for method in methods}
        if empirical_memory:
            for m in methods:
                results[m]["memory_gb_measured"] = []

        for n_qubits in n_qubits_list:
            hamiltonian, psi0, kdim, nrv = self._reconstruct_hamiltonian(n_qubits)
            n_terms = len(hamiltonian)

            for method in methods:
                results[method]["n_qubits"].append(n_qubits)

                # theoretical memory: cheap, no computation needed 
                if method == "naive":
                    est_gb = (2 ** n_qubits) ** 2 * 16 / 1e9
                elif method == "krylov":
                    est_gb = self.estimate_memory_gb(kdim, n_terms)["total_gb_deterministic"]
                elif method in ("randomized_krylov", "randomized_krylov_gaussian"):
                    est_gb = self.estimate_memory_gb(kdim, n_terms, nrv)["total_gb_randomized"]
                elif method == "trotter":
                    est_gb = (2 ** n_qubits * 16) / 1e9
                else:
                    est_gb = None
                results[method]["memory_gb"].append(est_gb)

                # single run: measures runtime, optionally wrapped in tracemalloc 
                try:
                    if empirical_memory:
                        tracemalloc.start()
                        state, runtime = self._run_method(method, hamiltonian, psi0, self.time, kdim, nrv)
                        _, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        results[method]["memory_gb_measured"].append(peak / 1e9)
                    else:
                        _, runtime = self._run_method(method, hamiltonian, psi0, self.time, kdim, nrv)

                    results[method]["runtime_sec"].append(runtime)
                    results[method]["state"].append(state)
                    log.debug("benchmark: point", extra={"context": {
                        "method": method, "n_qubits": n_qubits, "runtime_sec": runtime,
                    }})

                except Exception as e:
                    log.warning("benchmark: method failed, recording None", extra={"context": {
                        "method": method, "n_qubits": n_qubits, "error": str(e),
                    }})
                    results[method]["runtime_sec"].append(None)
                    if empirical_memory:
                        results[method]["memory_gb_measured"].append(None)
                        try:
                            tracemalloc.stop()  # in case it was left running after a mid-call failure
                        except Exception:
                            pass

        log.info("benchmark: complete")
        return results
    
    def parameter_study(self, n_qubits_list, krylov_dim_list, s_ratio_list, time_list,
                              methods=("krylov", "randomized_krylov"), seed=67,
                              accuracy_convergence_dims=None):
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
        log.info("run_parameter_study: starting", extra={"context": {"run_dir": str(run_dir),
            "n_qubits_list": list(n_qubits_list), "krylov_dim_list": list(krylov_dim_list),
            "s_ratio_list": list(s_ratio_list), "time_list": list(time_list),
            "methods": list(methods), "total_base_points": total_points,
        }})

        for n_qubits in n_qubits_list:
            hamiltonian, psi0, kdim, nrv = self._reconstruct_hamiltonian(n_qubits)
            n_terms = len(hamiltonian)
            N = 2 ** n_qubits

            for time in time_list:
                
                if n_qubits <= MAX_NAIVE_QUBITS:
                    _, exact_state = time_evolve_naive(hamiltonian, psi0, time)
                    ground_truth_kind = "exact"
                    exact_norm = np.linalg.norm(exact_state)
                elif accuracy_convergence_dims is not None:
                    # TODO implement
                    ground_truth_kind = "self_convergence"
                    # exact_norm = np.linalg.norm(exact_state)

                elif "trotter" in methods:
                    num_steps, order = self.trotter_kwargs['num_steps'], self.trotter_kwargs['order']
                    exact_state = time_evolve_trotter(hamiltonian, psi0, time, num_steps, order)
                    ground_truth_kind = "trotter"
                    exact_norm = np.linalg.norm(exact_state)

                else:
                    exact_state = None
                    ground_truth_kind = None

                for krylov_dim in krylov_dim_list:
                    if krylov_dim >= N:
                        log.warning("run_parameter_study: skipping krylov_dim >= N", extra={"context": {
                            "n_qubits": n_qubits, "krylov_dim": krylov_dim, "N": N,
                        }})
                        continue    

                    if "krylov" in methods:
                        tracemalloc.start()
                        state, runtime = self._run_method("krylov", hamiltonian, psi0, self.time, krylov_dim, nrv)
                        _, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()  
                        mem = (peak / 1e9)                          
                        rel_error = (float(np.linalg.norm(state - exact_state) / exact_norm) if exact_state is not None else None)
                        evolved_state_norm = float(np.linalg.norm(state))
                        row = {
                            "n_qubits": n_qubits, "krylov_dim": krylov_dim, "s_ratio": None,
                            "num_random_vectors": None, "time": time, "method": "krylov",
                            "runtime_sec": runtime, "memory_gb": mem, "rel_error": rel_error,
                            "evolved_state_norm": evolved_state_norm, "ground_truth": ground_truth_kind,
                            }
                        rows.append(row)
                        append_result_row(run_dir, row) 
                    
                    if "randomized_krylov" in methods:
                        for s_ratio in s_ratio_list:
                            s = int(np.ceil(s_ratio * krylov_dim))
                            if s >= N:
                                continue
                            tracemalloc.start()
                            state, runtime = self._run_method("randomized_krylov", hamiltonian, psi0, self.time, krylov_dim, s)
                            _, peak = tracemalloc.get_traced_memory()
                            tracemalloc.stop()
                            mem = peak / 1e9
                            rel_error = (float(np.linalg.norm(state - exact_state) / exact_norm) if exact_state is not None else None)
                            row = {
                                "n_qubits": n_qubits, "krylov_dim": krylov_dim, "s_ratio": s_ratio,
                                "num_random_vectors": s, "time": time, "method": "randomized_krylov",
                                "runtime_sec": runtime, "memory_gb": mem, "rel_error": rel_error,
                                "evolved_state_norm": evolved_state_norm,  "ground_truth": ground_truth_kind,
                            }
                            rows.append(row)
                            append_result_row(run_dir, row)

        log.info("parameter_study: complete", extra={"context": {"n_rows": len(rows), "run_dir": str(run_dir)}})
        return pd.DataFrame(rows), run_dir