##########################################################################
#
#             Benchmark Functions for Quantum Evolution
#
###########################################################################

from time import perf_counter
import numpy as np

from .evolution import (
    time_evolve_naive,
    time_evolve_krylov,
    time_evolve_randomized_krylov,
    time_evolve_randomized_krylov_gaussian,
    time_evolve_randomized_krylov_gramschmidt,
)
from .trotter import time_evolve_trotter
from .utils.logging_config import get_logger

log = get_logger(__name__)

NAIVE_MAX_QUBITS = 10

ALL_METHODS = ("naive", "krylov", "randomized_krylov", "randomized_krylov_gaussian", "randomized_krylov_gramschmidt", "trotter")
 
def experiment_benchmark_V2(hamiltonian, initial_state, time, krylov_dim, num_random_vectors,
                             trotter_num_steps=50, trotter_order=1, seed=None, methods=None):
    """
    Benchmark selected quantum evolution methods against exact diagonalization.
 
    Parameters
    ----------
    hamiltonian : SparsePauliOp
    initial_state : array-like
    time : float
    krylov_dim : int
        Krylov dimension for both krylov and randomized_krylov(_gaussian).
    num_random_vectors : int
        Sketch dimension for both randomized methods. Recommended:
        2*krylov_dim + 1 (see time_evolve_randomized_krylov's docstring for why).
    trotter_num_steps : int, default 50
        Number of Trotter steps for the trotter comparison method.
    trotter_order : int, default 1
        Trotter-Suzuki order (1 = first-order Lie-Trotter).
    seed : int, optional
        Seed passed through to both randomized methods for reproducibility.
    methods : list of str, optional
        Which methods to actually run, from {"naive", "krylov",
        "randomized_krylov", "randomized_krylov_gaussian", "trotter"}.
        Defaults to all five. Only requested methods are computed and
        returned -- e.g. methods=["krylov", "randomized_krylov"] skips
        exact diagonalization, the Gaussian variant, and Trotter entirely.
 
        NOTE: "rel_error" for non-naive methods is only populated if "naive"
        is ALSO in `methods` (and hamiltonian.num_qubits is within
        NAIVE_MAX_QUBITS) -- there's no ground truth to compare against
        otherwise. Include "naive" in `methods` whenever you want accuracy
        numbers, not just runtime/state_norm.
 
    Returns
    -------
    dict
        One key per requested method (not all five, unless requested). Each
        maps to a dict with "state", "runtime_sec", "rel_error", "state_norm".
    """
    if methods is None:
        methods = list(ALL_METHODS)
    unknown = set(methods) - set(ALL_METHODS)
    if unknown:
        raise ValueError(f"Unknown method(s) {sorted(unknown)}; valid options are {ALL_METHODS}")
 
    log.info(
        "benchmark: starting",
        extra={"context": {
            "n_qubits": hamiltonian.num_qubits,
            "time": time,
            "krylov_dim": krylov_dim,
            "num_random_vectors": num_random_vectors,
            "trotter_num_steps": trotter_num_steps,
            "trotter_order": trotter_order,
            "seed": seed,
            "methods": methods,
        }},
    )
 
    results = {}
    exact_state = None
    exact_norm = None
 
    # --- ground truth: exact diagonalization (only if requested) ---
    if "naive" in methods:
        if hamiltonian.num_qubits > NAIVE_MAX_QUBITS:
            log.warning(
                "benchmark: naive skipped due to large qubit count",
                extra={"context": {"n_qubits": hamiltonian.num_qubits, "max_qubits_dense": NAIVE_MAX_QUBITS}},
            )
            results["naive"] = {"state": None, "runtime_sec": None, "rel_error": None, "state_norm": None}
        else:
            t0 = perf_counter()
            _, exact_state = time_evolve_naive(hamiltonian, initial_state, time)
            runtime = perf_counter() - t0
            exact_norm = np.linalg.norm(exact_state)
            results["naive"] = {
                "state": exact_state, "runtime_sec": runtime, "rel_error": 0.0,
                "state_norm": float(exact_norm),
            }
            log.info("benchmark: naive complete", extra={"context": {"runtime_sec": runtime}})
 
    def _rel_error(state):
        return float(np.linalg.norm(state - exact_state) / exact_norm) if exact_state is not None else None
 
    # --- deterministic Krylov ---
    if "krylov" in methods:
        t0 = perf_counter()
        krylov_state = time_evolve_krylov(hamiltonian, initial_state, time, krylov_dim)
        runtime = perf_counter() - t0
        results["krylov"] = {
            "state": krylov_state, "runtime_sec": runtime, "rel_error": _rel_error(krylov_state),
            "state_norm": float(np.linalg.norm(krylov_state)),
        }
        log.info("benchmark: krylov complete", extra={"context": {"runtime_sec": runtime}})
 
    # --- randomized Krylov (fast SRTT sketch) ---
    if "randomized_krylov" in methods:
        t0 = perf_counter()
        rand_state = time_evolve_randomized_krylov(
            hamiltonian, initial_state, time, krylov_dim, num_random_vectors, seed=seed
        )
        runtime = perf_counter() - t0
        results["randomized_krylov"] = {
            "state": rand_state, "runtime_sec": runtime, "rel_error": _rel_error(rand_state),
            "state_norm": float(np.linalg.norm(rand_state)),
        }
        log.info("benchmark: randomized_krylov complete", extra={"context": {"runtime_sec": runtime}})
 
    # --- randomized Krylov (dense Gaussian sketch, comparison baseline) ---
    if "randomized_krylov_gaussian" in methods:
        t0 = perf_counter()
        gauss_state = time_evolve_randomized_krylov_gaussian(
            hamiltonian, initial_state, time, krylov_dim, num_random_vectors, seed=seed
        )
        runtime = perf_counter() - t0
        results["randomized_krylov_gaussian"] = {
            "state": gauss_state, "runtime_sec": runtime, "rel_error": _rel_error(gauss_state),
            "state_norm": float(np.linalg.norm(gauss_state)),
        }
        log.info("benchmark: randomized_krylov_gaussian complete", extra={"context": {"runtime_sec": runtime}})

    # --- randomized Krylov (dense Gaussian sketch with Gram Schmidt ) ---
    if "randomized_krylov_gramschmidt" in methods:
        t0 = perf_counter()
        gramschmidt_state = time_evolve_randomized_krylov_gramschmidt(
            hamiltonian, initial_state, time, krylov_dim, num_random_vectors, seed=seed
        )
        runtime = perf_counter() - t0
        results["randomized_krylov_gramschmidt"] = {
            "state": gramschmidt_state, "runtime_sec": runtime, "rel_error": _rel_error(gramschmidt_state),
            "state_norm": float(np.linalg.norm(gramschmidt_state)),
        }
        log.info("benchmark: randomized_krylov_gramschmidt complete", extra={"context": {"runtime_sec": runtime}})
 
 
    # --- Trotter (product-formula, NOT exact -- the "what a QPU would do" comparison) ---
    if "trotter" in methods:
        t0 = perf_counter()
        trotter_state = time_evolve_trotter(
            hamiltonian, initial_state, time, num_steps=trotter_num_steps, order=trotter_order
        )
        runtime = perf_counter() - t0
        results["trotter"] = {
            "state": trotter_state, "runtime_sec": runtime, "rel_error": _rel_error(trotter_state),
            "state_norm": float(np.linalg.norm(trotter_state)),
        }
        log.info("benchmark: trotter complete", extra={"context": {"runtime_sec": runtime}})
 
    log.info("benchmark: complete")
    return results
 


def sweep_num_random_vectors(hamiltonian, initial_state, time, krylov_dim, sketch_dims,
                              use_gaussian=False, seed=None):
    """
    Run the randomized Krylov method across a range of sketch dimensions,
    holding krylov_dim fixed, to see how accuracy and state-norm depend on
    the sketch size.

    Parameters
    ----------
    sketch_dims : list of int
        Values of num_random_vectors to sweep over.
    use_gaussian : bool, default False
        If True, sweep time_evolve_randomized_krylov_gaussian (dense sketch)
        instead of the default fast-sketch version -- useful for reproducing
        the "dense Gaussian doesn't improve with s" comparison.
    seed : int, optional
        If given, reseeds before EACH run in the sweep, so differences you
        observe come from sketch_dim alone.

    Returns
    -------
    dict with "sketch_dims", "rel_error", "state_norm", "runtime_sec" lists,
    aligned with sketch_dims, ready to plot.
    """
    log.info(
        "sweep_num_random_vectors: starting",
        extra={"context": {"krylov_dim": krylov_dim, "sketch_dims": list(sketch_dims),
                            "use_gaussian": use_gaussian, "seed": seed}},
    )

    _, exact_state = time_evolve_naive(hamiltonian, initial_state, time)
    exact_norm = np.linalg.norm(exact_state)

    evolve_fn = time_evolve_randomized_krylov_gaussian if use_gaussian else time_evolve_randomized_krylov

    rel_errors, state_norms, runtimes = [], [], []

    for s in sketch_dims:
        t0 = perf_counter()
        # FIX: `s` (num_random_vectors) was previously missing entirely from
        # this call -- `seed` was being passed positionally into the
        # num_random_vectors slot, so the sweep never actually varied the
        # sketch dimension at all.
        state = evolve_fn(hamiltonian, initial_state, time, krylov_dim, num_random_vectors=s, seed=seed)
        runtime = perf_counter() - t0

        rel_err = np.linalg.norm(state - exact_state) / exact_norm
        rel_errors.append(float(rel_err))
        state_norms.append(float(np.linalg.norm(state)))
        runtimes.append(runtime)

        log.debug(
            "sweep_num_random_vectors: point",
            extra={"context": {"s": s, "rel_error": float(rel_err), "state_norm": float(np.linalg.norm(state))}},
        )

    log.info("sweep_num_random_vectors: complete")

    return {
        "sketch_dims": list(sketch_dims),
        "rel_error": rel_errors,
        "state_norm": state_norms,
        "runtime_sec": runtimes,
    }