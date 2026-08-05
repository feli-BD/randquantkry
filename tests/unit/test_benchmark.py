import pytest
import numpy as np
from randquantkry.benchmark import Benchmark, ALL_METHODS
from randquantkry.hamiltonian import build_hamiltonian, build_initial_state

def make_small_benchmark(methods=None):
    H = build_hamiltonian(4, [("XX", 0, 1.0), ("Z", 1, 0.5)])
    psi0 = build_initial_state(4, 0)
    return Benchmark(H, psi0, time=1.0, krylov_dim=8, num_random_vectors=None,
                      methods=methods, output_dir="test_results")

# def test_benchmark_runs_without_crashing():
#     """Regression test for the psi0/kdim/nrv NameError bug."""
#     b = make_small_benchmark(methods=["krylov"])
#     results, run_dir = b.benchmark(n_qubits_list=[4], methods=["krylov"])
#     assert results["krylov"]["runtime_sec"][0] is not None

# def test_benchmark_records_measured_memory():
#     """Regression test for the missing 'memory_gb' dict key bug."""
#     b = make_small_benchmark(methods=["krylov"])
#     results, run_dir = b.benchmark(n_qubits_list=[4], methods=["krylov"])
#     assert "memory_gb" in results["krylov"]
#     assert results["krylov"]["memory_gb"][0] is not None


def test_benchmark_rejects_unknown_method():
    H = build_hamiltonian(4, [("Z", 0, 1.0)])
    psi0 = build_initial_state(4, 0)
    with pytest.raises(ValueError, match="Unknown method"):
        Benchmark(H, psi0, time=1.0, krylov_dim=8, num_random_vectors=None,
                   methods=["not_a_real_method"])

def test_benchmark_defaults_to_all_methods():
    H = build_hamiltonian(4, [("Z", 0, 1.0)])
    psi0 = build_initial_state(4, 0)
    b = Benchmark(H, psi0, time=1.0, krylov_dim=8, num_random_vectors=None)
    assert set(b.methods) == set(ALL_METHODS)

def test_resolve_krylov_dim_clamps_when_exceeding_N():
    b = make_small_benchmark()
    # n_qubits=2 -> N=4; request krylov_dim way above N
    resolved = b._resolve_krylov_dim(n_qubits=2)  # b.krylov_dim=8, N=4
    assert resolved == 3  # clamped to N - 1

def test_resolve_num_random_vectors_default_rule():
    b = make_small_benchmark()
    resolved = b._resolve_num_random_vectors(krylov_dim=5, n_qubits=10)
    assert resolved == 2 * 5 + 1  # default rule: 2*krylov_dim + 1

def test_resolve_num_random_vectors_clamps_when_exceeding_N():
    b = make_small_benchmark()
    resolved = b._resolve_num_random_vectors(krylov_dim=100, n_qubits=2)  # N=4
    assert resolved == 4  # clamped to N

def test_run_method_unknown_raises():
    b = make_small_benchmark()
    with pytest.raises(ValueError, match="Unknown method"):
        b._run_method("not_a_method", b.H, b.initial_state, 1.0, 8, None)

def test_run_method_naive_too_many_qubits_raises():
    H_big = build_hamiltonian(14, [("Z", 0, 1.0)])
    psi_big = build_initial_state(14, 0)
    b = make_small_benchmark()
    with pytest.raises(ValueError, match="requires n_qubits"):
        b._run_method("naive", H_big, psi_big, 1.0, 8, None)