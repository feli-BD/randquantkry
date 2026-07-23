# tests/test_verification.py
import numpy as np
import pytest
from randquantkry.hamiltonian import build_hamiltonian
from randquantkry.benchmark import Benchmark 

@pytest.fixture
def small_bench():
    H = build_hamiltonian(6, [('X', i, 1.0) for i in range(6)] + [('ZZ', i, 0.5) for i in range(5)])
    psi0 = np.zeros(2**6, dtype=complex); psi0[0] = 1.0
    return Benchmark(H, psi0, time=0.5, krylov_dim=20, num_random_vectors=41, seed=42)

def test_all_methods_pass_verification(small_bench):
    results = small_bench.verify_consistency()
    for method, r in results.items():
        assert r["passed"], f"{method} failed consistency check: {r}"

# def test_all_methods_meet_accuracy_threshold(small_bench):
#     results = small_bench.validate_accuracy()
#     for method, r in results.items():
#         if r["rel_error"] is not None:
#             assert r["rel_error"] < 1e-3, f"{method} rel_error too high: {r}"