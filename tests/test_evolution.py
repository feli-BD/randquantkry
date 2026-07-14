import numpy as np
import pytest
from scipy.linalg import expm

from rand_quantum_sim.evolution import (
    time_evolve_naive,
    time_evolve_krylov,
    time_evolve_randomized_krylov
)


def test_time_evolve_naive_matches_scipy_expm(small_hamiltonian, small_initial_state):
    _, out = time_evolve_naive(small_hamiltonian, small_initial_state, time=0.5)
    ref = expm(-1j * small_hamiltonian.to_matrix() * 0.5) @ small_initial_state
    assert np.allclose(out, ref)


# def test_time_evolve_naive_rejects_too_many_qubits():
#     import qiskit.quantum_info as qi
#     n = 11
#     labels = ["X" + "I" * (n - 1)]
#     H = qi.SparsePauliOp(labels, coeffs=[1.0])
#     psi0 = np.zeros(2 ** n, dtype=complex)
#     psi0[0] = 1.0
#     with pytest.raises(ValueError):
#         time_evolve_naive(H, psi0, time=0.1)


def test_time_evolve_krylov_matches_exact(small_hamiltonian, small_initial_state):
    _, exact = time_evolve_naive(small_hamiltonian, small_initial_state, time=0.5)
    out = time_evolve_krylov(small_hamiltonian, small_initial_state, time=0.5, krylov_dim=20)
    # 20 Krylov vectors should fully converge for a 6-qubit (64-dim) system
    assert np.linalg.norm(out - exact) < 1e-8


def test_time_evolve_krylov_preserves_norm(small_hamiltonian, small_initial_state):
    out = time_evolve_krylov(small_hamiltonian, small_initial_state, time=0.5, krylov_dim=15)
    assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-8)


def test_time_evolve_krylov_handles_early_breakdown(localized_hamiltonian):
    """A Hamiltonian acting on only 3 of 8 qubits has a tiny invariant
    subspace -- this should break down early and NOT crash regardless of
    the requested krylov_dim."""
    psi0 = np.zeros(2 ** localized_hamiltonian.num_qubits, dtype=complex)
    psi0[0] = 1.0
    out = time_evolve_krylov(localized_hamiltonian, psi0, time=1.0, krylov_dim=50)
    assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-8)


def test_time_evolve_randomized_krylov_matches_exact(small_hamiltonian, small_initial_state):
    _, exact = time_evolve_naive(small_hamiltonian, small_initial_state, time=0.5)
    out = time_evolve_randomized_krylov(
        small_hamiltonian, small_initial_state, time=0.5,
        krylov_dim=15, num_random_vectors=31, trunc=2, seed=42,
    )
    # sketched method has a small but bounded residual error, not machine precision
    assert np.linalg.norm(out - exact) < 1e-4


def test_time_evolve_randomized_krylov_preserves_norm(small_hamiltonian, small_initial_state):
    """This is the single most important regression test in this suite --
    it directly catches the normalization-drift instability found during
    development, where naive sketch-based normalization caused the state
    norm to drift far from 1.0 without raising any error."""
    out = time_evolve_randomized_krylov(
        small_hamiltonian, small_initial_state, time=0.5,
        krylov_dim=15, num_random_vectors=31, trunc=2, seed=42,
    )
    assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-6)


@pytest.mark.parametrize("sketch_dim", [16, 20, 31, 50])
def test_time_evolve_randomized_krylov_stable_across_sketch_sizes(
    small_hamiltonian, small_initial_state, sketch_dim
):
    """Regression test for the chaotic-error bug: error should be small and
    roughly stable across sketch dimensions, not wildly varying."""
    _, exact = time_evolve_naive(small_hamiltonian, small_initial_state, time=0.5)
    out = time_evolve_randomized_krylov(
        small_hamiltonian, small_initial_state, time=0.5,
        krylov_dim=15, num_random_vectors=sketch_dim, trunc=2, seed=42,
    )
    rel_error = np.linalg.norm(out - exact) / np.linalg.norm(exact)
    assert rel_error < 1e-3


# def test_randomized_krylov_deterministic_given_seed(small_hamiltonian, small_initial_state):
#     out1 = time_evolve_randomized_krylov(
#         small_hamiltonian, small_initial_state, time=0.5,
#         krylov_dim=10, num_random_vectors=25, seed=123,
#     )
#     out2 = time_evolve_randomized_krylov(
#         small_hamiltonian, small_initial_state, time=0.5,
#         krylov_dim=10, num_random_vectors=25, seed=123,
#     )
#     assert np.allclose(out1, out2)


# def test_randomized_krylov_different_seeds_give_different_results(small_hamiltonian, small_initial_state):
#     out1 = time_evolve_randomized_krylov(
#         small_hamiltonian, small_initial_state, time=0.5,
#         krylov_dim=10, num_random_vectors=25, seed=1,
#     )
#     out2 = time_evolve_randomized_krylov(
#         small_hamiltonian, small_initial_state, time=0.5,
#         krylov_dim=10, num_random_vectors=25, seed=2,
#     )
#     assert not np.allclose(out1, out2)