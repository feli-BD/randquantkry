import numpy as np
import pytest
from scipy.linalg import expm

from randquantkry.hamiltonian import build_hamiltonian, build_initial_state
from randquantkry.evolution import (
    time_evolve_naive,
    time_evolve_krylov,
    time_evolve_randomized_krylov
)

# Test for naive:

def test_time_evolve_naive_matches_scipy_expm(small_hamiltonian, small_initial_state):
    _, out = time_evolve_naive(small_hamiltonian, small_initial_state, time=0.5)
    ref = expm(-1j * small_hamiltonian.to_matrix() * 0.5) @ small_initial_state
    assert np.allclose(out, ref)


def test_naive_raises_when_memory_insufficient():
    # Use an absurdly large qubit count to guarantee it exceeds any real machine's RAM
    H_huge = build_hamiltonian(14, [("Z", 0, 1.0)])
    psi_huge = build_initial_state(14, 0)
    with pytest.raises(ValueError, match="exceeds available system memory"):
        time_evolve_naive(H_huge, psi_huge, time=1.0)


def test_naive_preserves_norm():
    H = build_hamiltonian(4, [("XX", 0, 1.0), ("Z", 1, 0.5)])
    psi0 = build_initial_state(4, 0)
    _, state = time_evolve_naive(H, psi0, time=1.0)
    assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-10)

# Test for krylov 

def test_time_evolve_krylov_matches_exact(small_hamiltonian, small_initial_state):
    _, exact = time_evolve_naive(small_hamiltonian, small_initial_state, time=0.5)
    out = time_evolve_krylov(small_hamiltonian, small_initial_state, time=0.5, krylov_dim=20)
    # 20 Krylov vectors should fully converge for a 6-qubit (64-dim) system
    assert np.linalg.norm(out - exact) < 1e-8


def test_time_evolve_krylov_preserves_norm(small_hamiltonian, small_initial_state):
    out = time_evolve_krylov(small_hamiltonian, small_initial_state, time=0.5, krylov_dim=15)
    assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-8)

def test_hamiltonian_state_dimension_mismatch_raises():
    H4 = build_hamiltonian(4, [("Z", 0, 1.0)])
    psi_wrong = build_initial_state(6, 0)  # wrong dimension
    with pytest.raises((ValueError, IndexError)):
        time_evolve_krylov(H4, psi_wrong, time=1.0, krylov_dim=4)

def test_krylov_low_dim_worse_than_high_dim():
    H = build_hamiltonian(4, [("XX", 0, 1.0), ("Z", 1, 0.5)])
    psi0 = build_initial_state(4, 0)
    _, exact = time_evolve_naive(H, psi0, time=1.0)
    approx_low = time_evolve_krylov(H, psi0, time=1.0, krylov_dim=2)
    approx_high = time_evolve_krylov(H, psi0, time=1.0, krylov_dim=10)
    err_low = np.linalg.norm(approx_low - exact) / np.linalg.norm(exact)
    err_high = np.linalg.norm(approx_high - exact) / np.linalg.norm(exact)
    assert err_high <= err_low, "Higher Krylov dim should not be less accurate"

def test_time_evolve_krylov_handles_early_breakdown(localized_hamiltonian):
    """A Hamiltonian acting on only 3 of 8 qubits has a tiny invariant
    subspace -- this should break down early and NOT crash regardless of
    the requested krylov_dim."""
    psi0 = np.zeros(2 ** localized_hamiltonian.num_qubits, dtype=complex)
    psi0[0] = 1.0
    out = time_evolve_krylov(localized_hamiltonian, psi0, time=1.0, krylov_dim=50)
    assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-8)

# Test for randomized Krylov

def test_time_evolve_randomized_krylov_matches_exact(small_hamiltonian, small_initial_state):
    _, exact = time_evolve_naive(small_hamiltonian, small_initial_state, time=0.5)
    out = time_evolve_randomized_krylov(
        small_hamiltonian, small_initial_state, time=0.5,
        krylov_dim=15, num_random_vectors=31, trunc=2, seed=42,
    )
    # sketched method has a small but bounded residual error, not machine precision
    assert np.linalg.norm(out - exact) < 1e-4

def test_time_evolve_randomized_krylov_preserves_norm(small_hamiltonian, small_initial_state):
    """This is the single most important regression test in this suite:
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


def test_randomized_krylov_deterministic_given_seed(small_hamiltonian, small_initial_state):
    out1 = time_evolve_randomized_krylov(
        small_hamiltonian, small_initial_state, time=0.5,
        krylov_dim=10, num_random_vectors=25, seed=123,
    )
    out2 = time_evolve_randomized_krylov(
        small_hamiltonian, small_initial_state, time=0.5,
        krylov_dim=10, num_random_vectors=25, seed=123,
    )
    assert np.allclose(out1, out2)


def test_randomized_krylov_different_seeds_give_different_results(small_hamiltonian, small_initial_state):
    out1 = time_evolve_randomized_krylov(
        small_hamiltonian, small_initial_state, time=0.5,
        krylov_dim=10, num_random_vectors=25, seed=1,
    )
    out2 = time_evolve_randomized_krylov(
        small_hamiltonian, small_initial_state, time=0.5,
        krylov_dim=10, num_random_vectors=25, seed=2,
    )
    assert not np.allclose(out1, out2)
