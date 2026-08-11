#############################################################################
#
#           Tests for the functionalities in hamiltonian.py:
#       build_hamiltonian, reconstruct_hamiltonian, build_initial_state
#                   Checksum Test for hamiltonian
#
#############################################################################

import numpy as np
import pytest
from randquantkry.hamiltonian import (
    build_hamiltonian,
    build_initial_state,
    reconstruct_hamiltonian,
)
from randquantkry.utils.reproducible_runs import hash_terms

# Tests for build hamiltonian


def test_build_hamiltonian_matches_intuitive_tensor_product():
    """Placing a Pauli at position k should yield P_0 (x) P_1 (x) ... when
    converted to a matrix. This is the core correctness property we spent
    significant effort verifying (see: endianness discussion)."""
    I = np.eye(2)  # noqa: E741 —> standard notation alongside Pauli matrices X, Z
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])

    def kron_all(ops):
        out = ops[0]
        for o in ops[1:]:
            out = np.kron(out, o)
        return out

    ref = 1.0 * kron_all([X, X, I, I]) + 0.5 * kron_all([I, Z, Z, I])
    H = build_hamiltonian(4, [("XX", 0, 1.0), ("ZZ", 1, 0.5)])
    assert np.allclose(np.asarray(H.to_matrix()), ref)


@pytest.mark.parametrize("n_qubits", [0, -1])
def test_build_hamiltonian_rejects_invalid_n_qubits(n_qubits):
    with pytest.raises(ValueError):
        build_hamiltonian(n_qubits, [("X", 0, 1.0)])


def test_build_hamiltonian_rejects_empty_terms():
    with pytest.raises(ValueError):
        build_hamiltonian(2, [])


def test_build_hamiltonian_rejects_invalid_pauli_letter():
    with pytest.raises(ValueError):
        build_hamiltonian(2, [("XA", 0, 1.0)])


def test_build_hamiltonian_rejects_negative_qubit():
    with pytest.raises(ValueError):
        build_hamiltonian(2, [("X", -1, 1.0)])


def test_build_hamiltonian_correct_dim():
    H = build_hamiltonian(4, [("X", 0, 1.0)])
    assert H.num_qubits == 4


def test_build_hamiltonian_hermitian():
    H = build_hamiltonian(4, [("XX", 0, 1.0), ("Z", 2, 0.5)])
    matrix = H.to_matrix()
    assert np.allclose(matrix, matrix.conj().T), "Hamiltonian must be Hermitian"


def test_build_hamiltonian_out_of_range_raises():
    with pytest.raises(ValueError):
        build_hamiltonian(3, [("XX", 2, 1.0)])  # qubit+1=3 out of range for n=3


# Tests for build_initial_state


def test_build_initial_state_unit_norm():
    psi0 = build_initial_state(4, 0)
    assert np.isclose(np.linalg.norm(psi0), 1.0)


# Tests for reconstruct_hamiltonian


def test_reconstruct_hamiltonian_faithful_roundtrip():
    terms = [
        ("XX", 0, 1.0),
        ("Z", 0, 0.5),
    ]  # Z only at qubit 0, deliberately non-uniform
    H = build_hamiltonian(3, terms)
    H_reconstructed = reconstruct_hamiltonian(3, H, uniform=False)
    assert np.allclose(H.to_matrix(), H_reconstructed.to_matrix(), atol=1e-10)


def test_reconstruct_hamiltonian_uniform_spreads_to_all_positions():
    terms = [("Z", 0, 0.5)]  # Z only at qubit 0
    H = build_hamiltonian(3, terms)
    H_reconstructed = reconstruct_hamiltonian(3, H, uniform=True)  # default
    expected_terms = [("Z", 0, 0.5), ("Z", 1, 0.5), ("Z", 2, 0.5)]
    H_expected = build_hamiltonian(3, expected_terms)
    assert np.allclose(H_reconstructed.to_matrix(), H_expected.to_matrix(), atol=1e-10)


# Checksum Tests


def test_hash_terms_deterministic():
    terms = [("XX", 0, 1.0), ("Z", 1, 0.5)]
    assert hash_terms(terms) == hash_terms(terms)


def test_hash_terms_detects_change():
    terms_a = [("XX", 0, 1.0)]
    terms_b = [("XX", 0, 1.0001)]  # tiny coefficient change
    assert hash_terms(terms_a) != hash_terms(terms_b)
