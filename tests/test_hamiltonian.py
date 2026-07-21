import numpy as np
import pytest

from randquantkry.hamiltonian import build_hamiltonian


def test_build_hamiltonian_matches_intuitive_tensor_product():
    """Placing a Pauli at position k should yield P_0 (x) P_1 (x) ... when
    converted to a matrix -- this is the core correctness property we spent
    significant effort verifying (see: endianness discussion)."""
    I = np.eye(2)
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


# def test_build_hamiltonian_combines_duplicate_terms():
#     H = build_hamiltonian(2, [("X", 0, 1.0), ("X", 0, 0.5)])
#     assert len(H) == 1
#     assert np.isclose(H.coeffs[0], 1.5)


# @pytest.mark.parametrize("n_qubits", [0, -1])
# def test_build_hamiltonian_rejects_invalid_n_qubits(n_qubits):
#     with pytest.raises(ValueError):
#         build_hamiltonian(n_qubits, [("X", 0, 1.0)])


# def test_build_hamiltonian_rejects_empty_terms():
#     with pytest.raises(ValueError):
#         build_hamiltonian(2, [])


# def test_build_hamiltonian_rejects_out_of_range_qubit():
#     with pytest.raises(ValueError):
#         build_hamiltonian(2, [("XX", 1, 1.0)])  # occupies qubits 1,2 -- out of range


# def test_build_hamiltonian_rejects_invalid_pauli_letter():
#     with pytest.raises(ValueError):
#         build_hamiltonian(2, [("XA", 0, 1.0)])


# def test_build_hamiltonian_rejects_gate_too_long():
#     with pytest.raises(ValueError):
#         build_hamiltonian(4, [("XXX", 0, 1.0)])


# def test_build_hamiltonian_rejects_negative_qubit():
#     with pytest.raises(ValueError):
#         build_hamiltonian(2, [("X", -1, 1.0)])