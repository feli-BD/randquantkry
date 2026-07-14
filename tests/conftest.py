import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp


@pytest.fixture
def small_hamiltonian():
    """A 6-qubit Hamiltonian that mixes the FULL Hilbert space (X-field +
    ZZ-chain), small enough for exact diagonalization to be used as ground
    truth in tests."""
    n_qubits = 6
    rng = np.random.default_rng(1)
    labels, coeffs = [], []
    for i in range(n_qubits):
        s = ["I"] * n_qubits
        s[i] = "X"
        labels.append("".join(s))
        coeffs.append(float(rng.uniform(0.5, 1.5)))
    for i in range(n_qubits - 1):
        s = ["I"] * n_qubits
        s[i] = "Z"
        s[i + 1] = "Z"
        labels.append("".join(s))
        coeffs.append(float(rng.uniform(0.3, 0.8)))
    return SparsePauliOp(labels, coeffs=coeffs)


@pytest.fixture
def small_initial_state(small_hamiltonian):
    n = small_hamiltonian.num_qubits
    psi0 = np.zeros(2 ** n, dtype=complex)
    psi0[0] = 1.0
    return psi0


@pytest.fixture
def localized_hamiltonian():
    """A Hamiltonian that only acts on qubits 0-2 of a larger register --
    used to test the early-breakdown / invariant-subspace behavior."""
    n_qubits = 8
    labels = ["XXIIIIII", "IZZIIIII"]
    coeffs = [1.0, 0.5]
    return SparsePauliOp(labels, coeffs=coeffs)