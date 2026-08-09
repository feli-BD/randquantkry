import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp
from randquantkry.hamiltonian import build_hamiltonian, build_initial_state
from randquantkry.benchmark import Benchmark

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


@pytest.fixture
def large_hamiltonian():
    return build_hamiltonian(4, [("XX", 0, 1.0), ("XX", 1, 1.0), ("Z", 0, 0.5)])

@pytest.fixture
def large_initial_state():
    return build_initial_state(4, flipped_qubit=0)

@pytest.fixture
def small_benchmark(large_hamiltonian, large_initial_state, tmp_path):
    """A Benchmark instance writing to a pytest-managed temp directory,
    so nothing pollutes your real results/ folder and every test starts clean."""
    return Benchmark(
        hamiltonian=large_hamiltonian,
        initial_state=large_initial_state,
        time=1.0,
        krylov_dim=8,
        num_random_vectors=None,
        methods=["naive", "krylov", "randomized_krylov"],
        seed=42,
        output_dir=str(tmp_path / "results"),
    )