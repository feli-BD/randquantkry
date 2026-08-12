#############################################################################
#
#           Trotter-Suzuki Time Evolution Function  
#
#############################################################################

# Imports
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from randquantkry.utils.logging_config import get_logger

log = get_logger(__name__)


def time_evolve_trotter(hamiltonian: SparsePauliOp, initial_state, time: float, num_steps: int, order=1):
    """
    Time evolution via Trotter-Suzuki product-formula approximation, using
    Qiskit's synthesis machinery.

    NOTE: this is a different approximation family from Krylov, converging to
    the true e^{-iHt} as num_steps -> inf. Its value as a comparison point is
    that it is literally the algorithm used to run time
    evolution on real quantum hardware so it answers "how does classical
    Krylov reach compare to what a QPU would actually do", not "what is
    ground truth".

    IMPORTANT: PauliEvolutionGate's matrix/Statevector shortcuts (Operator(gate),
    Statevector.evolve(gate)) silently return the EXACT e^{-iHt} matrix,
    ignoring the requested synthesis/num_steps entirely, the Trotter
    approximation is only actually applied once the circuit is decomposed
    into elementary gates. This function decomposes explicitly for that
    reason; skipping that step would silently defeat the whole point of
    calling this an approximation method (verified empirically: without
    decomposition, error is ~1e-15 regardless of num_steps).

    Parameters
    ----------
    hamiltonian : SparsePauliOp
        The Hamiltonian operator.
    initial_state : array-like
        The initial state of the quantum system (a normalized statevector).
    time : float
        The evolution time.
    num_steps : int
        Number of Trotter steps (repetitions). More steps -> smaller error,
        more circuit depth / more classical simulation cost.
    order : int, default 1
        Trotter-Suzuki order. 1 = first-order Lie-Trotter (cheapest, largest
        error per step). 2, 4, ... = higher-order Suzuki formulas (smaller
        error per step, more gates per step).

    Returns
    -------
    ndarray
        The evolved state vector.
    """
    log.info(
        "time_evolve_trotter: starting",
        extra={
            "context": {
                "n_qubits": hamiltonian.num_qubits,
                "time": time,
                "num_steps": num_steps,
                "order": order,
            }
        },
    )

    if order == 1:
        synthesis = LieTrotter(reps=num_steps)
    else:
        synthesis = SuzukiTrotter(order=order, reps=num_steps)

    evolution_gate = PauliEvolutionGate(hamiltonian, time=time, synthesis=synthesis)

    n_qubits = hamiltonian.num_qubits
    qc = QuantumCircuit(n_qubits)
    qc.append(evolution_gate, range(n_qubits))
    qc_decomposed = qc.decompose(reps=5)  # fully unroll -- see note above

    log.debug(
        "time_evolve_trotter: circuit decomposed",
        extra={"context": {"gate_count": len(qc_decomposed.data)}},
    )

    sv0 = Statevector(initial_state)
    evolved = sv0.evolve(qc_decomposed)
    evolved_state = evolved.data

    log.info(
        "time_evolve_trotter: complete",
        extra={"context": {"evolved_state_norm": float(np.linalg.norm(evolved_state))}},
    )

    return evolved_state
