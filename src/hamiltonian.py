##########################################################################
#
#                       Build Hamiltonian Function
#
###########################################################################

# imports
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from randquantkry.utils.logging_config import get_logger

log = get_logger(__name__)


def build_hamiltonian(n_qubits: int, terms: list[tuple[str, int, float]]) -> SparsePauliOp:
    """
    Build the Hamiltonian function from the given number of qubits and gates. Currently implemented for 1D Hamiltonians.

    Parameters
    ----------
    n_qubits (int): The number of qubits for the Hamiltonian.

    terms (list): List of (gate, start_qubit, coeff) tuples.
                'gate' (str) defines the Pauli string (e.g., "XX").
                'start_qubit' (int) is the index where the interaction begins.
                'coeff' (float) is the term's scalar weight.
    Returns
    -------
    SparsePauliOp
        A SparsePauliOp object that represents the Hamiltonian function, which can be called with the state variables to
        compute the Hamiltonian matrix.

    """

    labels = []
    coefficients = []

    for gate, qubit, coeff in terms:
        span = len(gate)
        if qubit < 0 or qubit + span - 1 >= n_qubits:
            raise ValueError(f"term {gate!r} at qubit {qubit} out of range for n_qubits={n_qubits}")

        s = ["I"] * n_qubits

        s[(qubit)] = gate[0]
        if len(gate) == 2:
            s[(qubit + 1)] = gate[1]

        labels.append("".join(s))
        coefficients.append(coeff)

    hamiltonian = SparsePauliOp(labels, coeffs=np.array(coefficients, dtype=complex))
    log.info(
        "build_hamiltonian: built",
        extra={
            "context": {
                "n_qubits": n_qubits,
                "matrix_dim": 2**n_qubits,
                "n_terms": len(terms),
            }
        },
    )
    return hamiltonian


def build_initial_state(n_qubits=int, flipped_qubit=int):
    """
    Build the initial state for a certain number of qubits.
    Warning: Currently only supports flipping one state.
    """
    initial_state = np.zeros(2**n_qubits, dtype=complex)
    initial_state[flipped_qubit] = 1.0  # |0000...0>)
    return initial_state


def reconstruct_hamiltonian(n_qubits: int, hamiltonian: SparsePauliOp, uniform: bool = True):
    """
    Parameters
    ----------
    n_qubits (int): The number of qubits for the Hamiltonian.
    hamiltonian (SparsePauliOp): Hamiltonian object constructed using build_hamiltonian function.
    uniform : bool, default True
        If True (original behavior), average each gate type's coefficient
        across all instances found, and reapply that average uniformly at
        every valid position -- use this to extrapolate a translationally-
        invariant pattern to a different n_qubits.
        If False, rebuild the Hamiltonian exactly as extracted, preserving
        each term's original position and coefficient -- use this for a
        faithful roundtrip (n_qubits must match the original).

    Returns
    -------
    SparsePauliOp
    """
    terms = []
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs, strict=True):
        label = str(pauli)
        nonzero_positions = [i for i, ch in enumerate(label) if ch != "I"]

        if not nonzero_positions:
            continue

        start_qubit = nonzero_positions[0]
        end_qubit = nonzero_positions[-1]

        if end_qubit - start_qubit + 1 != len(nonzero_positions):
            raise ValueError(
                f"term {label!r} has non-contiguous non-identity positions "
                f"{nonzero_positions}; extract_terms only supports contiguous "
                f"1- or 2-qubit terms (matching build_hamiltonian's own limits)."
            )
        if len(nonzero_positions) > 2:
            raise ValueError(
                f"term {label!r} spans {len(nonzero_positions)} qubits; "
                f"extract_terms only supports 1- or 2-qubit terms."
            )

        gate = "".join(label[i] for i in nonzero_positions)
        terms.append((gate, start_qubit, complex(coeff)))

    if not uniform:
        # Faithful roundtrip: rebuild exactly what was extracted, no averaging.
        log.info(
            "reconstruct_hamiltonian: faithful roundtrip",
            extra={"context": {"n_qubits": n_qubits, "n_terms": len(terms)}},
        )
        return build_hamiltonian(n_qubits, terms)

    # Original behavior: average per gate type, spread uniformly across all positions.
    coeffs_by_gate_type = {}
    for gate, _start_qubit, coeff in terms:
        coeffs_by_gate_type.setdefault(gate, []).append(coeff)

    avg_coeff_by_gate_type = {gate: np.mean(coeff_list) for gate, coeff_list in coeffs_by_gate_type.items()}

    new_terms = []
    for gate, avg_coeff in avg_coeff_by_gate_type.items():
        span = len(gate)
        if span == 1:
            positions = range(n_qubits)
        elif span == 2:
            positions = range(n_qubits - 1)
        else:
            raise ValueError(f"gate {gate!r} has unsupported span {span}")

        for start_qubit in positions:
            new_terms.append((gate, start_qubit, avg_coeff))

    log.info(
        "reconstruct_hamiltonian: reconstructed (uniform)",
        extra={
            "context": {
                "n_qubits": n_qubits,
                "n_original_terms": len(terms),
                "gate_types": list(avg_coeff_by_gate_type),
            }
        },
    )

    return build_hamiltonian(n_qubits, new_terms)
