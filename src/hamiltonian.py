##########################################################################
#
#                       Build Hamiltonian Function
#
###########################################################################

# imports
from qiskit.quantum_info import SparsePauliOp, Operator
import numpy as np
from scipy.linalg import expm


def build_hamiltonian(n_qubits: int,  terms: list[tuple[str, int, float]], angle: float = 0.0) -> SparsePauliOp:
    """
    Build the Hamiltonian function from the given number of qubits.

    Parameters
    ----------
    n_qubits : int
        The number of qubits for the Hamiltonian.
    terms (list): List of (gate, start_qubit, coeff) tuples. 
                'gate' (str) defines the Pauli string (e.g., "XX").
                'start_qubit' (int) is the index where the interaction begins.
                'coeff' (float) is the term's scalar weight.
    Returns
    -------
    SparsePauliOp
        A SparsePauliOp object that represents the Hamiltonian function, which can be called with the state variables to compute the Hamiltonian matrix.
        
    """

    labels = []
    coefficients = []

    for gate, qubit, coeff in terms:
        s = ['I'] * n_qubits
        
        s[(qubit)] = gate[0]
        if len(gate)==2:
            s[(qubit+1)] = gate[1]
        
        labels.append(''.join(s))
        coefficients.append(coeff)

    hamiltonian = SparsePauliOp(labels, coeffs=np.array(coefficients, dtype=complex))
    return hamiltonian

def build_initial_state(n_qubits=int, flipped_qubit=int):
    initial_state = np.zeros(2**n_qubits, dtype=complex)
    initial_state[flipped_qubit] = 1.0   # |0000...0>)
    return initial_state

def reconstruct_hamiltonian(n_qubits: int, hamiltonian: SparsePauliOp):
    terms = []
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        label = str(pauli)
        # find positions where the label is NOT identity
        nonzero_positions = [i for i, ch in enumerate(label) if ch != 'I']
 
        if not nonzero_positions:
            continue  # a pure-identity term contributes nothing meaningful to re-extract
 
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
 
        gate = ''.join(label[i] for i in nonzero_positions)
        terms.append((gate, start_qubit, complex(coeff)))

    coeffs_by_gate_type = {}
    for gate, _start_qubit, coeff in terms:
        coeffs_by_gate_type.setdefault(gate, []).append(coeff)

    avg_coeff_by_gate_type = {
        gate: np.mean(coeff_list) for gate, coeff_list in coeffs_by_gate_type.items()
    }

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
    
    return build_hamiltonian(n_qubits, new_terms)

