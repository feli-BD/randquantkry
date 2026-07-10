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

hamiltonian = build_hamiltonian(4, [('XX', 0, 1.0), ('ZZ', 1, 0.5)])
print(hamiltonian)
