##########################################################################
#
#                       Time Evolution of Quantum Systems
#                     Methods: Naive, Krylov and Randomized Krylov
#
###########################################################################

# imports
from scipy import linalg
from scipy.linalg import expm
from scipy import sparse
import numpy as np

def time_evolve_naive(hamiltonian, initial_state, time):
    """
    Perform naive time evolution of a quantum system.

    Parameters
    ----------
    hamiltonian : function
        The Hamiltonian function.
    initial_state : array-like
        The initial state of the quantum system.
    time : float
        The evolution time.

    Returns
    -------
    unitary : array-like
        The unitary operator used for the evolution.
    evolved_state : array-like
        The evolved state of the quantum system.
    """
    # calculate eigenvector and eigenvalues of the Hamiltonian
    H = hamiltonian.to_matrix()
    eigenvals, eigenvecs = np.linalg.eigh(H)

    # compute phase factors
    phases = np.exp(-1j * eigenvals * time)
    # Evolve initial state using unitary
    unitary = eigenvecs @ np.diag(phases) @ eigenvecs.conj().T 
    evolved_state = unitary @ initial_state

    #TODO: add fail loud option for when qubits are too many and the matrix is too large to compute

    return unitary, evolved_state



def time_evolve_krylov(hamiltonian, initial_state, time, krylov_dim):
    """
    Perform time evolution using the Krylov subspace method.

    Parameters
    ----------
    hamiltonian : function
        The Hamiltonian function.
    initial_state : array-like
        The initial state of the quantum system.
    time : float
        The evolution time.
    krylov_dim : int
        The dimension of the Krylov subspace.

    Returns
    -------
    array-like
        The evolved state of the quantum system.
    """

    H = hamiltonian.to_matrix()
    v = initial_state / np.linalg.norm(initial_state)
    V = [v]  # list to store Krylov vectors
    # store orthogonalized vectors and coefficients
    alphas = []
    betas = []

    for m in range(krylov_dim):
        
        # cheap matrix vector product
        mat_vec_prod = H @ V[m]

        # orthogonalize against previous vectors
        alpha_m = np.conjugate(V[m]) @ mat_vec_prod
        mat_vec_prod -= alpha_m * V[m]

        for i in range(m):
            overlap = np.conjugate(V[i]) @ mat_vec_prod
            mat_vec_prod -= overlap * V[i]
    
        alphas.append(alpha_m)
        beta_m = np.linalg.norm(mat_vec_prod)

        if beta_m < 1e-12:
            break

        betas.append(beta_m)
        V.append(mat_vec_prod / beta_m) # next Krylov vector

        # assemble the tridiagonal matrix
        T = np.tridiagonal_matrix(alphas, betas)

        # evolve in the Krylov subspace 
        evolved_state_krylov = expm(-1j * T * time) @ np.array([1] + [0] * (len(V) - 1))
        evolved_state = np.linalg.norm(initial_state) * (np.array(V[:-1]) @ evolved_state_krylov)

        # TODO: add logging and fail loud option for when krylov_dim is too small and the approximation is not accurate enough
        return evolved_state



def time_evolve_randomized_krylov(hamiltonian, initial_state, time, krylov_dim, num_random_vectors):
    """
    Perform time evolution using the randomized Krylov subspace method.

    Parameters
    ----------
    hamiltonian : function
        The Hamiltonian function.
    initial_state : array-like
        The initial state of the quantum system.
    time : float
        The evolution time.
    krylov_dim : int
        The dimension of the Krylov subspace.
    num_random_vectors : int
        The number of random vectors to use in the randomized Krylov method.

    Returns
    -------
    array-like
        The evolved state of the quantum system.
    """