##########################################################################
#
#                  Observable Functions for Quantum Evolution
#                  Bridge between quantum and classical Observables
#
###########################################################################

# imports
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import Latex, display
from scipy.sparse import csr_matrix, issparse


def print_quantum_state(state, label="|ψ⟩", threshold=1e-5, reverse_bits=False):
    """Print complex state vector in clean ASCII Dirac notation."""
    n_qubits = int(np.round(np.log2(len(state))))
    terms = []

    for i, amp in enumerate(state):
        if np.abs(amp) > threshold:
            bitstring = format(i, f"0{n_qubits}b")
            if reverse_bits:
                bitstring = bitstring[::-1]

            real, imag = amp.real, amp.imag

            if np.isclose(amp, 1.0):
                amp_str = ""
            elif np.isclose(amp, -1.0):
                amp_str = "-"
            elif abs(imag) < threshold:
                amp_str = f"{real:+.4f} "
            elif abs(real) < threshold:
                amp_str = f"{imag:+.4f}j "
            else:
                amp_str = f"({real:+.4f}{imag:+.4f}j) "

            terms.append(f"{amp_str}|{bitstring}⟩")

    print(f"{label} = " + " + ".join(terms).replace("+ -", "- "))


def plot_unitary_matrix(unitary):
    """Plot the magnitude heatmap of a Unitary matrix."""
    _, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.abs(unitary), cmap="viridis")

    plt.colorbar(im, label="|U_ij|")
    ax.set_title("Unitary Matrix Magnitude $|U(t)|$", fontweight="bold")
    ax.set_xlabel("Input State Index")
    ax.set_ylabel("Output State Index")
    plt.tight_layout()
    plt.show()


def display_quantum_state_latex(state, label=r"|\psi\rangle", threshold=1e-5, reverse_bits=False):
    """Render a NumPy state vector as a LaTeX Dirac formula in Jupyter."""
    n_qubits = int(np.round(np.log2(len(state))))
    terms = []

    for i, amp in enumerate(state):
        if np.abs(amp) > threshold:
            # 0 -> "00000000"
            bitstring = format(i, f"0{n_qubits}b")

            if reverse_bits:
                bitstring = bitstring[::-1]

            real, imag = amp.real, amp.imag

            if np.isclose(amp, 1.0):
                amp_str = ""  # |00000000⟩
            elif np.isclose(amp, -1.0):
                amp_str = "-"  #  -|00000000⟩
            elif abs(imag) < threshold:
                amp_str = f"{real:+.4f} "
            elif abs(real) < threshold:
                amp_str = f"{imag:+.4f}j "
            else:
                amp_str = f"({real:+.4f}{imag:+.4f}j) "

            terms.append(f"{amp_str}|{bitstring}\\rangle")

    latex_str = " + ".join(terms).replace("+ -", "- ")
    display(Latex(f"$${label} = {latex_str}$$"))


def plot_state_histogram(state, threshold=1e-3, title="State Probabilities"):
    """Plot a bar chart of measurement probabilities P(x) = |c_x|^2 above a threshold."""
    n_qubits = int(np.round(np.log2(len(state))))

    # Compute Born rule probabilities |c_x|^2
    probs = np.abs(state) ** 2

    # Filter states with probability above the threshold
    indices = np.where(probs > threshold)[0]
    filtered_probs = probs[indices]
    bitstrings = [format(i, f"0{n_qubits}b") for i in indices]

    # Plot
    _, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(bitstrings, filtered_probs, color="#6200ee", alpha=0.85)

    ax.set_ylabel("Probability $P(x) = |c_x|^2$", fontsize=11)
    ax.set_xlabel("Basis State $|x\\rangle$", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(filtered_probs) * 1.15 if len(filtered_probs) > 0 else 1.0)
    plt.xticks(rotation=70 if len(bitstrings) > 6 else 0)

    # Add text labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.show()


# Operator representation


def plot_hamiltonian_heatmap(H, title="Hamiltonian Matrix Elements |H_ij|"):
    """Plot a heatmap of the absolute magnitude of matrix elements."""
    H_dense = H.toarray() if hasattr(H, "toarray") else H

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.abs(H_dense), cmap="viridis", interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("$|H_{ij}|$", rotation=270, labelpad=15, fontsize=11)

    ax.set_title(
        f"{title}\n({H_dense.shape[0]}x{H_dense.shape[1]})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("State $|j\\rangle$", fontsize=10)
    ax.set_ylabel("State $|i\\rangle$", fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_operator_sparsity(
    operator,
    threshold=1e-8,
    is_unitary=None,
    title=None,
    markersize=1.5,
    color=None,
    ax=None,
):
    """Plot the sparsity pattern (spy plot) for an operator (Hamiltonian or Unitary).

    Parameters:
    -----------
    operator : array-like, SciPy sparse matrix, or Qiskit SparsePauliOp
        The operator matrix to plot.
    threshold : float, default=1e-8
        Entries with absolute value <= threshold are treated as zero.
    is_unitary : bool or None, default=None
        If None, automatically inferred based on whether max element <= 1.0.
    title : str or None, default=None
        Custom plot title.
    markersize : float, default=1.5
        Size of the dots in the spy plot.
    color : str or None, default=None
        Color of non-zero entries (defaults to '#6200ee' for H, '#1f77b4' for U).
    ax : matplotlib.axes.Axes or None, default=None
        Axes on which to draw the plot.
    """
    # 1. Convert to a sparse matrix representation safely
    if hasattr(operator, "to_matrix"):
        try:
            matrix = operator.to_matrix(sparse=True)
        except TypeError:
            matrix = operator.to_matrix()
    elif callable(operator):
        matrix = operator()
    else:
        matrix = operator

    if not issparse(matrix):
        matrix = csr_matrix(matrix)

    dim_i, dim_j = matrix.shape

    # 2. Infer operator type if not provided
    if is_unitary is None:
        max_val = np.max(np.abs(matrix.data)) if matrix.nnz > 0 else 0.0
        is_unitary = max_val <= 1.0 + 1e-5

    # 3. Apply threshold to construct non-zero boolean mask
    mask = np.abs(matrix.data) > threshold
    sparse_mask = csr_matrix((mask, matrix.indices, matrix.indptr), shape=matrix.shape)

    # 4. Set visual style based on operator type
    if is_unitary:
        # default_color = "#1f77b4"  # Deep Blue
        default_color = "#6200ee"  # Purple
        default_title = "Unitary Sparsity Pattern"
        xlabel = "Initial State Index ($j$)"
        ylabel = "Final State Index ($i$)"
    else:
        default_color = "#6200ee"  # Purple
        default_title = "Hamiltonian Sparsity Pattern"
        xlabel = "Basis State Index ($j$)"
        ylabel = "Basis State Index ($i$)"

    plot_color = color or default_color

    # 5. Plot using ax.spy
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.spy(sparse_mask, markersize=markersize, color=plot_color)

    ax.set_title(
        title or f"{default_title}\n({dim_i}x{dim_j} Matrix)",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)

    plt.tight_layout()

    return ax


def plot_operator_subblock(operator, block_size=8, is_unitary=None, title=None, ax=None):
    """Plot an annotated sub-block of an operator (Hamiltonian or Unitary).

    Parameters:
    -----------
    operator : array-like, SciPy sparse matrix, or Qiskit SparsePauliOp
        The operator matrix to plot.
    block_size : int, default=8
        Dimension of the sub-block (block_size x block_size).
    is_unitary : bool or None, default=None
        If None, automatically inferred based on whether max element <= 1.0.
    title : str or None, default=None
        Custom plot title.
    ax : matplotlib.axes.Axes or None, default=None
        Axes on which to draw the plot.
    """
    # 1. Convert operator to dense matrix
    if hasattr(operator, "to_matrix"):
        matrix = operator.to_matrix()
    elif hasattr(operator, "toarray"):
        matrix = operator.toarray()
    elif callable(operator):
        matrix = operator()
    else:
        matrix = np.asarray(operator)

    n_qubits = int(np.round(np.log2(matrix.shape[0])))
    block_size = min(block_size, matrix.shape[0])

    # 2. Infer operator type if not provided
    if is_unitary is None:
        # Check if bound within [0, 1] (characteristic of Unitary matrices)
        is_unitary = np.max(np.abs(matrix)) <= 1.0 + 1e-5

    # 3. Slice top-left sub-block & generate basis labels
    sub_block = matrix[:block_size, :block_size]
    labels = [format(i, f"0{n_qubits}b") for i in range(block_size)]

    # 4. Configure visual style based on operator type
    if is_unitary:
        cmap = "Blues"
        cbar_label = r"$|U_{ij}|$"
        vmin, vmax = 0.0, 1.0
        annot_data = np.abs(sub_block)  # Magnitudes for Unitary
        default_title = f"Unitary Sub-block U(t) ({block_size}x{block_size})"
        xlabel = r"Initial State $|j\rangle$"
        ylabel = r"Final State $|i\rangle$"
    else:
        cmap = "coolwarm"
        cbar_label = r"$|H_{ij}|$"
        vmin, vmax = None, None
        annot_data = sub_block.real  # Real values for Hamiltonian
        default_title = f"Hamiltonian Sub-block H ({block_size}x{block_size})"
        xlabel = r"State $|j\rangle$"
        ylabel = r"State $|i\rangle$"

    # 5. Plot heatmap
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    sns.heatmap(
        np.abs(sub_block),
        annot=annot_data,
        fmt=".2f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": cbar_label},
        ax=ax,
    )

    ax.set_title(
        title or default_title,
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    return ax
