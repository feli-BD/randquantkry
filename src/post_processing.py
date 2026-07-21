###############################################################################
#
#             Postprocessing function responsible for plotting
#
##############################################################################
"""
Takes the plain dicts produced by QuantumBenchmark (benchmark.py) and turns
them into styled matplotlib figures. Deliberately separate from benchmark.py:
computation is expensive and stable; plotting is cheap and gets tweaked
constantly -- keeping them apart means styling changes never require
re-running a sweep, and benchmark.py stays free of any matplotlib dependency
(so it can be tested headlessly in CI without a display backend).
"""

# imports
import matplotlib.pyplot as plt
from .benchmark import Benchmark

# One consistent color/marker/linestyle per method, reused across every plot
# in this module, so "randomized_krylov" is always the same green line
# whether it's on a runtime plot or a memory plot.
METHOD_STYLE = {
    "naive": {"color": "#E8871E", "marker": "o", "linestyle": "-", "label": "Naive (exact)"},
    "krylov": {"color": "#2E5EAA", "marker": "s", "linestyle": "--", "label": "Krylov"},
    "randomized_krylov": {"color": "#1B8A5A", "marker": "^", "linestyle": "-", "label": "Randomized Krylov"},
    "randomized_krylov_gaussian": {"color": "#8A5FBF", "marker": "D", "linestyle": ":", "label": "Randomized Krylov (Gaussian)"},
    "trotter": {"color": "#C2496D", "marker": "v", "linestyle": "-.", "label": "Trotter"},
}


def _apply_base_style(ax, xlabel, ylabel, title):
    """Shared visual styling: consistent typography, subtle grid, clean spines."""
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle="-", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)


def _filtered_xy(n_qubits, values):
    """Drop (x, y) pairs where y is None -- these mark methods that were
    infeasible/skipped at that n_qubits (e.g. naive above MAX_QUBITS_DENSE),
    and matplotlib can't plot None on a log scale."""
    pairs = [(x, y) for x, y in zip(n_qubits, values) if y is not None]
    if not pairs:
        return [], []
    xs, ys = zip(*pairs)
    return list(xs), list(ys)


def plot_runtime(results, methods=None, title="Runtime vs. System Size", save_path=None, figsize=(8, 5.5)):
    """
    Plot wall-clock runtime vs. n_qubits for each method.

    Parameters
    ----------
    results : dict
        Output of QuantumBenchmark.benchmark_runtime():
        {method: {"n_qubits": [...], "runtime_sec": [...]}}.
    methods : list of str, optional
        Subset of results.keys() to plot; defaults to all methods present.
    save_path : str, optional
        If given, saves the figure (and does not display it inline).
    """
    fig, ax = plt.subplots(figsize=figsize)
    methods = methods if methods is not None else list(results.keys())

    for method in methods:
        if method not in results:
            continue
        style = METHOD_STYLE.get(method, {"color": "gray", "marker": "o", "linestyle": "-", "label": method})
        xs, ys = _filtered_xy(results[method]["n_qubits"], results[method]["runtime_sec"])
        if not xs:
            continue
        ax.plot(xs, ys, color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
                 label=style["label"], linewidth=2.2, markersize=7, markeredgecolor="white", markeredgewidth=0.8)

    ax.set_yscale("log")
    _apply_base_style(ax, xlabel="Number of Qubits", ylabel="Runtime (seconds, log-scale)", title=title)
    ax.legend(fontsize=10, frameon=True, framealpha=0.9, loc="best")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return fig


def plot_memory(results, methods=None, ram_limits=None,
                 title="Quantum Complexity Walls: Memory vs. System Size",
                 save_path=None, figsize=(8, 5.5), empirical=False):
    """
    Plot memory (GB) vs. n_qubits for each method.

    Parameters
    ----------
    results : dict
        Output of QuantumBenchmark.benchmark_memory():
        {method: {"n_qubits": [...], "memory_gb": [...], ("memory_gb_measured": [...])}}.
    ram_limits : dict, optional
        {"label": gb_value} horizontal reference lines, e.g.
        {"Typical PC RAM: 16GB": 16, "Typical HPC RAM: 512GB": 512}
        -- matches the style of common "complexity wall" reference plots.
    empirical : bool, default False
        If True, plot the empirically-MEASURED memory (memory_gb_measured)
        instead of the theoretical estimate -- only valid if the results
        dict was produced with benchmark_memory(empirical=True).
    """
    fig, ax = plt.subplots(figsize=figsize)
    methods = methods if methods is not None else list(results.keys())
    key = "memory_gb_measured" if empirical else "memory_gb"

    for method in methods:
        if method not in results or key not in results[method]:
            continue
        style = METHOD_STYLE.get(method, {"color": "gray", "marker": "o", "linestyle": "-", "label": method})
        xs, ys = _filtered_xy(results[method]["n_qubits"], results[method][key])
        if not xs:
            continue
        ax.plot(xs, ys, color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
                 label=style["label"], linewidth=2.2, markersize=7, markeredgecolor="white", markeredgewidth=0.8)

    if ram_limits:
        for label, gb_value in ram_limits.items():
            ax.axhline(gb_value, color="firebrick", linestyle="--", linewidth=1.5, alpha=0.8)
            ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 1, gb_value, f"  {label}",
                     color="firebrick", fontsize=9, va="center", ha="left")

    ax.set_yscale("log")
    ylabel = "Memory (Gigabytes, log-scale)" + (" -- measured" if empirical else " -- theoretical")
    _apply_base_style(ax, xlabel="Number of Qubits", ylabel=ylabel, title=title)
    ax.legend(fontsize=10, frameon=True, framealpha=0.9, loc="upper left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return fig

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_parameter_study(df, x="krylov_dim", y="runtime_sec", fixed=None, hue="method", save_path=None):
    """
    Plot y vs x from a parameter_study() results DataFrame, one line per
    distinct value of `hue`, after filtering to a fixed slice.

    Two automatic safeguards, based on lessons from earlier plots:

    1. If hue="method" and x is NOT "s_ratio": rows that share a method but
       differ in s_ratio (e.g. randomized_krylov at s_ratio=1.5, 2.0, 3.0)
       would otherwise collapse into one jagged, meaningless line. This
       auto-builds a combined label ("randomized_krylov (s_ratio=2.0)") so
       each sketch size gets its own clean line instead.

    2. If x == "s_ratio": rows with no meaningful s_ratio (e.g. plain
       "krylov", which has s_ratio=None/NaN) are dropped automatically,
       since they'd otherwise contribute a stray single point or a NaN gap.
       hue is forced to "method" in this case (using it as both x and hue
       would create one-point "lines" with nothing to connect).
    """
    data = df.copy()
    if fixed:
        for col, val in fixed.items():
            data = data[data[col] == val]
    if data.empty:
        raise ValueError(f"No rows match fixed={fixed} -- check your filter values against df's actual contents.")

    if x == "s_ratio":
        # rows without a real s_ratio (e.g. plain "krylov") can't sit on this x-axis at all
        data = data.dropna(subset=["s_ratio"])
        if data.empty:
            raise ValueError("No rows have a non-null s_ratio after filtering -- did you mean to include 'krylov'?")
        hue_col = "method"   # using s_ratio as both x and hue would give one-point "lines"
    elif hue == "method" and "s_ratio" in data.columns:
        def make_label(row):
            if pd.isna(row["s_ratio"]):
                return row["method"]
            return f"{row['method']} (s_ratio={row['s_ratio']})"
        data["_hue_label"] = data.apply(make_label, axis=1)
        hue_col = "_hue_label"
    else:
        hue_col = hue

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for hue_val, group in data.groupby(hue_col):
        group_sorted = group.sort_values(x)
        ax.plot(group_sorted[x], group_sorted[y], marker="o", label=str(hue_val), linewidth=2)

    if y in ("runtime_sec", "memory_gb"):
        ax.set_yscale("log")

    ax.set_xlabel(x, fontsize=12)
    ax.set_ylabel(y, fontsize=12)
    title = f"{y} vs {x}" + (f"  (fixed: {fixed})" if fixed else "")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return fig