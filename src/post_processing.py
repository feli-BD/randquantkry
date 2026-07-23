from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_parameter_study(
    df,
    x="krylov_dim",
    y="evolved_state_norm",
    fixed=None,
    hue="method",
    y_log=None,
    reference_line=None,
    save_path=None,
    run_dir=None, 
    filename=None,  
):
    """Plot y vs x from a parameter_study() results DataFrame.

    Parameters
    ----------
    run_dir : str or Path, optional
        Path to the specific run directory (e.g. `output_dir`).
        If provided and `save_path` is None, the plot will automatically be
        saved into `run_dir / plots / {y}_vs_{x}.png`.
    filename : str, optional
        Custom image filename if `run_dir` is used.
    """
    data = df.copy()

    # 1. Apply fixed slice filters
    if fixed:
        for col, val in fixed.items():
            data = data[data[col] == val]
    if data.empty:
        raise ValueError(
            f"No rows match fixed={fixed} -- check your filter values against df's actual contents."
        )

    # 2. Handle s_ratio safeguards
    if x == "s_ratio":
        data = data.dropna(subset=["s_ratio"])
        if data.empty:
            raise ValueError(
                "No rows have a non-null s_ratio after filtering -- did you mean to include 'krylov'?"
            )
        hue_col = "method"
    elif hue == "method" and "s_ratio" in data.columns:

        def make_label(row):
            if pd.isna(row["s_ratio"]):
                return row["method"]
            return f"{row['method']} (s_ratio={row['s_ratio']})"

        data["_hue_label"] = data.apply(make_label, axis=1)
        hue_col = "_hue_label"
    else:
        hue_col = hue

    # 3. Create plot
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for hue_val, group in data.groupby(hue_col):
        group_sorted = group.sort_values(x)
        ax.plot(
            group_sorted[x],
            group_sorted[y],
            marker="o",
            label=str(hue_val),
            linewidth=2,
            alpha=0.85,
        )

    # 4. Reference line
    if reference_line is not None:
        ax.axhline(
            y=reference_line,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"Ideal ({reference_line})",
            zorder=1,
        )

    # 5. Log-scale decision logic
    auto_log_metrics = {"runtime_sec", "memory_gb", "rel_error", "norm_error"}
    should_log = (
        y_log if y_log is not None else (y in auto_log_metrics or "error" in y)
    )

    if should_log:
        ax.set_yscale("log")

    # 6. Formatting & Labels
    ax.set_xlabel(x, fontsize=12)
    ax.set_ylabel(y, fontsize=12)
    title = f"{y} vs {x}" + (f"  (fixed: {fixed})" if fixed else "")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3, linestyle=":")
    plt.tight_layout()

    # 7. Dynamic Resolution for Output Directory / Saving
    target_save_path = save_path

    if target_save_path is None and run_dir is not None:
        target_dir = Path(run_dir) / "plots"
        target_dir.mkdir(
            parents=True, exist_ok=True
        )  # Creates run_dir/plots if missing

        # Generate a descriptive filename if not specified
        if not filename:
            fixed_suffix = (
                "_" + "_".join(f"{k}{v}" for k, v in fixed.items())
                if fixed
                else ""
            )
            filename = f"{y}_vs_{x}{fixed_suffix}.png"

        target_save_path = target_dir / filename

    if target_save_path:
        plt.savefig(target_save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {target_save_path}")
        plt.close(fig)
        plt.show()

    return fig