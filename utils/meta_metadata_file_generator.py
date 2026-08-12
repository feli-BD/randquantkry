##########################################################################
#
#            Create a master index of all runs in the results folder, 
#               by scanning for metadata.json files
#
###########################################################################
"""
A script to scan the results folder for all runs and create a master index of their metadata.
The index is saved as a CSV file and a Markdown table for easy viewing in VS Code.
"""
#imports
import json
from pathlib import Path
import pandas as pd


def _format_list(val_list: list) -> str:
    """Helper to convert list [6, 8, 10] into a clean string '6, 8, 10' without brackets."""
    if not val_list:
        return ""
    return ", ".join(map(str, val_list))


def build_run_index(results_dir: str = "results") -> pd.DataFrame:
    """
    Function to scan the results folder for all runs and create a master index of their metadata.
    Parameters
    ----------
    results_dir : str
        The path to the results folder containing run directories.
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the metadata of all runs found in the results folder.
    The index is saved as a CSV file and a Markdown table for easy viewing in VS Code
    """
    results_path = Path(results_dir)
    records = []

    # Iterate through all run folders (e.g., results/run_20260723T081242Z)
    for run_folder in sorted(results_path.glob("run_*")):
        metadata_file = run_folder / "metadata.json"

        if not metadata_file.exists():
            continue

        try:
            with open(metadata_file) as f:
                meta = json.load(f)

            cfg = meta.get("run_config", {})

            # Extract key experiment parameters as clean comma-separated strings
            record = {
                "run_id": run_folder.name,
                "timestamp": meta.get("timestamp_utc", ""),
                "qubits": _format_list(cfg.get("n_qubits_list", [])),
                "krylov_dims": _format_list(cfg.get("krylov_dim_list", [])),
                "times": _format_list(cfg.get("time_list", [])),
                "methods": ", ".join(cfg.get("methods", [])),
                "s_ratios": _format_list(cfg.get("s_ratio_list", [])),
                "seed": cfg.get("seed", None),
            }
            records.append(record)

        except (
            Exception  # noqa: BLE001 —> skip unparseable metadata files, don't abort the scan
        ) as e:
            print(f"Warning: Could not parse {metadata_file}: {e}")

    df = pd.DataFrame(records)

    if not df.empty:
        # Save master index files to results root
        csv_out = results_path / "run_index.csv"
        df.to_csv(csv_out, index=False)

        # Save Markdown overview table to read directly in VS Code / GitHub
        md_out = results_path / "RUN_INDEX.md"
        with open(md_out, "w") as f:
            f.write("# Experiment Run Catalog\n\n")
            f.write(
                df[
                    [
                        "run_id",
                        "qubits",
                        "krylov_dims",
                        "times",
                        "methods",
                        "s_ratios",
                    ]
                ].to_markdown(index=False)
            )

        print(f"Successfully cataloged {len(df)} runs -> {csv_out} and {md_out}")
    else:
        print("No run directories with valid metadata.json found.")

    return df


if __name__ == "__main__":
    build_run_index()
