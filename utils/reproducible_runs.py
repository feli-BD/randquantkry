##########################################################################
#
#  Reproducible, crash-safe I/O for long-running parameter studies.
#
###########################################################################
"""
Each run gets its own timestamped directory containing:
  - metadata.json : sweep configuration, seed, package versions, git commit
  - results.jsonl : one JSON line per result row, written and flushed to disk
                    IMMEDIATELY as each point is computed -- if the process
                    dies partway through (OOM, crash, Ctrl+C), everything
                    computed so far is still safely on disk and loadable.
"""

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _get_package_versions(packages=("numpy", "scipy", "qiskit", "pandas")):
    import importlib.metadata
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def _get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def start_run(output_dir, run_config):
    """
    Create a new timestamped run directory and write its metadata.json.

    Parameters
    ----------
    output_dir : str or Path
        Parent directory to create the timestamped run folder inside.
    run_config : dict
        Everything needed to reproduce this run -- sweep lists, methods,
        seed, etc. Stored verbatim in metadata.json, alongside auto-captured
        package versions and (if available) the current git commit hash.

    Returns
    -------
    Path
        The created run directory (e.g. results/run_20260721T143205Z/).
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_config": run_config,
        "package_versions": _get_package_versions(),
        "git_commit": _get_git_commit(),
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return run_dir


def append_result_row(run_dir, row):
    """
    Append a single result row to results.jsonl, flushing and fsyncing
    IMMEDIATELY so it survives a crash that happens right after this call.
    """
    path = Path(run_dir) / "results.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_run(run_dir):
    """
    Load a run's metadata and results (even if the run crashed partway
    through -- whatever rows were written are still loadable).

    Returns
    -------
    (pandas.DataFrame, dict)
        The results so far, and the run's metadata.
    """
    run_dir = Path(run_dir)

    with open(run_dir / "metadata.json") as f:
        metadata = json.load(f)

    rows = []
    results_path = run_dir / "results.jsonl"
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    return pd.DataFrame(rows), metadata