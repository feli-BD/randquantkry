##########################################################################
#
#  Reproducible, crash-safe I/O for long-running parameter studies.
#
###########################################################################
"""
Each run gets its own timestamped directory containing:
  - metadata.json : sweep configuration, seed, package versions, git commit
  - results.jsonl : one JSON line per result row, written and flushed to disk
                    IMMEDIATELY as each point is computed, if the process
                    dies partway through (OOM, crash, Ctrl+C), everything
                    computed so far is still safely on disk and loadable.
  - run.log.jsonl : a JSON-lines file (one JSON object per line) that logs
                    all run information.
"""
# Imports
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
import importlib.metadata


def _get_package_versions(packages=("numpy", "scipy", "qiskit", "pandas")):
    """Get the versions of key packages used in this run."""
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def _get_git_commit():
    """Get the current git commit hash, if available."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],  # noqa: S607 —> trusted local git call, not exposed to untrusted input
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (
        Exception  # noqa: BLE001 —> best-effort metadata, failure just means it's unavailable
    ):
        return None


def _get_hardware():
    """Get basic hardware information, if available."""
    try:
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(logical=True),
            "total_memory_gb": round(psutil.virtual_memory().total / 1e9, 2),
        }
    except (
        Exception  # noqa: BLE001 —> best-effort metadata, failure just means it's unavailable
    ):
        return None


def _get_environment_metadata():
    """Get metadata about the Python environment, including the lock file hash and BLAS backend."""
    repo_root = Path(__file__).resolve().parents[1]
    lock_path = repo_root / "uv.lock"
    metadata = {
        # Exact Python interpreter, not just package versions
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),  # CPython vs PyPy
        # Lock file hash
        "uv_lock_sha256": _hash_file(lock_path),
        # The actual math backend NumPy/SciPy are using — this is the missing piece
        "numpy_blas_info": _get_blas_backend_info(),
    }
    return metadata


def _hash_file(path: str) -> str:
    """Helper function: Compute the SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _get_blas_backend_info():
    """Extract which BLAS/LAPACK library NumPy is actually linked against."""
    config = np.show_config(mode="dicts")
    return {
        "blas": config.get("Build Dependencies", {}).get("blas", {}).get("name", "unknown"),
        "lapack": config.get("Build Dependencies", {}).get("lapack", {}).get("name", "unknown"),
    }


def _method_shorthand(methods):
    """Build a short, readable tag from a list of method names."""
    shorthand_map = {
        "naive": "nv",
        "krylov": "kry",
        "randomized_krylov": "rkry",
        "randomized_krylov_gaussian": "rkryg",
        "trotter": "trot",
    }
    tags = [shorthand_map.get(m, m[:4]) for m in methods]  # fallback: first 4 chars of unknown methods
    return "-".join(tags)


def _qubit_range_shorthand(n_qubits_list):
    """Build a compact qubit-range tag, e.g. [4,6,8,10] -> 'q4-10'."""
    if not n_qubits_list:
        return "qNA"
    lo, hi = min(n_qubits_list), max(n_qubits_list)
    return f"q{lo}" if lo == hi else f"q{lo}-{hi}"


def start_run(output_dir, run_config):
    """
    Create a new timestamped run directory and write its metadata.json.

    Parameters
    ----------
    output_dir : str or Path
        Parent directory to create the run folder inside.
    run_config : dict
        Everything needed to reproduce this run -- sweep lists, methods,
        seed, etc. Stored verbatim in metadata.json, alongside auto-captured
        package versions and (if available) the current git commit hash.
        Used here to build a readable run_id: run_id is built from
        run_config["methods"] and run_config["n_qubits_list"] when present
        (falling back to generic tags if either is missing, so this stays
        safe for config shapes that don't include them).

    Returns
    -------
    Path
        The created run directory, e.g.
        results/run_kry-rkry_q4-10_20260721T143205Z/
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    methods = run_config.get("methods", [])
    n_qubits_list = run_config.get("n_qubits_list", [])

    method_tag = _method_shorthand(methods) if methods else "nomethod"
    qubit_tag = _qubit_range_shorthand(n_qubits_list) if n_qubits_list else "qNA"

    run_id = f"run_{method_tag}_{qubit_tag}_{timestamp}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "run_config": run_config,
        "environment": _get_environment_metadata(),
        "package_versions": _get_package_versions(),
        "git_commit": _get_git_commit(),
        "hardware": _get_hardware(),
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


# Hash term for checksum calculation

def hash_terms(terms: list[tuple[str, int, float]]) -> str:
    """Checksums for verifying experiment inputs haven't
    silently changed between runs."""

    def _serialize_coeff(c):
        if isinstance(c, complex):
            return [c.real, c.imag]
        return c

    serializable = json.dumps(
        [[gate, qubit, _serialize_coeff(coeff)] for gate, qubit, coeff in terms],
        sort_keys=True,
    )
    return hashlib.sha256(serializable.encode()).hexdigest()
