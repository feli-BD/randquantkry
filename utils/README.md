# `utils/`

Shared infrastructure used across the package for logging and reproducible
run I/O. Kept separate from [`src/`](../src/) because these modules are
infrastructure, not simulation logic — every solver and benchmark in `src/`
depends on them, not the other way around.

## Modules

- **`logging_config.py`** — central logging setup. Provides dual-channel
  logging so benchmark runs are fully traceable: a human-readable console
  stream (`ConsoleContextFormatter`) plus a machine-parseable JSON-lines file
  (`JsonLinesFormatter`, one JSON object per log record). Call
  `configure_logging(...)` once at the start of a session, then have every
  module fetch its logger via `get_logger(__name__)` so all output flows
  through the same handlers into one coherent, timestamped audit trail.

- **`reproducible_runs.py`** — crash-safe I/O for long-running parameter
  studies. `start_run(output_dir, run_config)` creates a timestamped run
  directory containing `metadata.json` (sweep configuration, seed, package
  versions, git commit, hardware/environment info) and `results.jsonl`, to
  which `append_result_row` writes and flushes one JSON line per result
  *immediately* as it's computed — if the process dies partway through (OOM,
  crash, Ctrl+C), everything computed so far is still safely on disk.
  `load_run(run_dir)` reads a run back in; `hash_terms` fingerprints a
  Hamiltonian's term list for cache/comparison keys.

- **`meta_metadata_file_generator.py`** — `build_run_index(results_dir)`
  scans a `results/` directory of past runs and builds a single `pandas`
  index/summary DataFrame over their metadata, for browsing or comparing runs
  at a glance.

## Related

- Consumers of this infrastructure live in [`src/`](../src/), notably
  `benchmark.py` (uses both `logging_config` and `reproducible_runs`).
