# Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible dependency
management.

```bash
git clone <repo-url>
cd randquantkry
uv sync --extra dev
```

## Using notebooks with the project's environment (recommended)

```bash
uv add --dev ipykernel
uv run python -m ipykernel install --user --name randquantkry --display-name "randquantkry (uv)"
```

Then select **"randquantkry (uv)"** as the kernel in Jupyter/VS Code. You might need to restart
VS Code to find the new kernel. Alternatively, run `uv run --extra dev jupyter lab` to launch
Jupyter directly inside the project's environment without registering a persistent kernel.

## Alternative: pip

```bash
python -m venv .venv
source .venv/bin/activate  # Linux / Mac
.venv\Scripts\activate.bat  # Windows
pip install -e ".[dev]"
```

## Testing

```bash
uv run pytest --cov=randquantkry
```

To test the CI pipeline itself locally before pushing (catches YAML/config issues without
waiting for a real pipeline run):

```bash
gitlab-ci-local              # full pipeline
gitlab-ci-local unit-tests   # a single job
```