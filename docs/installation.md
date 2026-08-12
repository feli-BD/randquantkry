# Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible dependency
management.

**Just want to run the code** (no tests, linting, or docs tooling):

```bash
git clone https://gitlab.git.nrw/felicia.erb/randquantkry.git # <- https clone for ssh clone replace repo link with: git@gitlab.git.nrw:felicia.erb/randquantkry.git
cd randquantkry
uv sync --no-dev
```

**Contributing, testing, or grading this project** (installs pytest, ruff, black, and the docs
toolchain in addition to the core dependencies):

```bash
git clone https://gitlab.git.nrw/felicia.erb/randquantkry.git # <- https clone for ssh clone replace repo link with: git@gitlab.git.nrw:felicia.erb/randquantkry.git
cd randquantkry
uv sync --extra dev
```

To use notebooks with this project's environment (RECOMMENDED):

```bash
uv run python -m ipykernel install --user --name randquantkry --display-name "randquantkry (uv)"
```

Then select **"randquantkry (uv)"** as the kernel in Jupyter/VS Code. You might need to restart
VS Code to find the new kernel. Alternatively, run `uv run jupyter lab` (after `uv sync --extra dev`)
to launch Jupyter directly inside the project's environment without registering a persistent kernel.

If you prefer `pip`:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux (or Mac)
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