# Contributing

## Development setup

Create a virtual environment and install the development dependencies with `pip install -e ".[dev]"`.

On Windows PowerShell, activate with `.venv\\Scripts\\Activate.ps1`.

## Checks

Run `pytest` and `ruff check src tests --select E9,F --ignore F401` before opening a pull request.

New functionality should include focused unit tests and, where appropriate, an integration test. Avoid committing generated artifacts, datasets, credentials, or environment-specific paths.

## Pull requests

Keep changes focused. Explain behavioral changes, compatibility implications, and test coverage in the pull request description.
