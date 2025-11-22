# Contributing to iAODE

Welcome! This project thrives on community contributions. Please follow the steps below to get up and running and make your work easy to review.

## Getting Started

1. **Clone the repository**:

    ```bash
    git clone https://github.com/PeterPonyu/iAODE.git
    cd iAODE
    ```

2. **Create and activate a virtual environment**:

    ```bash
    python -m venv .venv
    source .venv/bin/activate   # Windows: .venv\Scripts\activate
    python -m pip install --upgrade pip
    ```

3. **Install development dependencies**:

    ```bash
    pip install -e ".[dev]"
    ```

4. **Keep dependencies in sync** by updating `requirements.txt` or `pyproject.toml` whenever you add new packages.

## Development Workflow

1. **Fork** the repository if you do not have write access.
2. **Create a feature branch**: `git checkout -b feat/your-topic` (use a name that reflects the change).
3. **Make incremental commits** with clear messages referencing the work (e.g., `feat: add encoder factory tests`).
4. **Run formatting and linting** before committing (see Code Standards).
5. **Push** your branch: `git push origin feat/your-topic`.
6. **Open a Pull Request** against `main` with a concise summary, testing performed, and any follow-up notes.

## Code Standards

- Follow [PEP 8](https://peps.python.org/pep-0008/) styling.
- Prefer descriptive type hints and docstrings for all public APIs.
- Use the configured formatter and linters:

  ```bash
  black iaode/          # enforce formatting (line length 100)
  ```

- Explain complex logic with brief inline comments or helper functions rather than long paragraphs.
- Keep imports ordered and grouped (stdlib, third-party, local).

## Testing

- Write or update pytest tests for any functional change, bug fix, or new feature.
- Tests should live under the `tests/` directory or near the components they target.
- Execute the full suite before pushing:

  ```bash
  pytest
  ```

- If a test is flaky, document the failure in the PR so reviewers can rerun the suite.

## Documentation & Examples

- Update `README.md` or `docs/` when adding new user-facing functionality.
- Add runnable examples under `examples/` for new workflows or APIs.
- Keep docstrings consistent: include `Args`, `Returns`, `Raises`, and a short usage snippet when appropriate.

## Reporting Issues

When opening an issue, include:

1. Python and iAODE versions (`python -V`, `pip show iaode`).
2. OS/environment details.
3. Minimal code snippet to reproduce the problem.
4. Full error trace or logs.
5. Expected vs actual behavior.

Issues that do not follow these steps may be delayed for clarification.

## Pull Request Expectations

- Link related issues or discussions in the PR description.
- Describe the problem, your solution, and verification steps (commands/outputs).
- Add screenshots or sample outputs when relevant.
- Tag maintainers or reviewers if the change requires special attention.
- Keep PRs focused—larger changes may be easier to review if split into multiple PRs.

## Adding New Components

- **Encoders/Losses**: extend the relevant module (`iaode/module.py`, `iaode/model.py`), update the factory methods, and add tests.
- **Evaluation**: update `iaode/DRE.py` or `iaode/LSE.py`, document the interpretation of new metrics, and include usage examples.
- **New APIs**: ensure updates propagate to documentation, `examples/`, and tests.

## Community & Support

- Ask questions in Discussions or search Issues before opening a new thread.
- Respect others’ time—keep comments focused and constructive.
- If you are unsure about the scope, open an issue to discuss the direction before coding.

## Licensing

By contributing, you agree all your contributions are covered under the [MIT License](LICENSE).

Thank you for helping iAODE grow!
