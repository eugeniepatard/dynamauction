# Contributing to DynamAuction

Thank you for your interest in DynamAuction! Bug reports, questions, new scenarios, baselines, documentation fixes and
code improvements are all welcome. By taking part in this project you agree to follow its
[Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- **Report a bug or ask a question:** open an issue. For a bug, give the DynamAuction and Python versions, the code
  you ran (the scenario preset and seed if possible) and what you expected.
- **Propose a change:** open a pull request (see below). Adding something new (a kernel, a policy family, a preset, a
  primitive) can go straight to a pull request. To change something that already exists (the environment dynamics, an
  existing kernel, primitive, preset or policy), please open an issue first so that we can discuss it.

## Development setup

DynamAuction needs Python 3.10 or later.

```bash
git clone <your fork's URL> dynamauction
cd dynamauction
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # the library + pytest, black, flake8
pip install -e ".[benchmark]"      # optional: RL training, the MLP valuation head, the benchmark scripts
```

## Running the tests

```bash
pytest tests/                                  # the default run
pytest tests/ -m "not slow and not paper"      # the quick run (what most CI jobs run)
pytest tests/ -m slow                          # the trained models shipped in benchmark/models/
pytest tests/ -m paper                         # the paper's results (see below)
```

Please add tests for new code and make sure the default run passes before you open a pull request.

### The paper's results

`pytest -m paper` recomputes a few test campaigns of the paper's benchmark and compares them with the results stored
in `benchmark/benchmark_results/main/`. A failure means that the code now produces results that no longer align with
the paper's results. That is expected after an intentional change of the environment, of a kernel or of the planning
code, and it does not block a pull request: in CI this check only raises a warning. If your change moves these numbers
on purpose, say so in the pull request.

## Code style

- Follow the style of the surrounding code: naming, docstrings (NumPy style for public classes, as in the existing
  primitives) and comment density.
- `black` and `flake8` are installed with the `dev` extra. They are not enforced by CI, but please keep new code clean.
- Keep the public API documented: a new primitive, kernel, scenario component or preset goes in
  [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md), and a new preset also in the presets table of the
  [README](README.md).

### Adding a primitive or a scenario

- A new primitive (kernel, carry-over context γ, arrival process, horizon, competition, ...) is a class in
  `dynamauction/primitives/`, registered under a name in `dynamauction/scenarios/builder.py` and exported in
  `dynamauction/primitives/__init__.py` and `dynamauction/__init__.py`.
- A new preset is an entry of `SCENARIO_PRESETS` in `dynamauction/scenarios/presets.py`, with a short description, and a
  test in `tests/unit_tests/test_scenarios/`.
- Changing an existing primitive or preset changes every result computed with it: prefer adding a new one, or explain
  the change in the pull request.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Make your changes, with tests and documentation.
3. Run the tests (`pytest tests/`).
4. Open a pull request against `main` and describe what it changes and why.

CI runs the test suite on Linux, macOS and Windows and several Python versions, executes the tutorial notebooks, and
runs the paper check. CI starts only once a maintainer has approved the run, on every push to the pull request. A
maintainer reviews every pull request, and `main` only changes by merging an approved pull request.

## License

DynamAuction is released under the [MIT License](LICENSE). By contributing, you agree that your contributions are
licensed under the same MIT License.
