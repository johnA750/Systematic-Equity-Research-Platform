# Contributing

This repository is primarily a portfolio research project, but issues and small improvements are welcome.

## Development Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Contribution Guidelines

- Keep production strategy logic stable unless the change fixes a clear bug.
- Prefer adding research or reporting behavior behind explicit configuration flags.
- Do not commit generated `reports/` output or private point-in-time constituent datasets.
- Include concise validation notes for changes that affect return paths, execution costs, or report exports.
- Document data-source assumptions when adding universe, factor, or market-data functionality.

## Validation Before Submitting

At minimum, run:

```powershell
python -m py_compile run_backtest.py research\run_walk_forward.py
python run_production_baseline.py
```

For broader changes, also run:

```powershell
python research\run_sp500_research.py
python research\run_walk_forward.py
```
