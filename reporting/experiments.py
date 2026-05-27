import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def create_experiment_directory(report_root: str, mode_name: str) -> tuple[Path, str]:
    """Create a timestamped directory for one reproducible experiment run."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = Path(report_root) / mode_name / timestamp
    path.mkdir(parents=True, exist_ok=True)
    return path, timestamp


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    return value


def save_metadata(
    report_directory: Path,
    timestamp: str,
    mode: Any,
    universe: dict,
    execution_assumptions: dict,
    config_snapshot: dict,
) -> Path:
    """Persist experiment identity, universe, execution, and effective config."""
    metadata = {
        "experiment_timestamp_utc": timestamp,
        "mode": _json_ready(mode),
        "universe": _json_ready(universe),
        "execution_assumptions": _json_ready(execution_assumptions),
        "config_snapshot": _json_ready(config_snapshot),
    }
    output = report_directory / "experiment_metadata.json"
    output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


def save_tables(report_directory: Path, tables: dict[str, pd.DataFrame | pd.Series]) -> list[Path]:
    """Save named result tables with consistent CSV filenames."""
    outputs = []
    for name, table in tables.items():
        output = report_directory / f"{name}.csv"
        if isinstance(table, pd.Series):
            table.to_frame().to_csv(output)
        elif isinstance(table.index, pd.DatetimeIndex):
            table.to_csv(output, index_label="date")
        else:
            table.to_csv(output, index=False)
        outputs.append(output)
    return outputs


def save_figures(report_directory: Path, figures: dict[str, Any]) -> list[Path]:
    """Save named matplotlib figures with consistent PNG filenames."""
    outputs = []
    for name, figure in figures.items():
        safe_name = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")
        output = report_directory / f"{safe_name}.png"
        figure.savefig(output, dpi=150, bbox_inches="tight")
        outputs.append(output)
    return outputs
