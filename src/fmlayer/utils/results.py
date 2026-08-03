import csv
import os
from datetime import datetime, timezone
from pathlib import Path

from src.fmlayer.data.specs import REPO_ROOT

RUNS_FILENAME = "runs.csv"
RUNS_COLUMNS = (
    "method",
    "dataset",
    "encoder",
    "k",
    "seed",
    "t",
    "split",
    "accuracy",
    "num_items",
    "timestamp",
)
# A run is uniquely identified by these fields; re-running replaces the old row.
# ``t`` is the flow-matching integration time; methods without one record NO_VALUE.
IDENTITY_COLUMNS = ("method", "dataset", "encoder", "k", "seed", "t", "split")
NO_VALUE = "none"


def normalize_row(row: dict) -> dict:
    """Fill in any column a CSV written by an earlier schema predates.

    Args:
        row: A row read from ``runs.csv``.

    Returns:
        The row with every column of :data:`RUNS_COLUMNS` present.
    """
    return {column: row.get(column, NO_VALUE) for column in RUNS_COLUMNS}


def default_results_root() -> Path:
    """Resolve the results directory: ``FMLAYER_RESULTS_ROOT``, then Colab, then the repo.

    Returns:
        Directory holding ``runs.csv``, figures and checkpoints.
    """
    env = os.environ.get("FMLAYER_RESULTS_ROOT")
    if env:
        return Path(env)
    if Path("/content").is_dir():
        return Path("/content/results")
    return REPO_ROOT / "results"


def runs_csv_path(results_root: Path | None = None) -> Path:
    """Locate the tidy results table.

    Args:
        results_root: Results directory; defaults to :func:`default_results_root`.

    Returns:
        Path of ``runs.csv``.
    """
    root = Path(results_root) if results_root is not None else default_results_root()
    return root / RUNS_FILENAME


def load_runs(results_root: Path | None = None) -> list[dict]:
    """Read every recorded run.

    Args:
        results_root: Results directory; defaults to :func:`default_results_root`.

    Returns:
        One dict per row, or an empty list when nothing has been recorded yet.
    """
    path = runs_csv_path(results_root)
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [normalize_row(row) for row in csv.DictReader(handle)]


def record_run(row: dict, results_root: Path | None = None) -> Path:
    """Append a result row, replacing any earlier row describing the same run.

    Args:
        row: Values for :data:`RUNS_COLUMNS`; ``timestamp`` is filled in automatically.
        results_root: Results directory; defaults to :func:`default_results_root`.

    Returns:
        Path of the rewritten ``runs.csv``.
    """
    entry = {column: NO_VALUE for column in RUNS_COLUMNS}
    entry.update({key: value for key, value in row.items() if key in RUNS_COLUMNS})
    entry["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    identity = tuple(str(entry[column]) for column in IDENTITY_COLUMNS)
    kept = [
        existing
        for existing in load_runs(results_root)
        if tuple(str(existing[column]) for column in IDENTITY_COLUMNS) != identity
    ]
    kept.append(entry)

    path = runs_csv_path(results_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUNS_COLUMNS)
        writer.writeheader()
        writer.writerows(kept)
    return path
