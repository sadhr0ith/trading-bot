"""Lightweight experiment tracking helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def dataset_hash(data: pd.DataFrame) -> str | None:
    """Compute a stable hash for a DataFrame."""
    if data is None or data.empty:
        return None
    hashed = pd.util.hash_pandas_object(data, index=True)
    digest = hashlib.sha256(hashed.values.tobytes()).hexdigest()
    return digest


def write_experiment_report(report: dict[str, Any], output_dir: str | Path, prefix: str = "train") -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    signature = report.get("config_signature", "config")
    filename = f"{prefix}_{report.get('strategy', 'strategy')}_{signature}_{timestamp}.json"
    full_path = path / filename
    with full_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)
    return full_path
