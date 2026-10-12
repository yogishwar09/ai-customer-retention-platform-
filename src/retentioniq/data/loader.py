"""Load the reference training dataset through the data contract."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from retentioniq import config
from retentioniq.data.schema import TARGET_COLUMN, ValidationResult, validate_frame


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or config.DATA_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def load_training_data(path: Path | None = None) -> tuple[pd.DataFrame, ValidationResult]:
    """Return (validated labelled frame, validation result)."""
    raw = load_raw(path)
    result = validate_frame(raw, require_ids=True)
    labelled = result.valid.dropna(subset=[TARGET_COLUMN]).copy()
    labelled[TARGET_COLUMN] = labelled[TARGET_COLUMN].astype(int)
    return labelled, result
