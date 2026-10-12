"""Paths and configuration loading.

All locations can be overridden with environment variables so the same code
runs locally, in Docker, in CI and on Streamlit Cloud.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path_from_env(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


CONFIG_DIR = _path_from_env("RETENTIONIQ_CONFIG_DIR", PROJECT_ROOT / "configs")
DATA_PATH = _path_from_env(
    "RETENTIONIQ_DATA_PATH", PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
ARTIFACTS_DIR = _path_from_env("RETENTIONIQ_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts")
MODELS_DIR = ARTIFACTS_DIR / "models"
DB_PATH = _path_from_env("RETENTIONIQ_DB_PATH", ARTIFACTS_DIR / "retentioniq.db")
REPORTS_DIR = _path_from_env("RETENTIONIQ_REPORTS_DIR", PROJECT_ROOT / "reports")


@lru_cache(maxsize=16)
def _load_yaml_cached(name: str, mtime_ns: int) -> dict[str, Any]:
    with open(CONFIG_DIR / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_yaml(name: str) -> dict[str, Any]:
    """Cached per file modification time, so edits apply without a restart."""
    return _load_yaml_cached(name, (CONFIG_DIR / name).stat().st_mtime_ns)


def settings() -> dict[str, Any]:
    return _load_yaml("settings.yaml")


def playbooks() -> dict[str, Any]:
    return _load_yaml("playbooks.yaml")


def reload() -> None:
    """Drop cached config (used by tests and after admin edits)."""
    _load_yaml_cached.cache_clear()
