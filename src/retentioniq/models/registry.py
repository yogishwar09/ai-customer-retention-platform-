"""Versioned model registry on the local filesystem.

Layout::

    artifacts/models/<version>/model.joblib
    artifacts/models/<version>/metadata.json
    artifacts/models/<version>/evaluation.json
    artifacts/models/registry.json      # status of every version

Only artifacts produced by this project's training pipeline are loaded, and the
SHA-256 recorded at training time is verified before deserialisation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from retentioniq import config


class ModelNotAvailable(RuntimeError):
    pass


class ArtifactIntegrityError(RuntimeError):
    pass


@dataclass
class LoadedModel:
    version: str
    pipeline: Any
    metadata: dict[str, Any]
    evaluation: dict[str, Any]

    @property
    def threshold(self) -> float:
        return float(self.metadata["operating_threshold"])

    @property
    def algorithm(self) -> str:
        return self.metadata["algorithm"]


def _registry_path() -> Path:
    return config.MODELS_DIR / "registry.json"


def read_registry() -> dict[str, Any]:
    path = _registry_path()
    if not path.exists():
        return {"production": None, "models": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_registry(reg: dict[str, Any]) -> None:
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


STATUSES = ("candidate", "staging", "production", "archived", "rejected")


class PromotionError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def register(version: str, status: str, summary: dict[str, Any], actor: str = "training-pipeline") -> None:
    reg = read_registry()
    reg["models"] = [m for m in reg["models"] if m["version"] != version]
    entry = {"version": version, "status": status, "registered_at": _now(),
             "history": [{"status": status, "by": actor, "at": _now(), "reason": "registered"}], **summary}
    reg["models"].append(entry)
    if status == "production":
        for m in reg["models"]:
            if m["version"] != version and m["status"] == "production":
                m["status"] = "archived"
        reg["production"] = version
    _write_registry(reg)


def set_status(version: str, status: str, actor: str, reason: str = "") -> None:
    """Governed lifecycle transition.

    candidate -> staging -> production; any -> archived. Only gate-passing models
    can reach staging/production; production requires a stated reason (approval).
    Promoting an archived model is the rollback path.
    """
    if status not in STATUSES:
        raise PromotionError(f"Unknown status {status}")
    reg = read_registry()
    entry = next((m for m in reg["models"] if m["version"] == version), None)
    if entry is None:
        raise ModelNotAvailable(f"Unknown model version {version}")
    if status in ("staging", "production"):
        if not entry.get("gate_passed"):
            raise PromotionError("Model did not pass the release gate")
        if not (config.MODELS_DIR / version / "model.joblib").exists():
            raise ModelNotAvailable(f"Artifact for {version} is missing")
    if status == "production" and not reason.strip():
        raise PromotionError("Promotion to production requires an approval reason")
    if status == "staging":
        for m in reg["models"]:
            if m["status"] == "staging" and m["version"] != version:
                m["status"] = "candidate"
                m.setdefault("history", []).append({"status": "candidate", "by": actor, "at": _now(),
                                                    "reason": f"replaced in staging by {version}"})
    if status == "production":
        for m in reg["models"]:
            if m["status"] == "production" and m["version"] != version:
                m["status"] = "archived"
                m.setdefault("history", []).append({"status": "archived", "by": actor, "at": _now(),
                                                    "reason": f"superseded by {version}"})
        reg["production"] = version
        entry["promoted_at"] = _now()
    elif reg.get("production") == version:
        raise PromotionError("Promote another model before demoting the production model")
    entry["status"] = status
    entry.setdefault("history", []).append({"status": status, "by": actor, "at": _now(), "reason": reason})
    _write_registry(reg)
    if status == "production":
        from retentioniq.reporting import publish
        publish(version)


def promote(version: str, actor: str = "admin", reason: str = "manual promotion") -> None:
    set_status(version, "production", actor, reason)


def production_version() -> str | None:
    return read_registry().get("production")


def staging_version() -> str | None:
    return next((m["version"] for m in read_registry()["models"] if m["status"] == "staging"), None)


def load(version: str | None = None) -> LoadedModel:
    version = version or production_version()
    if not version:
        raise ModelNotAvailable("No production model registered. Run `make train`.")
    folder = config.MODELS_DIR / version
    model_path = folder / "model.joblib"
    meta_path = folder / "metadata.json"
    if not model_path.exists() or not meta_path.exists():
        raise ModelNotAvailable(f"Artifacts for {version} not found in {folder}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = metadata.get("artifact_sha256")
    if not expected or sha256(model_path) != expected:
        raise ArtifactIntegrityError(
            f"Checksum mismatch for {model_path}; refusing to load an untrusted artifact.")
    pipeline = joblib.load(model_path)
    eval_path = folder / "evaluation.json"
    evaluation = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}
    return LoadedModel(version, pipeline, metadata, evaluation)
