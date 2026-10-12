"""Object storage abstraction. Local filesystem now; an S3/GCS backend implements
the same three methods."""

from __future__ import annotations

import re
from pathlib import Path

from retentioniq import config

_SAFE = re.compile(r"[^A-Za-z0-9._/-]")


class LocalBlobStore:
    scheme = "local"

    def __init__(self, root: Path | None = None):
        self.root = (root or config.ARTIFACTS_DIR / "blobs").resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        key = _SAFE.sub("_", key).lstrip("/")
        path = (self.root / key).resolve()
        if not str(path).startswith(str(self.root)):  # path-traversal guard
            raise ValueError("Invalid blob key")
        return path

    def put(self, key: str, data: bytes) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return f"{self.scheme}://{path.relative_to(self.root)}"

    def get(self, uri: str) -> bytes:
        if not uri.startswith(f"{self.scheme}://"):
            raise ValueError(f"Unsupported blob uri {uri}")
        return self._path(uri.split("://", 1)[1]).read_bytes()

    def delete(self, uri: str) -> None:
        self._path(uri.split("://", 1)[1]).unlink(missing_ok=True)


def store() -> LocalBlobStore:
    return LocalBlobStore()
