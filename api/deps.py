"""Shared API dependencies: authentication, RBAC, tenant repository, rate limiting."""

import threading
import time
from collections import defaultdict
from functools import lru_cache

from fastapi import Depends, Header, HTTPException, Request, status

from retentioniq import config
from retentioniq.auth.rbac import AuthError, Principal
from retentioniq.models import registry
from retentioniq.service import PredictionService
from retentioniq.storage.accounts import Accounts
from retentioniq.storage.repository import Repository

_service_lock = threading.Lock()
_service_cache: dict[str, PredictionService] = {}


def get_service() -> PredictionService:
    """One service per (production, staging) pair; reloads automatically after promotion."""
    key = f"{registry.production_version()}|{registry.staging_version()}"
    with _service_lock:
        if key not in _service_cache:
            _service_cache.clear()
            _service_cache[key] = PredictionService()
        return _service_cache[key]


@lru_cache(maxsize=1)
def get_accounts() -> Accounts:
    return Accounts()


class RateLimiter:
    """Token bucket per principal (in-process; use Redis for multi-replica deployments)."""

    def __init__(self):
        self._buckets: dict[str, tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))
        self._lock = threading.Lock()

    def allow(self, key: str, per_minute: int) -> tuple[bool, float]:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (float(per_minute), now))
            tokens = min(float(per_minute), tokens + (now - last) * per_minute / 60.0)
            if tokens < 1:
                self._buckets[key] = (tokens, now)
                return False, (1 - tokens) * 60.0 / per_minute
            self._buckets[key] = (tokens - 1, now)
            return True, 0.0


rate_limiter = RateLimiter()


def get_principal(request: Request, authorization: str | None = Header(default=None),
                  x_api_key: str | None = Header(default=None)) -> Principal:
    token = x_api_key
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing API key (Authorization: Bearer riq_...)",
                            headers={"WWW-Authenticate": "Bearer"})
    try:
        principal = get_accounts().verify_api_key(token)
    except AuthError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key",
                            headers={"WWW-Authenticate": "Bearer"}) from None
    limit = int(config.settings().get("api", {}).get("rate_limit_per_minute", 120))
    allowed, retry = rate_limiter.allow(f"{principal.org_id}:{principal.actor}", limit)
    if not allowed:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Rate limit exceeded",
                            headers={"Retry-After": str(int(retry) + 1)})
    request.state.principal = principal
    return principal


def require(permission: str):
    def dep(principal: Principal = Depends(get_principal)) -> Principal:
        if not principal.can(permission):
            raise HTTPException(status.HTTP_403_FORBIDDEN,
                                f"Role '{principal.role}' cannot {permission.replace('_', ' ')}")
        return principal
    return dep


def repo_for(principal: Principal) -> Repository:
    return Repository(principal.org_id)
