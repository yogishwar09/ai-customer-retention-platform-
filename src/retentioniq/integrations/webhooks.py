"""Outbound webhooks to CRM / helpdesk systems.

Delivery guarantees: at-least-once via a transactional outbox. Receivers
de-duplicate on the ``Idempotency-Key`` header (the event id). Each request is
signed: ``X-RetentionIQ-Signature: t=<unix>,v1=<hex HMAC-SHA256(secret, "t.body")>``.
Failed deliveries retry with exponential backoff; after ``MAX_ATTEMPTS`` they are
moved to the dead-letter state and can be replayed by an admin.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import socket
import time
from typing import Any
from urllib.parse import urlparse

import pandas as pd
from sqlalchemy import text

from retentioniq.storage.repository import EVENT_TYPES, Repository
from retentioniq.storage.util import new_id, now_iso

MAX_ATTEMPTS = 6
BASE_BACKOFF_SECONDS = 30
TIMEOUT_SECONDS = 5
SANDBOX_SCHEME = "sandbox"


class WebhookError(ValueError):
    pass


# --------------------------------------------------------------- security
def sign(secret: str, body: bytes, timestamp: int | None = None) -> str:
    ts = int(timestamp if timestamp is not None else time.time())
    mac = hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    return f"t={ts},v1={mac}"


def verify_signature(secret: str, body: bytes, header: str, tolerance_seconds: int = 300) -> bool:
    try:
        parts = dict(p.split("=", 1) for p in header.split(","))
        ts = int(parts["t"])
    except (ValueError, KeyError):
        return False
    if abs(time.time() - ts) > tolerance_seconds:
        return False
    expected = hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, parts.get("v1", ""))


def validate_url(url: str) -> None:
    """Block SSRF: only http(s) to public addresses (or sandbox://), unless private
    targets are explicitly allowed for local development."""
    parsed = urlparse(url)
    if parsed.scheme == SANDBOX_SCHEME:
        return
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise WebhookError("Webhook URL must be http(s):// or sandbox://")
    if os.environ.get("RETENTIONIQ_ALLOW_PRIVATE_WEBHOOKS") == "1":
        return
    if parsed.scheme != "https":
        raise WebhookError("Webhook URL must use https")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or 443)
    except socket.gaierror as exc:
        raise WebhookError(f"Cannot resolve {parsed.hostname}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise WebhookError("Webhook URL resolves to a private or reserved address")


# ------------------------------------------------------------- management
class Webhooks:
    def __init__(self, repo: Repository):
        self.repo = repo

    def create(self, name: str, url: str, event_types: list[str], actor: str) -> tuple[str, str]:
        bad = [e for e in event_types if e not in EVENT_TYPES]
        if bad or not event_types:
            raise WebhookError(f"Unknown or empty event types: {bad}")
        validate_url(url)
        webhook_id, secret = new_id("whk"), "whsec_" + secrets.token_urlsafe(24)
        self.repo.execute(
            "INSERT INTO webhooks (webhook_id, org_id, name, url, secret, event_types, active, created_by, "
            "created_at) VALUES (:i, :org, :n, :u, :s, :e, 1, :a, :t)",
            dict(i=webhook_id, n=name, u=url, s=secret, e=json.dumps(sorted(event_types)), a=actor, t=now_iso()))
        self.repo.audit(actor, "webhook.created", "webhook", webhook_id, {"url": url, "events": event_types})
        return webhook_id, secret

    def list(self) -> pd.DataFrame:
        df = self.repo.query_df("SELECT webhook_id, name, url, event_types, active, created_by, created_at "
                                "FROM webhooks WHERE org_id = :org ORDER BY created_at DESC")
        if not df.empty:
            df["event_types"] = df["event_types"].map(lambda s: ", ".join(json.loads(s)))
        return df

    def set_active(self, webhook_id: str, active: bool, actor: str) -> None:
        if self.repo.execute("UPDATE webhooks SET active = :a WHERE webhook_id = :i AND org_id = :org",
                             dict(a=int(active), i=webhook_id)) == 0:
            raise KeyError(webhook_id)
        self.repo.audit(actor, "webhook.enabled" if active else "webhook.disabled", "webhook", webhook_id)

    def deliveries(self, limit: int = 200) -> pd.DataFrame:
        return self.repo.query_df(
            "SELECT d.delivery_id, d.status, d.attempts, d.last_status_code, d.last_error, d.next_attempt_at, "
            "d.delivered_at, e.event_type, e.event_id, w.name AS webhook FROM webhook_deliveries d "
            "JOIN outbox_events e ON e.event_id = d.event_id JOIN webhooks w ON w.webhook_id = d.webhook_id "
            "WHERE d.org_id = :org ORDER BY e.created_at DESC LIMIT :n", dict(n=limit))

    def replay(self, delivery_id: str, actor: str) -> None:
        n = self.repo.execute("UPDATE webhook_deliveries SET status = 'pending', next_attempt_at = :t, attempts = 0 "
                              "WHERE delivery_id = :d AND org_id = :org AND status = 'dead'",
                              dict(t=now_iso(), d=delivery_id))
        if n == 0:
            raise KeyError(delivery_id)
        self.repo.audit(actor, "webhook.replayed", "webhook_delivery", delivery_id)

    def send_test(self, actor: str) -> str | None:
        return self.repo.emit("alert.raised", {"test": True, "message": "Test event from RetentionIQ",
                                               "sent_by": actor})

    def sandbox_inbox(self) -> pd.DataFrame:
        return self.repo.query_df("SELECT * FROM sandbox_inbox WHERE org_id = :org ORDER BY received_at DESC")


# --------------------------------------------------------------- delivery
def _sandbox_receive(engine, org_id: str, secret: str, body: bytes, headers: dict[str, str]) -> int:
    """In-process mock CRM: verifies the signature and de-duplicates on the
    idempotency key, exactly like a real receiver should. Set
    RETENTIONIQ_SANDBOX_FAILURE_RATE (0-1) to rehearse retries."""
    rate = float(os.environ.get("RETENTIONIQ_SANDBOX_FAILURE_RATE", "0"))
    if rate and secrets.randbelow(1000) < rate * 1000:
        return 503
    valid = verify_signature(secret, body, headers["X-RetentionIQ-Signature"])
    if not valid:
        return 401
    payload = json.loads(body)
    with engine.begin() as c:
        c.execute(text(
            "INSERT INTO sandbox_inbox (idempotency_key, org_id, event_type, signature_valid, payload, "
            "delivery_count, received_at) VALUES (:k, :o, :e, 1, :p, 1, :t) ON CONFLICT (idempotency_key) "
            "DO UPDATE SET delivery_count = sandbox_inbox.delivery_count + 1"),
            dict(k=headers["Idempotency-Key"], o=org_id, e=payload["type"], p=body.decode(), t=now_iso()))
    return 200


def dispatch_due(engine, limit: int = 100, now: str | None = None, org_id: str | None = None) -> dict[str, int]:
    """Deliver pending webhook deliveries whose next attempt is due (all tenants, or one)."""
    import httpx

    now = now or now_iso()
    stats = {"attempted": 0, "succeeded": 0, "retrying": 0, "dead": 0}
    with engine.connect() as c:
        due = c.execute(text(
            "SELECT d.delivery_id, d.org_id, d.attempts, e.event_id, e.event_type, e.payload, w.url, w.secret "
            "FROM webhook_deliveries d JOIN outbox_events e ON e.event_id = d.event_id "
            "JOIN webhooks w ON w.webhook_id = d.webhook_id "
            "WHERE d.status = 'pending' AND d.next_attempt_at <= :now AND w.active = 1 "
            + ("AND d.org_id = :org " if org_id else "") +
            "ORDER BY d.next_attempt_at LIMIT :n"), dict(now=now, n=limit, org=org_id)).mappings().all()
    for d in due:
        body = d["payload"].encode()
        headers = {"Content-Type": "application/json", "User-Agent": "RetentionIQ-Webhooks/2",
                   "X-RetentionIQ-Event": d["event_type"], "X-RetentionIQ-Delivery": d["delivery_id"],
                   "Idempotency-Key": d["event_id"], "X-RetentionIQ-Signature": sign(d["secret"], body)}
        stats["attempted"] += 1
        status_code, error = None, None
        try:
            if d["url"].startswith(f"{SANDBOX_SCHEME}://"):
                status_code = _sandbox_receive(engine, d["org_id"], d["secret"], body, headers)
            else:
                validate_url(d["url"])
                resp = httpx.post(d["url"], content=body, headers=headers, timeout=TIMEOUT_SECONDS,
                                  follow_redirects=False)
                status_code = resp.status_code
        except Exception as exc:  # network errors, SSRF block, timeouts
            error = f"{type(exc).__name__}: {exc}"[:500]
        attempts = d["attempts"] + 1
        ok = status_code is not None and 200 <= status_code < 300
        if ok:
            new_status, next_at = "succeeded", now
            stats["succeeded"] += 1
        elif attempts >= MAX_ATTEMPTS or (status_code is not None and status_code in (400, 401, 403, 404, 410)):
            new_status, next_at = "dead", now
            stats["dead"] += 1
        else:
            new_status = "pending"
            next_at = now_iso(BASE_BACKOFF_SECONDS * 2 ** (attempts - 1))
            stats["retrying"] += 1
        with engine.begin() as c:
            c.execute(text(
                "UPDATE webhook_deliveries SET status = CAST(:s AS VARCHAR), attempts = :a, next_attempt_at = :n, "
                "last_status_code = :code, last_error = :err, delivered_at = CASE WHEN CAST(:s AS VARCHAR) = "
                "'succeeded' THEN CAST(:now AS VARCHAR) ELSE delivered_at END WHERE delivery_id = :d"),
                dict(s=new_status, a=attempts, n=next_at, code=status_code,
                     err=error or (None if ok else f"HTTP {status_code}"), now=now, d=d["delivery_id"]))
    return stats


def summary(repo: Repository) -> dict[str, Any]:
    df = repo.query_df("SELECT status, COUNT(*) AS n FROM webhook_deliveries WHERE org_id = :org GROUP BY status")
    return {r.status: int(r.n) for r in df.itertuples()}
