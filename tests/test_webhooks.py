import time

import pytest

from retentioniq.integrations.webhooks import (
    MAX_ATTEMPTS,
    WebhookError,
    Webhooks,
    dispatch_due,
    sign,
    validate_url,
    verify_signature,
)


def test_signature_roundtrip_and_tamper():
    body = b'{"a":1}'
    header = sign("secret", body)
    assert verify_signature("secret", body, header)
    assert not verify_signature("secret", body + b" ", header)
    assert not verify_signature("other", body, header)
    stale = sign("secret", body, timestamp=int(time.time()) - 3600)
    assert not verify_signature("secret", body, stale)


@pytest.mark.parametrize("url", ["ftp://x.com/h", "http://example.com/h", "https://127.0.0.1/h",
                                 "https://10.0.0.5/h", "https://169.254.169.254/latest/meta-data", "notaurl"])
def test_ssrf_guard_blocks(url, monkeypatch):
    monkeypatch.delenv("RETENTIONIQ_ALLOW_PRIVATE_WEBHOOKS", raising=False)
    with pytest.raises(WebhookError):
        validate_url(url)


def test_sandbox_delivery_signed_and_idempotent(org):
    hooks = Webhooks(org)
    hooks.create("crm", "sandbox://crm", ["alert.raised"], "admin")
    ev = org.emit("alert.raised", {"msg": "hi"})
    stats = dispatch_due(org.engine, org_id=org.org_id)
    assert stats["succeeded"] >= 1
    inbox = hooks.sandbox_inbox()
    assert list(inbox["idempotency_key"]) == [ev] and inbox["signature_valid"].iloc[0] == 1
    # at-least-once redelivery is absorbed by the receiver's idempotency key
    org.execute("UPDATE webhook_deliveries SET status = 'pending' WHERE org_id = :org")
    dispatch_due(org.engine, org_id=org.org_id)
    assert hooks.sandbox_inbox()["delivery_count"].iloc[0] == 2


def test_failed_deliveries_back_off_then_dead_letter(org, monkeypatch):
    monkeypatch.setenv("RETENTIONIQ_SANDBOX_FAILURE_RATE", "1")
    hooks = Webhooks(org)
    hooks.create("crm", "sandbox://crm", ["alert.raised"], "admin")
    org.emit("alert.raised", {"msg": "x"})
    for attempt in range(1, MAX_ATTEMPTS + 1):
        org.execute("UPDATE webhook_deliveries SET next_attempt_at = '2000-01-01T00:00:00+00:00' "
                    "WHERE org_id = :org AND status = 'pending'")
        dispatch_due(org.engine, org_id=org.org_id)
        d = hooks.deliveries().iloc[0]
        assert d["attempts"] == attempt and d["last_status_code"] == 503
    assert d["status"] == "dead"
    monkeypatch.setenv("RETENTIONIQ_SANDBOX_FAILURE_RATE", "0")
    hooks.replay(d["delivery_id"], "admin")
    dispatch_due(org.engine, org_id=org.org_id)
    assert hooks.deliveries().iloc[0]["status"] == "succeeded"


def test_webhook_validation(org):
    with pytest.raises(WebhookError):
        Webhooks(org).create("x", "sandbox://crm", ["not.an.event"], "a")
    wid, secret = Webhooks(org).create("x", "sandbox://crm", ["action.approved"], "a")
    assert secret.startswith("whsec_")
    assert "secret" not in Webhooks(org).list().columns  # never listed back
