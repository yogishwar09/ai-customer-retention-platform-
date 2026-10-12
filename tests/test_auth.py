import pytest

from retentioniq.auth.passwords import generate_password, hash_password, validate_password_strength, verify_password
from retentioniq.auth.rbac import AuthError, Principal, require
from retentioniq.storage.accounts import MAX_FAILED_LOGINS, Accounts


def test_password_hash_roundtrip():
    h = hash_password("Correct-Horse-9", iterations=1000)
    assert h.startswith("pbkdf2_sha256$") and "Correct" not in h
    assert verify_password("Correct-Horse-9", h)
    assert not verify_password("wrong", h)
    assert not verify_password("x", "garbage")


def test_password_policy():
    assert validate_password_strength("short")
    assert not validate_password_strength(generate_password())


def test_authenticate_lockout_and_reset(org):
    acc = Accounts()
    uid = acc.create_user(org.org_id, "Lock@Test.io", "Good-Passw0rd!", "agent")
    p = acc.authenticate("lock@test.io", "Good-Passw0rd!")
    assert p.role == "agent" and p.org_id == org.org_id and p.actor == "lock@test.io"
    for _ in range(MAX_FAILED_LOGINS):
        with pytest.raises(AuthError, match="Invalid email or password"):
            acc.authenticate("lock@test.io", "nope")
    with pytest.raises(AuthError, match="locked"):
        acc.authenticate("lock@test.io", "Good-Passw0rd!")
    acc.update_user(org.org_id, uid, password="New-Passw0rd!!")
    assert acc.authenticate("lock@test.io", "New-Passw0rd!!").user_id == uid


def test_inactive_and_unknown_users_get_same_error(org):
    acc = Accounts()
    uid = acc.create_user(org.org_id, "gone@test.io", "Good-Passw0rd!", "viewer")
    acc.update_user(org.org_id, uid, active=False)
    for email in ("gone@test.io", "nobody@test.io"):
        with pytest.raises(AuthError, match="Invalid email or password"):
            acc.authenticate(email, "Good-Passw0rd!")


def test_user_validation(org):
    acc = Accounts()
    with pytest.raises(ValueError):
        acc.create_user(org.org_id, "not-an-email", "Good-Passw0rd!", "agent")
    with pytest.raises(ValueError):
        acc.create_user(org.org_id, "x@test.io", "weak", "agent")
    with pytest.raises(ValueError):
        acc.create_user(org.org_id, "y@test.io", "Good-Passw0rd!", "superuser")


def test_api_keys_hashed_verified_revoked(org):
    acc = Accounts()
    kid, token = acc.create_api_key(org.org_id, "ci", "manager", "admin")
    assert token.startswith("riq_")
    listed = acc.list_api_keys(org.org_id)
    assert all(token not in str(v) for row in listed for v in row.values())
    p = acc.verify_api_key(token)
    assert p.role == "manager" and p.kind == "api_key" and p.org_id == org.org_id
    with pytest.raises(AuthError):
        acc.verify_api_key(token[:-2] + "xx")
    acc.revoke_api_key(org.org_id, kid)
    with pytest.raises(AuthError):
        acc.verify_api_key(token)


def test_rbac_matrix():
    viewer = Principal("o", "O", "v", "viewer")
    admin = Principal("o", "O", "a", "admin")
    assert viewer.can("read") and not viewer.can("score")
    assert admin.can("manage_models")
    with pytest.raises(AuthError):
        require(viewer, "approve_action")
    with pytest.raises(AuthError):
        require(None, "read")
