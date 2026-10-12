"""Organisations, users and API keys (platform-level, not tenant-scoped)."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy import Engine, text

from retentioniq.auth.passwords import hash_password, validate_password_strength, verify_password
from retentioniq.auth.rbac import ROLES, AuthError, Principal
from retentioniq.storage.engine import get_engine
from retentioniq.storage.util import new_id, now_iso

MAX_FAILED_LOGINS = 5
LOCKOUT_MINUTES = 15
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class Accounts:
    def __init__(self, engine: Engine | None = None):
        self.engine = engine or get_engine()

    # ------------------------------------------------------------ orgs
    def create_org(self, name: str, slug: str | None = None, settings: dict | None = None) -> str:
        slug = slug or re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        org_id = new_id("org")
        with self.engine.begin() as c:
            c.execute(text("INSERT INTO organizations (org_id, name, slug, settings, created_at) "
                           "VALUES (:i, :n, :s, :st, :t)"),
                      dict(i=org_id, n=name, s=slug, st=json.dumps(settings or {}), t=now_iso()))
        return org_id

    def get_org(self, org_id: str) -> dict | None:
        with self.engine.connect() as c:
            row = c.execute(text("SELECT * FROM organizations WHERE org_id = :i"), dict(i=org_id)).mappings().first()
        return dict(row) if row else None

    def org_by_slug(self, slug: str) -> dict | None:
        with self.engine.connect() as c:
            row = c.execute(text("SELECT * FROM organizations WHERE slug = :s"), dict(s=slug)).mappings().first()
        return dict(row) if row else None

    def list_orgs(self) -> list[dict]:
        with self.engine.connect() as c:
            return [dict(r) for r in c.execute(text("SELECT * FROM organizations ORDER BY created_at")).mappings()]

    # ----------------------------------------------------------- users
    def create_user(self, org_id: str, email: str, password: str, role: str,
                    display_name: str = "", *, enforce_strength: bool = True) -> str:
        email = email.strip().lower()
        if not _EMAIL.match(email):
            raise ValueError("Invalid email address")
        if role not in ROLES:
            raise ValueError(f"Unknown role {role}")
        if enforce_strength and (problems := validate_password_strength(password)):
            raise ValueError("Password needs " + ", ".join(problems))
        user_id = new_id("usr")
        with self.engine.begin() as c:
            c.execute(text(
                "INSERT INTO users (user_id, org_id, email, display_name, role, password_hash, active, "
                "failed_attempts, created_at) VALUES (:u, :o, :e, :d, :r, :p, 1, 0, :t)"),
                dict(u=user_id, o=org_id, e=email, d=display_name or email.split("@")[0], r=role,
                     p=hash_password(password), t=now_iso()))
        return user_id

    def list_users(self, org_id: str) -> list[dict]:
        with self.engine.connect() as c:
            rows = c.execute(text(
                "SELECT user_id, email, display_name, role, active, last_login_at, created_at, locked_until "
                "FROM users WHERE org_id = :o ORDER BY created_at"), dict(o=org_id)).mappings()
            return [dict(r) for r in rows]

    def update_user(self, org_id: str, user_id: str, *, role: str | None = None,
                    active: bool | None = None, password: str | None = None,
                    enforce_strength: bool = True) -> None:
        sets, params = [], dict(o=org_id, u=user_id)
        if role is not None:
            if role not in ROLES:
                raise ValueError(f"Unknown role {role}")
            sets.append("role = :r")
            params["r"] = role
        if active is not None:
            sets.append("active = :a")
            params["a"] = int(active)
        if password is not None:
            if enforce_strength and (problems := validate_password_strength(password)):
                raise ValueError("Password needs " + ", ".join(problems))
            sets += ["password_hash = :p", "failed_attempts = 0", "locked_until = NULL"]
            params["p"] = hash_password(password)
        if not sets:
            return
        with self.engine.begin() as c:
            res = c.execute(text(f"UPDATE users SET {', '.join(sets)} WHERE user_id = :u AND org_id = :o"), params)
            if res.rowcount == 0:
                raise KeyError(user_id)

    def authenticate(self, email: str, password: str) -> Principal:
        """Verify credentials with lockout. Error messages never reveal which part was wrong.
        A bare username (no "@") is treated as <username>@demo.com for the local demo accounts."""
        email = (email or "").strip().lower()
        if email and "@" not in email:
            email = f"{email}@demo.com"
        with self.engine.begin() as c:
            row = c.execute(text(
                "SELECT u.*, o.name AS org_name FROM users u JOIN organizations o USING (org_id) "
                "WHERE u.email = :e"), dict(e=email)).mappings().first()
            if row is None:
                verify_password(password, hash_password("timing-equaliser", iterations=1000))
                raise AuthError("Invalid email or password")
            now = datetime.now(timezone.utc)
            if row["locked_until"] and datetime.fromisoformat(row["locked_until"]) > now:
                raise AuthError("Account temporarily locked after repeated failed sign-ins. Try again later.")
            if not row["active"]:
                raise AuthError("Invalid email or password")
            ok = verify_password(password, row["password_hash"])
            if not ok:
                # Persist the failure before raising (raising inside the transaction would roll it back).
                failed = row["failed_attempts"] + 1
                locked = ((now + timedelta(minutes=LOCKOUT_MINUTES)).isoformat(timespec="seconds")
                          if failed >= MAX_FAILED_LOGINS else None)
                c.execute(text("UPDATE users SET failed_attempts = :f, locked_until = :l WHERE user_id = :u"),
                          dict(f=0 if locked else failed, l=locked, u=row["user_id"]))
            else:
                c.execute(text("UPDATE users SET failed_attempts = 0, locked_until = NULL, last_login_at = :t "
                               "WHERE user_id = :u"), dict(t=now_iso(), u=row["user_id"]))
        if not ok:
            raise AuthError("Invalid email or password")
        return Principal(org_id=row["org_id"], org_name=row["org_name"], actor=row["email"],
                         role=row["role"], kind="user", user_id=row["user_id"])

    # --------------------------------------------------------- API keys
    @staticmethod
    def _hash_key(secret: str) -> str:
        return hashlib.sha256(secret.encode()).hexdigest()

    def create_api_key(self, org_id: str, name: str, role: str, created_by: str) -> tuple[str, str]:
        """Returns (key_id, plaintext). The plaintext is shown once and never stored."""
        if role not in ROLES:
            raise ValueError(f"Unknown role {role}")
        prefix = secrets.token_hex(4)
        token = f"riq_{prefix}_{secrets.token_urlsafe(32)}"
        key_id = new_id("key")
        with self.engine.begin() as c:
            c.execute(text("INSERT INTO api_keys (key_id, org_id, name, role, prefix, key_hash, created_by, "
                           "created_at) VALUES (:k, :o, :n, :r, :p, :h, :c, :t)"),
                      dict(k=key_id, o=org_id, n=name, r=role, p=prefix, h=self._hash_key(token),
                           c=created_by, t=now_iso()))
        return key_id, token

    def verify_api_key(self, token: str) -> Principal:
        parts = (token or "").split("_")
        if len(parts) < 3 or parts[0] != "riq":
            raise AuthError("Invalid API key")
        prefix = parts[1]
        with self.engine.begin() as c:
            rows = c.execute(text(
                "SELECT k.*, o.name AS org_name FROM api_keys k JOIN organizations o USING (org_id) "
                "WHERE k.prefix = :p AND k.revoked_at IS NULL"), dict(p=prefix)).mappings().all()
            digest = self._hash_key(token)
            for row in rows:
                if hmac.compare_digest(row["key_hash"], digest):
                    c.execute(text("UPDATE api_keys SET last_used_at = :t WHERE key_id = :k"),
                              dict(t=now_iso(), k=row["key_id"]))
                    return Principal(org_id=row["org_id"], org_name=row["org_name"],
                                     actor=f"api-key:{row['name']}", role=row["role"], kind="api_key")
        raise AuthError("Invalid API key")

    def list_api_keys(self, org_id: str) -> list[dict]:
        with self.engine.connect() as c:
            rows = c.execute(text(
                "SELECT key_id, name, role, prefix, created_by, created_at, last_used_at, revoked_at "
                "FROM api_keys WHERE org_id = :o ORDER BY created_at DESC"), dict(o=org_id)).mappings()
            return [dict(r) for r in rows]

    def revoke_api_key(self, org_id: str, key_id: str) -> None:
        with self.engine.begin() as c:
            res = c.execute(text("UPDATE api_keys SET revoked_at = :t WHERE key_id = :k AND org_id = :o"),
                            dict(t=now_iso(), k=key_id, o=org_id))
            if res.rowcount == 0:
                raise KeyError(key_id)
