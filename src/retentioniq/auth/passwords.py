"""Password hashing with PBKDF2-HMAC-SHA256 (stdlib only, OWASP-recommended iterations)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets

ITERATIONS = 310_000
MIN_LENGTH = 10


def hash_password(password: str, *, iterations: int = ITERATIONS) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algo, iterations, salt_b64, digest_b64 = encoded.split("$")
    except ValueError:
        return False
    if algo != "pbkdf2_sha256":
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), base64.b64decode(salt_b64), int(iterations))
    return hmac.compare_digest(digest, base64.b64decode(digest_b64))


def validate_password_strength(password: str) -> list[str]:
    problems = []
    if len(password) < MIN_LENGTH:
        problems.append(f"at least {MIN_LENGTH} characters")
    if password.lower() == password or password.upper() == password:
        problems.append("mixed upper and lower case")
    if not any(ch.isdigit() for ch in password):
        problems.append("at least one digit")
    return problems


def generate_password(length: int = 16) -> str:
    alphabet = "abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    while True:
        pw = "".join(secrets.choice(alphabet) for _ in range(length))
        if not validate_password_strength(pw):
            return pw
