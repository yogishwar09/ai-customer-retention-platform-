"""Roles, permissions and the authenticated principal.

Authorisation is checked server-side (repository/API dependencies), so hiding a
button in the UI is a convenience, never the control.
"""

from __future__ import annotations

from dataclasses import dataclass

ROLES = ("viewer", "agent", "manager", "admin")
_RANK = {r: i for i, r in enumerate(ROLES)}

# permission -> minimum role
PERMISSIONS = {
    "read": "viewer",
    "score": "agent",
    "persist_scores": "agent",
    "log_action": "agent",
    "record_outcome": "agent",
    "approve_action": "manager",
    "export": "manager",
    "import_labels": "manager",
    "manage_experiments": "manager",
    "use_assistant": "viewer",
    "manage_integrations": "admin",
    "manage_users": "admin",
    "manage_models": "admin",
    "reset_demo": "admin",
}


class AuthError(PermissionError):
    pass


@dataclass(frozen=True)
class Principal:
    org_id: str
    org_name: str
    actor: str          # user email or "api-key:<name>"
    role: str
    kind: str = "user"  # user | api_key | system
    user_id: str | None = None

    def can(self, permission: str) -> bool:
        return _RANK[self.role] >= _RANK[PERMISSIONS[permission]]


def require(principal: Principal | None, permission: str) -> Principal:
    if principal is None:
        raise AuthError("Authentication required")
    if not principal.can(permission):
        raise AuthError(f"Role '{principal.role}' is not allowed to {permission.replace('_', ' ')}")
    return principal


def system_principal(org_id: str, org_name: str = "system") -> Principal:
    return Principal(org_id=org_id, org_name=org_name, actor="system", role="admin", kind="system")
