from __future__ import annotations

from enum import Enum


class Role(str, Enum):
    VIEWER = "viewer"
    ANALYST = "analyst"
    ADMIN = "admin"
    OWNER = "owner"


class Permission(str, Enum):
    ORGANIZATION_READ = "organization:read"
    SOURCE_READ = "source:read"
    JOB_RUN = "job:run"
    SOURCE_MANAGE = "source:manage"
    CREDENTIAL_MANAGE = "credential:manage"
    MEMBER_MANAGE = "member:manage"
    ORGANIZATION_DELETE = "organization:delete"


ROLE_PERMISSIONS = {
    Role.VIEWER: {
        Permission.ORGANIZATION_READ,
        Permission.SOURCE_READ,
    },
    Role.ANALYST: {
        Permission.ORGANIZATION_READ,
        Permission.SOURCE_READ,
        Permission.JOB_RUN,
    },
    Role.ADMIN: {
        Permission.ORGANIZATION_READ,
        Permission.SOURCE_READ,
        Permission.JOB_RUN,
        Permission.SOURCE_MANAGE,
        Permission.CREDENTIAL_MANAGE,
        Permission.MEMBER_MANAGE,
    },
    Role.OWNER: set(Permission),
}


def has_permission(role: Role | str, permission: Permission) -> bool:
    try:
        normalized = role if isinstance(role, Role) else Role(role)
    except ValueError:
        return False
    return permission in ROLE_PERMISSIONS[normalized]


def can_assign_role(actor: Role | str, requested: Role | str) -> bool:
    """Admins can manage non-owners; only owners can create another owner."""

    try:
        actor_role = actor if isinstance(actor, Role) else Role(actor)
        requested_role = requested if isinstance(requested, Role) else Role(requested)
    except ValueError:
        return False
    if actor_role is Role.OWNER:
        return True
    return actor_role is Role.ADMIN and requested_role is not Role.OWNER
