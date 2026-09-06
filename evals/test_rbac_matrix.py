from __future__ import annotations

import pytest

from cacheeconomics_control_plane.rbac import (
    Permission,
    Role,
    can_assign_role,
    has_permission,
)


EXPECTED = {
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


@pytest.mark.parametrize("role", list(Role))
@pytest.mark.parametrize("permission", list(Permission))
def test_complete_permission_matrix(role, permission):
    assert has_permission(role, permission) is (permission in EXPECTED[role])


@pytest.mark.parametrize("actor", list(Role))
@pytest.mark.parametrize("requested", list(Role))
def test_complete_role_assignment_matrix(actor, requested):
    expected = actor is Role.OWNER or (
        actor is Role.ADMIN and requested is not Role.OWNER
    )
    assert can_assign_role(actor, requested) is expected


def test_unknown_roles_fail_closed():
    assert not has_permission("super-admin", Permission.SOURCE_READ)
    assert not can_assign_role("super-admin", Role.VIEWER)
