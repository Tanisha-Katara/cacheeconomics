from __future__ import annotations

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from cacheeconomics_control_plane.models import Membership, Source

from conftest import create_organization, user_headers


def provision(client, token):
    response = client.get("/v1/me", headers=user_headers(token))
    assert response.status_code == 200
    return response.json()["user"]


def force_insert_race(monkeypatch, model):
    original_flush = Session.flush
    raised = False

    def raced_flush(session, *args, **kwargs):
        nonlocal raised
        if not raised and any(isinstance(item, model) for item in session.new):
            raised = True
            raise IntegrityError("INSERT", {}, RuntimeError("simulated uniqueness race"))
        return original_flush(session, *args, **kwargs)

    monkeypatch.setattr(Session, "flush", raced_flush)


def test_first_user_creates_an_organization_as_owner(client):
    organization = create_organization(client, "alice-token", "alpha-team")
    assert organization["role"] == "owner"
    response = client.get("/v1/organizations", headers=user_headers("alice-token"))
    assert response.status_code == 200
    assert response.json() == [organization]


def test_request_body_cannot_choose_an_organization(client):
    response = client.post(
        "/v1/organizations",
        headers=user_headers("alice-token"),
        json={
            "name": "Alpha",
            "slug": "alpha-team",
            "organization_id": "00000000-0000-0000-0000-000000000000",
        },
    )
    assert response.status_code == 422


def test_source_uniqueness_race_returns_conflict(client, monkeypatch):
    organization = create_organization(client, "alice-token", "alpha-team")
    headers = user_headers("alice-token", organization["id"])
    force_insert_race(monkeypatch, Source)

    response = client.post(
        "/v1/sources",
        headers=headers,
        json={"name": "Raced source", "kind": "litellm"},
    )
    assert response.status_code == 409
    assert response.json() == {"detail": "source name already exists"}


def test_membership_uniqueness_race_returns_conflict(client, monkeypatch):
    organization = create_organization(client, "alice-token", "alpha-team")
    bob = provision(client, "bob-token")
    headers = user_headers("alice-token", organization["id"])
    force_insert_race(monkeypatch, Membership)

    response = client.post(
        "/v1/memberships",
        headers=headers,
        json={"user_id": bob["id"], "role": "viewer"},
    )
    assert response.status_code == 409
    assert response.json() == {"detail": "membership already exists"}


def test_two_organizations_cannot_read_each_others_sources(client):
    alpha = create_organization(client, "alice-token", "alpha-team")
    beta = create_organization(client, "bob-token", "beta-team")

    alpha_source = client.post(
        "/v1/sources",
        headers=user_headers("alice-token", alpha["id"]),
        json={"name": "Alpha Gateway", "kind": "litellm"},
    )
    beta_source = client.post(
        "/v1/sources",
        headers=user_headers("bob-token", beta["id"]),
        json={"name": "Beta Uploads", "kind": "file_upload"},
    )
    assert alpha_source.status_code == 201
    assert beta_source.status_code == 201

    denied = client.get(
        "/v1/sources", headers=user_headers("alice-token", beta["id"])
    )
    assert denied.status_code == 403

    visible = client.get(
        "/v1/sources", headers=user_headers("alice-token", alpha["id"])
    )
    assert [item["id"] for item in visible.json()] == [alpha_source.json()["id"]]

    hidden_source = client.get(
        f"/v1/sources/{beta_source.json()['id']}/credentials",
        headers=user_headers("alice-token", alpha["id"]),
    )
    assert hidden_source.status_code == 404


def test_viewer_analyst_admin_owner_permissions(client):
    alpha = create_organization(client, "alice-token", "alpha-team")
    bob = provision(client, "bob-token")
    carol = provision(client, "carol-token")
    owner_headers = user_headers("alice-token", alpha["id"])

    added = client.post(
        "/v1/memberships",
        headers=owner_headers,
        json={"user_id": bob["id"], "role": "analyst"},
    )
    assert added.status_code == 201

    analyst_headers = user_headers("bob-token", alpha["id"])
    assert client.get("/v1/sources", headers=analyst_headers).status_code == 200
    assert (
        client.post(
            "/v1/sources",
            headers=analyst_headers,
            json={"name": "Forbidden", "kind": "litellm"},
        ).status_code
        == 403
    )

    promoted = client.patch(
        f"/v1/memberships/{bob['id']}",
        headers=owner_headers,
        json={"role": "admin"},
    )
    assert promoted.status_code == 200
    admin_headers = user_headers("bob-token", alpha["id"])
    assert (
        client.post(
            "/v1/sources",
            headers=admin_headers,
            json={"name": "Managed", "kind": "litellm"},
        ).status_code
        == 201
    )

    added_viewer = client.post(
        "/v1/memberships",
        headers=owner_headers,
        json={"user_id": carol["id"], "role": "viewer"},
    )
    assert added_viewer.status_code == 201
    viewer_headers = user_headers("carol-token", alpha["id"])
    assert client.get("/v1/sources", headers=viewer_headers).status_code == 200
    assert (
        client.post(
            "/v1/sources",
            headers=viewer_headers,
            json={"name": "Viewer cannot create", "kind": "litellm"},
        ).status_code
        == 403
    )
    assert client.get("/v1/audit-events", headers=viewer_headers).status_code == 403

    admin_cannot_make_owner = client.patch(
        f"/v1/memberships/{carol['id']}",
        headers=admin_headers,
        json={"role": "owner"},
    )
    assert admin_cannot_make_owner.status_code == 403
    owner_can = client.patch(
        f"/v1/memberships/{carol['id']}",
        headers=owner_headers,
        json={"role": "owner"},
    )
    assert owner_can.status_code == 200


def test_last_owner_cannot_be_demoted_or_removed(client):
    alpha = create_organization(client, "alice-token", "alpha-team")
    me = provision(client, "alice-token")
    headers = user_headers("alice-token", alpha["id"])

    demote = client.patch(
        f"/v1/memberships/{me['id']}", headers=headers, json={"role": "admin"}
    )
    assert demote.status_code == 409
    remove = client.delete(f"/v1/memberships/{me['id']}", headers=headers)
    assert remove.status_code == 409


def test_role_changes_and_source_changes_are_audited(client):
    alpha = create_organization(client, "alice-token", "alpha-team")
    bob = provision(client, "bob-token")
    headers = user_headers("alice-token", alpha["id"])
    client.post(
        "/v1/memberships",
        headers=headers,
        json={"user_id": bob["id"], "role": "viewer"},
    )
    client.post(
        "/v1/sources",
        headers=headers,
        json={"name": "Gateway", "kind": "litellm"},
    )

    response = client.get("/v1/audit-events", headers=headers)
    assert response.status_code == 200
    actions = {event["action"] for event in response.json()}
    assert {"organization.created", "membership.created", "source.created"} <= actions
    assert all("token" not in str(event["details"]).lower() for event in response.json())
