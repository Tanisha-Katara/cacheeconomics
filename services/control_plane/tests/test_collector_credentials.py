from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from cacheeconomics_control_plane.models import SourceCredential
from cacheeconomics_control_plane.security import new_collector_credential

from conftest import create_organization, user_headers


def create_source_and_credential(client):
    organization = create_organization(client, "alice-token", "alpha-team")
    headers = user_headers("alice-token", organization["id"])
    source = client.post(
        "/v1/sources",
        headers=headers,
        json={"name": "Gateway", "kind": "litellm"},
    ).json()
    response = client.post(
        f"/v1/sources/{source['id']}/credentials",
        headers=headers,
        json={"name": "Production collector"},
    )
    assert response.status_code == 201, response.text
    return organization, source, response.json(), headers


def test_generated_credentials_are_random_and_only_store_a_digest():
    one = new_collector_credential()
    two = new_collector_credential()
    assert one.token.startswith("cec_")
    assert one.token != two.token
    assert one.prefix != two.prefix
    assert one.token not in one.digest
    assert one.digest == hashlib.sha256(one.token.encode()).hexdigest()


def test_naive_credential_expiry_is_persisted_as_utc(client):
    organization = create_organization(client, "alice-token", "alpha-team")
    headers = user_headers("alice-token", organization["id"])
    source = client.post(
        "/v1/sources",
        headers=headers,
        json={"name": "Gateway", "kind": "litellm"},
    ).json()
    naive_expiry = (datetime.now(timezone.utc) + timedelta(hours=1)).replace(
        microsecond=0,
        tzinfo=None,
    )

    response = client.post(
        f"/v1/sources/{source['id']}/credentials",
        headers=headers,
        json={"name": "Expiring collector", "expires_at": naive_expiry.isoformat()},
    )
    assert response.status_code == 201, response.text
    returned = datetime.fromisoformat(response.json()["expires_at"].replace("Z", "+00:00"))
    assert returned.utcoffset() == timedelta(0)
    assert returned.replace(tzinfo=None) == naive_expiry


def test_api_shows_secret_once_and_collector_can_authenticate(client, engine):
    organization, source, created, headers = create_source_and_credential(client)
    token = created["token"]
    assert created["warning"].startswith("Save this token now")

    listed = client.get(
        f"/v1/sources/{source['id']}/credentials", headers=headers
    )
    assert listed.status_code == 200
    assert "token" not in listed.json()[0]

    with Session(engine) as session:
        stored = session.scalar(select(SourceCredential))
        assert stored.credential_digest == hashlib.sha256(token.encode()).hexdigest()
        assert token not in stored.credential_digest
        assert stored.credential_prefix == created["prefix"]

    identity = client.get(
        "/v1/collector/whoami", headers={"Authorization": f"Bearer {token}"}
    )
    assert identity.status_code == 200
    assert identity.json()["organization_id"] == organization["id"]
    assert identity.json()["source_id"] == source["id"]


def test_revoked_or_malformed_collector_credential_is_rejected(client):
    _, source, created, headers = create_source_and_credential(client)
    token = created["token"]
    revoke = client.delete(
        f"/v1/sources/{source['id']}/credentials/{created['id']}",
        headers=headers,
    )
    assert revoke.status_code == 204
    assert (
        client.get(
            "/v1/collector/whoami",
            headers={"Authorization": f"Bearer {token}"},
        ).status_code
        == 401
    )
    assert (
        client.get(
            "/v1/collector/whoami",
            headers={"Authorization": "Bearer definitely-not-a-token"},
        ).status_code
        == 401
    )


def test_other_organization_cannot_revoke_or_list_a_credential(client):
    _, alpha_source, created, _ = create_source_and_credential(client)
    beta = create_organization(client, "bob-token", "beta-team")
    beta_headers = user_headers("bob-token", beta["id"])

    listed = client.get(
        f"/v1/sources/{alpha_source['id']}/credentials", headers=beta_headers
    )
    assert listed.status_code == 404
    revoked = client.delete(
        f"/v1/sources/{alpha_source['id']}/credentials/{created['id']}",
        headers=beta_headers,
    )
    assert revoked.status_code == 404
