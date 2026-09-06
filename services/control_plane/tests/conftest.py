from __future__ import annotations

import pathlib
import sys

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool


CONTROL_PLANE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTROL_PLANE / "src"))

from cacheeconomics_control_plane.app import create_app  # noqa: E402
from cacheeconomics_control_plane.models import Base  # noqa: E402
from cacheeconomics_control_plane.security import (  # noqa: E402
    AuthenticationError,
    IdentityClaims,
)
from cacheeconomics_control_plane.settings import Settings  # noqa: E402


class FakeTokenVerifier:
    def __init__(self):
        self.identities = {
            "alice-token": IdentityClaims(
                issuer="https://issuer.test/",
                subject="alice",
                email="alice@example.test",
                display_name="Alice",
            ),
            "bob-token": IdentityClaims(
                issuer="https://issuer.test/",
                subject="bob",
                email="bob@example.test",
                display_name="Bob",
            ),
            "carol-token": IdentityClaims(
                issuer="https://issuer.test/",
                subject="carol",
                email="carol@example.test",
                display_name="Carol",
            ),
        }

    def verify(self, token: str) -> IdentityClaims:
        try:
            return self.identities[token]
        except KeyError as exc:
            raise AuthenticationError("unknown test token") from exc


@pytest.fixture()
def engine():
    database = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(database)
    try:
        yield database
    finally:
        Base.metadata.drop_all(database)
        database.dispose()


@pytest.fixture()
def client(engine):
    settings = Settings(
        _env_file=None,
        environment="test",
        database_url="sqlite+pysqlite://",
        oidc_issuer="https://issuer.test/",
        oidc_audience="cacheeconomics-test",
        oidc_jwks_url="https://issuer.test/jwks",
        allowed_hosts=["testserver"],
    )
    application = create_app(
        settings,
        engine=engine,
        token_verifier=FakeTokenVerifier(),
    )
    with TestClient(application) as test_client:
        yield test_client


def user_headers(token: str, organization_id: str | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {token}"}
    if organization_id is not None:
        headers["X-Organization-ID"] = organization_id
    return headers


def create_organization(client: TestClient, token: str, slug: str) -> dict:
    response = client.post(
        "/v1/organizations",
        headers=user_headers(token),
        json={"name": slug.replace("-", " ").title(), "slug": slug},
    )
    assert response.status_code == 201, response.text
    return response.json()
