from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from cacheeconomics_control_plane.app import create_app
from cacheeconomics_control_plane.security import (
    AuthenticationError,
    IdentityClaims,
    OidcTokenVerifier,
)
from cacheeconomics_control_plane.settings import Settings

from conftest import FakeTokenVerifier, user_headers


class SigningKey:
    def __init__(self, key):
        self.key = key


class LocalJwks:
    def __init__(self, public_key):
        self.public_key = public_key

    def get_signing_key_from_jwt(self, token):
        return SigningKey(self.public_key)


@pytest.fixture()
def oidc():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    settings = Settings(
        _env_file=None,
        environment="test",
        database_url="sqlite+pysqlite://",
        oidc_issuer="https://issuer.test/",
        oidc_audience="cacheeconomics-api",
        oidc_jwks_url="https://issuer.test/jwks",
        oidc_algorithms=("RS256",),
    )
    verifier = OidcTokenVerifier(settings)
    verifier._jwks = LocalJwks(private_key.public_key())
    return private_key, verifier


def signed(private_key, **overrides):
    now = datetime.now(timezone.utc)
    claims = {
        "iss": "https://issuer.test/",
        "sub": "user-123",
        "aud": "cacheeconomics-api",
        "iat": now,
        "exp": now + timedelta(minutes=5),
        "email": "person@example.test",
        "email_verified": True,
        "name": "Person",
    }
    claims.update(overrides)
    return jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": "test"})


def test_valid_oidc_token_is_verified_without_network(oidc):
    private_key, verifier = oidc
    claims = verifier.verify(signed(private_key))
    assert claims.issuer == "https://issuer.test/"
    assert claims.subject == "user-123"
    assert claims.email == "person@example.test"


@pytest.mark.parametrize(
    "change",
    [
        {"aud": "some-other-api"},
        {"iss": "https://attacker.test/"},
        {"exp": datetime.now(timezone.utc) - timedelta(seconds=1)},
        {"sub": ""},
    ],
)
def test_wrong_audience_issuer_expiry_or_subject_is_rejected(oidc, change):
    private_key, verifier = oidc
    with pytest.raises(AuthenticationError):
        verifier.verify(signed(private_key, **change))


def test_unverified_email_is_not_trusted(oidc):
    private_key, verifier = oidc
    claims = verifier.verify(signed(private_key, email_verified=False))
    assert claims.email is None


def test_claims_removed_by_identity_provider_are_cleared(client):
    first = client.get("/v1/me", headers=user_headers("alice-token"))
    assert first.json()["user"]["email"] == "alice@example.test"
    assert first.json()["user"]["display_name"] == "Alice"

    client.app.state.token_verifier.identities["alice-token"] = IdentityClaims(
        issuer="https://issuer.test/",
        subject="alice",
    )
    refreshed = client.get("/v1/me", headers=user_headers("alice-token"))
    assert refreshed.status_code == 200
    assert refreshed.json()["user"]["email"] is None
    assert refreshed.json()["user"]["display_name"] is None


def test_unknown_or_missing_bearer_is_401(client):
    assert client.get("/v1/me").status_code == 401
    response = client.get("/v1/me", headers=user_headers("made-up"))
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_security_headers_and_request_id_are_present(client):
    response = client.get("/healthz", headers={"X-Request-ID": "request-123"})
    assert response.status_code == 200
    assert response.headers["x-request-id"] == "request-123"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_unsafe_request_id_is_replaced(client):
    response = client.get("/healthz", headers={"X-Request-ID": "bad\nvalue"})
    assert response.status_code == 200
    assert response.headers["x-request-id"] != "bad\nvalue"


def test_production_identity_urls_must_use_https():
    with pytest.raises(ValueError, match="HTTPS"):
        Settings(
            _env_file=None,
            environment="production",
            oidc_issuer="http://issuer.test/",
            oidc_jwks_url="https://issuer.test/jwks",
        )


def test_production_does_not_allow_every_host():
    with pytest.raises(ValueError, match="allowed_hosts"):
        Settings(
            _env_file=None,
            environment="production",
            oidc_issuer="https://issuer.test/",
            oidc_jwks_url="https://issuer.test/jwks",
            allowed_hosts=["*"],
        )


def test_production_rejects_placeholder_identity_host():
    with pytest.raises(ValueError, match="placeholder"):
        Settings(
            _env_file=None,
            environment="production",
            database_url="postgresql+psycopg://app:secret@db/cacheeconomics",
            allowed_hosts=["api.example.test"],
        )


def test_production_rejects_development_database_defaults():
    with pytest.raises(ValueError, match="production database URL"):
        Settings(
            _env_file=None,
            environment="production",
            oidc_issuer="https://issuer.test/",
            oidc_jwks_url="https://issuer.test/jwks",
            allowed_hosts=["api.example.test"],
        )


def test_loopback_health_probe_does_not_require_public_host_allowlist(engine):
    settings = Settings(
        _env_file=None,
        environment="production",
        database_url="postgresql+psycopg://app:secret@db/cacheeconomics",
        oidc_issuer="https://issuer.test/",
        oidc_jwks_url="https://issuer.test/jwks",
        allowed_hosts=["api.example.test"],
        metrics_bearer_token="m" * 32,
    )
    application = create_app(
        settings,
        engine=engine,
        token_verifier=FakeTokenVerifier(),
    )
    with TestClient(
        application,
        base_url="http://127.0.0.1:8000",
        client=("127.0.0.1", 50000),
    ) as production_client:
        health = production_client.get("/healthz")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}
        assert health.headers["cache-control"] == "no-store"

        # Only the loopback liveness path bypasses TrustedHostMiddleware.
        assert production_client.get("/v1/me").status_code == 400

    with TestClient(
        application,
        base_url="http://127.0.0.1:8000",
        client=("203.0.113.10", 50000),
    ) as external_client:
        assert external_client.get("/healthz").status_code == 400
