from __future__ import annotations

import pytest

from cacheeconomics_control_plane.settings import Settings


PRODUCTION = {
    "_env_file": None,
    "environment": "production",
    "database_url": "postgresql+psycopg://app:secret@db/cacheeconomics",
    "oidc_issuer": "https://identity.example.test/",
    "oidc_jwks_url": "https://identity.example.test/jwks",
    "allowed_hosts": ["cacheeconomics.example.test"],
}


def test_production_requires_metrics_authentication_when_metrics_are_enabled():
    with pytest.raises(ValueError, match="metrics require a bearer token"):
        Settings(**PRODUCTION)


def test_production_rejects_a_blank_oidc_audience():
    with pytest.raises(ValueError, match="oidc_audience"):
        Settings(**PRODUCTION, oidc_audience="   ", metrics_enabled=False)


def test_production_rejects_short_metrics_token():
    with pytest.raises(ValueError, match="32 characters"):
        Settings(**PRODUCTION, metrics_bearer_token="short")


def test_production_rejects_partial_dashboard_oidc_configuration():
    with pytest.raises(ValueError, match="dashboard OIDC settings must be complete"):
        Settings(
            **PRODUCTION,
            metrics_bearer_token="m" * 32,
            dashboard_oidc_client_id="dashboard",
        )


def test_production_rejects_development_token_entry():
    with pytest.raises(ValueError, match="development token entry"):
        Settings(
            **PRODUCTION,
            metrics_bearer_token="m" * 32,
            dashboard_allow_development_token=True,
        )


def test_complete_production_dashboard_configuration_is_accepted():
    settings = Settings(
        **PRODUCTION,
        metrics_bearer_token="m" * 32,
        dashboard_oidc_authorization_endpoint="https://identity.example.test/authorize",
        dashboard_oidc_token_endpoint="https://identity.example.test/token",
        dashboard_oidc_client_id="cacheeconomics-dashboard",
        dashboard_redirect_uri="https://cacheeconomics.example.test/",
    )

    assert settings.dashboard_allow_development_token is False


def test_blank_compose_dashboard_values_remain_unconfigured():
    settings = Settings(
        _env_file=None,
        environment="development",
        dashboard_oidc_authorization_endpoint="",
        dashboard_oidc_token_endpoint="  ",
        dashboard_oidc_client_id="",
        dashboard_redirect_uri="",
    )

    assert settings.dashboard_oidc_authorization_endpoint is None
    assert settings.dashboard_oidc_token_endpoint is None
    assert settings.dashboard_oidc_client_id is None
    assert settings.dashboard_redirect_uri is None
