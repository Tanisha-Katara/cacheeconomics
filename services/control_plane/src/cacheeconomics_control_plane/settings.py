from __future__ import annotations

from typing import Literal, Optional
from urllib.parse import urlparse

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Control-plane configuration loaded from ``CACHEECONOMICS_*`` variables."""

    model_config = SettingsConfigDict(
        env_prefix="CACHEECONOMICS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Literal["development", "test", "production"] = "development"
    database_url: str = (
        "postgresql+psycopg://cacheeconomics_app:dev-only@127.0.0.1/"
        "cacheeconomics"
    )
    oidc_issuer: str = "https://identity.example.invalid/"
    oidc_audience: str = "cacheeconomics-api"
    oidc_jwks_url: str = "https://identity.example.invalid/.well-known/jwks.json"
    oidc_algorithms: tuple[str, ...] = ("RS256",)
    ingest_max_body_bytes: int = Field(default=1_048_576, ge=1_024, le=16_777_216)
    ingest_max_event_bytes: int = Field(default=131_072, ge=1_024, le=1_048_576)
    ingest_max_batch_events: int = Field(default=100, ge=1, le=1_000)
    analysis_max_events: int = Field(default=1_000, ge=1, le=100_000)
    dashboard_max_events: int = Field(default=5_000, ge=100, le=100_000)
    job_max_attempts: int = Field(default=3, ge=1, le=20)
    job_lease_seconds: int = Field(default=120, ge=10, le=3_600)
    job_retry_base_seconds: int = Field(default=5, ge=1, le=3_600)
    job_retry_max_seconds: int = Field(default=300, ge=1, le=86_400)
    worker_poll_seconds: float = Field(default=2.0, ge=0.1, le=60.0)
    allowed_hosts: list[str] = Field(
        default_factory=lambda: ["localhost", "127.0.0.1", "testserver", "api"]
    )
    dashboard_oidc_authorization_endpoint: Optional[str] = None
    dashboard_oidc_token_endpoint: Optional[str] = None
    dashboard_oidc_client_id: Optional[str] = None
    dashboard_oidc_scopes: str = "openid profile email"
    dashboard_redirect_uri: Optional[str] = None
    dashboard_allow_development_token: bool = False
    metrics_enabled: bool = True
    metrics_bearer_token: Optional[SecretStr] = None
    json_logs: bool = True

    @field_validator(
        "dashboard_oidc_authorization_endpoint",
        "dashboard_oidc_token_endpoint",
        "dashboard_oidc_client_id",
        "dashboard_redirect_uri",
        mode="before",
    )
    @classmethod
    def blank_dashboard_values_are_unconfigured(cls, value):
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    @model_validator(mode="after")
    def production_configuration_is_explicit(self) -> "Settings":
        if self.ingest_max_event_bytes > self.ingest_max_body_bytes:
            raise ValueError("ingest_max_event_bytes must not exceed ingest_max_body_bytes")
        if self.job_retry_max_seconds < self.job_retry_base_seconds:
            raise ValueError("job_retry_max_seconds must not be below job_retry_base_seconds")
        if self.environment != "production":
            return self
        identity_urls = (
            ("oidc_issuer", self.oidc_issuer),
            ("oidc_jwks_url", self.oidc_jwks_url),
        )
        for label, value in identity_urls:
            parsed = urlparse(value)
            if parsed.scheme != "https":
                raise ValueError(f"{label} must use HTTPS in production")
            if parsed.hostname is None or parsed.hostname.endswith(".invalid"):
                raise ValueError(f"{label} cannot use a placeholder host in production")
        if "*" in self.allowed_hosts:
            raise ValueError("allowed_hosts cannot contain '*' in production")
        if self.allowed_hosts == ["localhost", "127.0.0.1", "testserver", "api"]:
            raise ValueError("allowed_hosts must be configured for production")
        if not self.oidc_algorithms:
            raise ValueError("at least one OIDC signature algorithm is required")
        if not self.oidc_audience.strip():
            raise ValueError("oidc_audience cannot be blank in production")
        if self.database_url.startswith("sqlite") or "dev-only" in self.database_url:
            raise ValueError("a production database URL must replace development defaults")
        dashboard_values = (
            self.dashboard_oidc_authorization_endpoint,
            self.dashboard_oidc_token_endpoint,
            self.dashboard_oidc_client_id,
            self.dashboard_redirect_uri,
        )
        if any(dashboard_values) and not all(dashboard_values):
            raise ValueError("production dashboard OIDC settings must be complete")
        for endpoint in (
            self.dashboard_oidc_authorization_endpoint,
            self.dashboard_oidc_token_endpoint,
            self.dashboard_redirect_uri,
        ):
            if endpoint is None:
                continue
            parsed = urlparse(endpoint)
            if parsed.scheme != "https" or parsed.hostname is None:
                raise ValueError(
                    "production dashboard OIDC URLs must be absolute HTTPS URLs"
                )
        for endpoint in (
            self.dashboard_oidc_authorization_endpoint,
            self.dashboard_oidc_token_endpoint,
        ):
            hostname = urlparse(endpoint).hostname if endpoint is not None else None
            if hostname is not None and hostname.endswith(".invalid"):
                raise ValueError(
                    "production dashboard OIDC endpoints cannot use placeholder hosts"
                )
        if self.dashboard_allow_development_token:
            raise ValueError("development token entry cannot be enabled in production")
        if self.metrics_enabled and self.metrics_bearer_token is None:
            raise ValueError("production metrics require a bearer token")
        if (
            self.metrics_bearer_token is not None
            and len(self.metrics_bearer_token.get_secret_value()) < 32
        ):
            raise ValueError("production metrics bearer tokens must contain 32 characters")
        return self
