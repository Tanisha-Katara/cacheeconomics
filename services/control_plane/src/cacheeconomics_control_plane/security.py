from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import dataclass
from typing import Optional, Protocol

import jwt
from jwt import PyJWKClient

from .settings import Settings


class AuthenticationError(Exception):
    """A credential is absent, invalid, expired, or outside its audience."""


@dataclass(frozen=True)
class IdentityClaims:
    issuer: str
    subject: str
    email: Optional[str] = None
    display_name: Optional[str] = None


class TokenVerifier(Protocol):
    def verify(self, token: str) -> IdentityClaims: ...


class OidcTokenVerifier:
    """Validate externally issued OIDC/OAuth bearer tokens against a JWKS."""

    def __init__(self, settings: Settings):
        self._issuer = settings.oidc_issuer
        self._audience = settings.oidc_audience
        self._algorithms = list(settings.oidc_algorithms)
        self._jwks = PyJWKClient(
            settings.oidc_jwks_url,
            cache_keys=True,
            cache_jwk_set=True,
            lifespan=300,
        )

    def verify(self, token: str) -> IdentityClaims:
        try:
            signing_key = self._jwks.get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=self._algorithms,
                audience=self._audience,
                issuer=self._issuer,
                options={"require": ["exp", "iss", "sub", "aud"]},
            )
        except jwt.PyJWTError as exc:
            raise AuthenticationError("bearer token could not be verified") from exc
        except Exception as exc:
            # Network/JWKS parsing errors are deliberately indistinguishable to a
            # caller. Operational logs can classify them without echoing tokens.
            raise AuthenticationError("identity keys are unavailable") from exc

        subject = claims.get("sub")
        issuer = claims.get("iss")
        if not isinstance(subject, str) or not subject or len(subject) > 512:
            raise AuthenticationError("token subject is invalid")
        if not isinstance(issuer, str) or issuer != self._issuer:
            raise AuthenticationError("token issuer is invalid")

        email = claims.get("email") if claims.get("email_verified") is True else None
        if not isinstance(email, str) or len(email) > 320:
            email = None
        name = claims.get("name")
        if not isinstance(name, str) or len(name) > 256:
            name = None
        return IdentityClaims(
            issuer=issuer,
            subject=subject,
            email=email,
            display_name=name,
        )


COLLECTOR_TOKEN = re.compile(
    r"^cec_(?P<prefix>[A-Za-z0-9_-]{12})_(?P<secret>[A-Za-z0-9_-]{43})$"
)


@dataclass(frozen=True)
class NewCollectorCredential:
    token: str
    prefix: str
    digest: str


def new_collector_credential() -> NewCollectorCredential:
    prefix = secrets.token_urlsafe(9)
    secret = secrets.token_urlsafe(32)
    token = f"cec_{prefix}_{secret}"
    return NewCollectorCredential(
        token=token,
        prefix=prefix,
        digest=hashlib.sha256(token.encode("utf-8")).hexdigest(),
    )


def parse_collector_credential(token: str) -> tuple[str, str]:
    match = COLLECTOR_TOKEN.fullmatch(token)
    if not match:
        raise AuthenticationError("collector credential is invalid")
    return (
        match.group("prefix"),
        hashlib.sha256(token.encode("utf-8")).hexdigest(),
    )
