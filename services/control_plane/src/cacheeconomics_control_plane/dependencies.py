from __future__ import annotations

import hmac
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Callable, Generator, Optional

from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .db import set_identity_context
from .models import Membership, Source, SourceCredential, User
from .rbac import Permission, Role, has_permission
from .security import (
    AuthenticationError,
    IdentityClaims,
    TokenVerifier,
    parse_collector_credential,
)


human_bearer = HTTPBearer(auto_error=False, scheme_name="OIDC bearer token")
collector_bearer = HTTPBearer(auto_error=False, scheme_name="Collector credential")


def get_session(request: Request) -> Generator[Session, None, None]:
    session = request.app.state.session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_token_verifier(request: Request) -> TokenVerifier:
    return request.app.state.token_verifier


@dataclass(frozen=True)
class UserPrincipal:
    user: User
    claims: IdentityClaims


def _unauthorized(detail: str = "authentication required") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def current_user(
    credentials: Annotated[
        Optional[HTTPAuthorizationCredentials], Depends(human_bearer)
    ],
    verifier: Annotated[TokenVerifier, Depends(get_token_verifier)],
    session: Annotated[Session, Depends(get_session)],
) -> UserPrincipal:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized()
    try:
        claims = verifier.verify(credentials.credentials)
    except AuthenticationError:
        raise _unauthorized("bearer token is invalid") from None

    user = session.scalar(
        select(User).where(
            User.oidc_issuer == claims.issuer,
            User.oidc_subject == claims.subject,
        )
    )
    if user is None:
        user = User(
            oidc_issuer=claims.issuer,
            oidc_subject=claims.subject,
            email=claims.email,
            display_name=claims.display_name,
        )
        try:
            with session.begin_nested():
                session.add(user)
                session.flush()
        except IntegrityError:
            # Two first requests for one OIDC subject may race. The unique
            # identity constraint chooses one row; the loser reuses it.
            user = session.scalar(
                select(User).where(
                    User.oidc_issuer == claims.issuer,
                    User.oidc_subject == claims.subject,
                )
            )
            if user is None:
                raise
    else:
        # These values mirror the current verified token. Keeping an older
        # value after the provider removes a claim (or stops verifying an
        # email) would turn stale identity data back into trusted data.
        user.email = claims.email
        user.display_name = claims.display_name
    set_identity_context(session, user_id=user.id)
    return UserPrincipal(user=user, claims=claims)


@dataclass(frozen=True)
class OrganizationContext:
    organization_id: uuid.UUID
    user: User
    role: Role


def current_organization(
    principal: Annotated[UserPrincipal, Depends(current_user)],
    session: Annotated[Session, Depends(get_session)],
    organization_header: Annotated[
        str, Header(alias="X-Organization-ID", min_length=1, max_length=64)
    ],
) -> OrganizationContext:
    try:
        organization_id = uuid.UUID(organization_header)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="X-Organization-ID must be a UUID")

    membership = session.scalar(
        select(Membership).where(
            Membership.organization_id == organization_id,
            Membership.user_id == principal.user.id,
        )
    )
    if membership is None:
        raise HTTPException(status_code=403, detail="organization access denied")
    set_identity_context(
        session,
        user_id=principal.user.id,
        organization_id=organization_id,
    )
    return OrganizationContext(
        organization_id=organization_id,
        user=principal.user,
        role=Role(membership.role),
    )


def require_permission(permission: Permission) -> Callable[..., OrganizationContext]:
    def dependency(
        context: Annotated[OrganizationContext, Depends(current_organization)],
    ) -> OrganizationContext:
        if not has_permission(context.role, permission):
            raise HTTPException(status_code=403, detail="permission denied")
        return context

    return dependency


@dataclass(frozen=True)
class CollectorPrincipal:
    credential_id: uuid.UUID
    organization_id: uuid.UUID
    source_id: uuid.UUID


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def current_collector(
    credentials: Annotated[
        Optional[HTTPAuthorizationCredentials], Depends(collector_bearer)
    ],
    session: Annotated[Session, Depends(get_session)],
) -> CollectorPrincipal:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized()
    try:
        prefix, digest = parse_collector_credential(credentials.credentials)
    except AuthenticationError:
        raise _unauthorized("collector credential is invalid") from None

    if session.bind is not None and session.bind.dialect.name == "postgresql":
        match = session.execute(
            text(
                "SELECT credential_id, organization_id, source_id "
                "FROM authenticate_source_credential(:prefix, :digest)"
            ),
            {"prefix": prefix, "digest": digest},
        ).mappings().one_or_none()
        if match is None:
            raise _unauthorized("collector credential is invalid")
        principal = CollectorPrincipal(
            credential_id=match["credential_id"],
            organization_id=match["organization_id"],
            source_id=match["source_id"],
        )
    else:
        stored = session.scalar(
            select(SourceCredential).where(SourceCredential.credential_prefix == prefix)
        )
        now = datetime.now(timezone.utc)
        source = session.get(Source, stored.source_id) if stored is not None else None
        invalid = (
            stored is None
            or not hmac.compare_digest(stored.credential_digest, digest)
            or stored.revoked_at is not None
            or (stored.expires_at is not None and _aware(stored.expires_at) <= now)
            or source is None
            or not source.enabled
        )
        if invalid:
            raise _unauthorized("collector credential is invalid")
        stored.last_used_at = now
        principal = CollectorPrincipal(
            credential_id=stored.id,
            organization_id=stored.organization_id,
            source_id=stored.source_id,
        )

    set_identity_context(session, organization_id=principal.organization_id)
    return principal
