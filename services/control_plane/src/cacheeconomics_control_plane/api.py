from __future__ import annotations

import hmac
import json
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import ValidationError
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .api_schemas import (
    AuditEventResponse,
    AnalysisRecordResponse,
    CollectorIdentityResponse,
    CredentialCreate,
    CredentialResponse,
    DashboardConfigResponse,
    IngestBatchEnvelope,
    IngestBatchResponse,
    IngestEventV1,
    IngestItemResult,
    JobCreate,
    JobResponse,
    MeResponse,
    MembershipCreate,
    MembershipResponse,
    MembershipUpdate,
    NewCredentialResponse,
    OrganizationCreate,
    OrganizationResponse,
    OperationsBucket,
    OperationsErrorCount,
    OperationsStatusCounts,
    SourceCreate,
    SourceHealthResponse,
    SourceOperationsResponse,
    SourceResponse,
    TimingSummary,
    UserResponse,
)
from .db import set_identity_context
from .dependencies import (
    CollectorPrincipal,
    OrganizationContext,
    UserPrincipal,
    current_collector,
    current_organization,
    current_user,
    get_session,
    require_permission,
)
from .models import (
    AnalysisResult,
    AuditEvent,
    IngestEvent,
    Job,
    Membership,
    Organization,
    Source,
    SourceCredential,
    SourceHealth,
    User,
)
from .rbac import Permission, Role, can_assign_role
from .security import new_collector_credential


router = APIRouter()


def _organization_rows(session: Session, user_id: uuid.UUID):
    return session.execute(
        select(Organization, Membership.role)
        .join(Membership, Membership.organization_id == Organization.id)
        .where(Membership.user_id == user_id)
        .order_by(Organization.name, Organization.id)
    ).all()


def _organization_response(organization: Organization, role: str) -> OrganizationResponse:
    return OrganizationResponse(
        id=organization.id,
        name=organization.name,
        slug=organization.slug,
        role=Role(role),
    )


def _audit(
    session: Session,
    context: OrganizationContext,
    *,
    action: str,
    resource_type: str,
    resource_id: uuid.UUID | None,
    details: dict | None = None,
) -> None:
    session.add(
        AuditEvent(
            organization_id=context.organization_id,
            actor_user_id=context.user.id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
        )
    )


def _source_in_organization(
    session: Session, organization_id: uuid.UUID, source_id: uuid.UUID
) -> Source:
    source = session.scalar(
        select(Source).where(
            Source.id == source_id,
            Source.organization_id == organization_id,
        )
    )
    if source is None:
        raise HTTPException(status_code=404, detail="source not found")
    return source


def _credential_response(credential: SourceCredential) -> CredentialResponse:
    return CredentialResponse(
        id=credential.id,
        source_id=credential.source_id,
        name=credential.name,
        prefix=credential.credential_prefix,
        created_at=credential.created_at,
        expires_at=credential.expires_at,
        last_used_at=credential.last_used_at,
        revoked_at=credential.revoked_at,
    )


def _membership_response(user: User, membership: Membership) -> MembershipResponse:
    return MembershipResponse(
        user=UserResponse.model_validate(user),
        role=Role(membership.role),
        created_at=membership.created_at,
    )


@router.get("/healthz", tags=["operations"])
def health() -> dict:
    return {"status": "ok"}


@router.get("/metrics", include_in_schema=False, tags=["operations"])
def metrics(request: Request) -> Response:
    settings = request.app.state.settings
    if not settings.metrics_enabled:
        raise HTTPException(status_code=404, detail="metrics are disabled")
    configured = settings.metrics_bearer_token
    if configured is not None:
        supplied = request.headers.get("Authorization", "")
        expected = f"Bearer {configured.get_secret_value()}"
        if not hmac.compare_digest(supplied, expected):
            raise HTTPException(
                status_code=401,
                detail="metrics authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
    return Response(
        request.app.state.metrics.render(),
        media_type="text/plain; version=0.0.4",
    )


@router.get(
    "/v1/dashboard/config",
    response_model=DashboardConfigResponse,
    tags=["dashboard"],
)
def dashboard_config(request: Request) -> DashboardConfigResponse:
    settings = request.app.state.settings
    enabled = all(
        (
            settings.dashboard_oidc_authorization_endpoint,
            settings.dashboard_oidc_token_endpoint,
            settings.dashboard_oidc_client_id,
            settings.dashboard_redirect_uri,
            settings.oidc_audience,
        )
    )
    return DashboardConfigResponse(
        oidc_enabled=enabled,
        authorization_endpoint=settings.dashboard_oidc_authorization_endpoint,
        token_endpoint=settings.dashboard_oidc_token_endpoint,
        client_id=settings.dashboard_oidc_client_id,
        audience=settings.oidc_audience if enabled else None,
        scopes=settings.dashboard_oidc_scopes,
        redirect_uri=settings.dashboard_redirect_uri,
        allow_development_token=settings.dashboard_allow_development_token,
    )


@router.get("/readyz", tags=["operations"])
def ready(session: Annotated[Session, Depends(get_session)]) -> dict:
    session.execute(text("SELECT 1"))
    return {"status": "ready"}


@router.get("/v1/me", response_model=MeResponse, tags=["identity"])
def me(
    principal: Annotated[UserPrincipal, Depends(current_user)],
    session: Annotated[Session, Depends(get_session)],
) -> MeResponse:
    organizations = [
        _organization_response(organization, role)
        for organization, role in _organization_rows(session, principal.user.id)
    ]
    return MeResponse(
        user=UserResponse.model_validate(principal.user),
        organizations=organizations,
    )


@router.get(
    "/v1/organizations",
    response_model=list[OrganizationResponse],
    tags=["organizations"],
)
def list_organizations(
    principal: Annotated[UserPrincipal, Depends(current_user)],
    session: Annotated[Session, Depends(get_session)],
) -> list[OrganizationResponse]:
    return [
        _organization_response(organization, role)
        for organization, role in _organization_rows(session, principal.user.id)
    ]


@router.post(
    "/v1/organizations",
    response_model=OrganizationResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["organizations"],
)
def create_organization(
    body: OrganizationCreate,
    principal: Annotated[UserPrincipal, Depends(current_user)],
    session: Annotated[Session, Depends(get_session)],
) -> OrganizationResponse:
    if session.scalar(select(Organization.id).where(Organization.slug == body.slug)):
        raise HTTPException(status_code=409, detail="organization slug already exists")

    organization = Organization(id=uuid.uuid4(), name=body.name, slug=body.slug)
    set_identity_context(
        session,
        user_id=principal.user.id,
        organization_id=organization.id,
    )
    session.add(organization)
    try:
        session.flush()
    except IntegrityError:
        # RLS deliberately hides slugs belonging to other organizations, so a
        # preflight query cannot reliably detect this race/conflict.
        session.rollback()
        raise HTTPException(status_code=409, detail="organization slug already exists")
    membership = Membership(
        organization_id=organization.id,
        user_id=principal.user.id,
        role=Role.OWNER.value,
    )
    session.add(membership)
    context = OrganizationContext(
        organization_id=organization.id,
        user=principal.user,
        role=Role.OWNER,
    )
    _audit(
        session,
        context,
        action="organization.created",
        resource_type="organization",
        resource_id=organization.id,
    )
    session.flush()
    return _organization_response(organization, membership.role)


@router.get(
    "/v1/organization",
    response_model=OrganizationResponse,
    tags=["organizations"],
)
def get_organization(
    context: Annotated[OrganizationContext, Depends(current_organization)],
    session: Annotated[Session, Depends(get_session)],
) -> OrganizationResponse:
    organization = session.get(Organization, context.organization_id)
    if organization is None:
        raise HTTPException(status_code=404, detail="organization not found")
    return _organization_response(organization, context.role.value)


member_manager = require_permission(Permission.MEMBER_MANAGE)
source_reader = require_permission(Permission.SOURCE_READ)
source_manager = require_permission(Permission.SOURCE_MANAGE)
credential_manager = require_permission(Permission.CREDENTIAL_MANAGE)
job_runner = require_permission(Permission.JOB_RUN)


@router.get(
    "/v1/memberships",
    response_model=list[MembershipResponse],
    tags=["memberships"],
)
def list_memberships(
    context: Annotated[OrganizationContext, Depends(member_manager)],
    session: Annotated[Session, Depends(get_session)],
) -> list[MembershipResponse]:
    rows = session.execute(
        select(User, Membership)
        .join(Membership, Membership.user_id == User.id)
        .where(Membership.organization_id == context.organization_id)
        .order_by(User.id)
    ).all()
    return [_membership_response(user, membership) for user, membership in rows]


@router.post(
    "/v1/memberships",
    response_model=MembershipResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["memberships"],
)
def add_membership(
    body: MembershipCreate,
    context: Annotated[OrganizationContext, Depends(member_manager)],
    session: Annotated[Session, Depends(get_session)],
) -> MembershipResponse:
    if not can_assign_role(context.role, body.role):
        raise HTTPException(status_code=403, detail="only an owner can assign owner")
    user = session.get(User, body.user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="user not found; the user must sign in first")
    existing = session.get(Membership, (context.organization_id, body.user_id))
    if existing is not None:
        raise HTTPException(status_code=409, detail="membership already exists")

    membership = Membership(
        organization_id=context.organization_id,
        user_id=body.user_id,
        role=body.role.value,
    )
    session.add(membership)
    try:
        session.flush()
    except IntegrityError:
        # The membership may have been added after the preflight lookup but
        # before this insert. Return the same public conflict either way.
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail="membership already exists",
        ) from None
    _audit(
        session,
        context,
        action="membership.created",
        resource_type="user",
        resource_id=body.user_id,
        details={"role": body.role.value},
    )
    return _membership_response(user, membership)


def _membership_or_404(
    session: Session, organization_id: uuid.UUID, user_id: uuid.UUID
) -> Membership:
    membership = session.get(Membership, (organization_id, user_id))
    if membership is None:
        raise HTTPException(status_code=404, detail="membership not found")
    return membership


def _assert_owner_change_is_safe(
    session: Session,
    context: OrganizationContext,
    membership: Membership,
    requested_role: Role | None,
) -> None:
    current_role = Role(membership.role)
    if not can_assign_role(context.role, current_role):
        raise HTTPException(status_code=403, detail="only an owner can change an owner")
    if requested_role is not None and not can_assign_role(context.role, requested_role):
        raise HTTPException(status_code=403, detail="only an owner can assign owner")
    removing_owner = current_role is Role.OWNER and requested_role is not Role.OWNER
    if removing_owner:
        if session.bind is not None and session.bind.dialect.name == "postgresql":
            # Serialize owner removals for this organization. Without the lock,
            # two owners could each observe a count of two and remove both.
            session.execute(
                text("SELECT pg_advisory_xact_lock(hashtextextended(:org_id, 0))"),
                {"org_id": str(context.organization_id)},
            )
        owner_count = session.scalar(
            select(func.count())
            .select_from(Membership)
            .where(
                Membership.organization_id == context.organization_id,
                Membership.role == Role.OWNER.value,
            )
        )
        if owner_count == 1:
            raise HTTPException(status_code=409, detail="organization must keep at least one owner")


@router.patch(
    "/v1/memberships/{user_id}",
    response_model=MembershipResponse,
    tags=["memberships"],
)
def update_membership(
    user_id: uuid.UUID,
    body: MembershipUpdate,
    context: Annotated[OrganizationContext, Depends(member_manager)],
    session: Annotated[Session, Depends(get_session)],
) -> MembershipResponse:
    membership = _membership_or_404(session, context.organization_id, user_id)
    _assert_owner_change_is_safe(session, context, membership, body.role)
    membership.role = body.role.value
    _audit(
        session,
        context,
        action="membership.role_changed",
        resource_type="user",
        resource_id=user_id,
        details={"role": body.role.value},
    )
    session.flush()
    user = session.get(User, user_id)
    return _membership_response(user, membership)


@router.delete(
    "/v1/memberships/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["memberships"],
)
def delete_membership(
    user_id: uuid.UUID,
    context: Annotated[OrganizationContext, Depends(member_manager)],
    session: Annotated[Session, Depends(get_session)],
) -> Response:
    membership = _membership_or_404(session, context.organization_id, user_id)
    _assert_owner_change_is_safe(session, context, membership, None)
    session.delete(membership)
    _audit(
        session,
        context,
        action="membership.deleted",
        resource_type="user",
        resource_id=user_id,
        details={"previous_role": membership.role},
    )
    session.flush()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/v1/sources", response_model=list[SourceResponse], tags=["sources"])
def list_sources(
    context: Annotated[OrganizationContext, Depends(source_reader)],
    session: Annotated[Session, Depends(get_session)],
) -> list[SourceResponse]:
    return list(
        session.scalars(
            select(Source)
            .where(Source.organization_id == context.organization_id)
            .order_by(Source.name, Source.id)
        )
    )


@router.post(
    "/v1/sources",
    response_model=SourceResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["sources"],
)
def create_source(
    body: SourceCreate,
    context: Annotated[OrganizationContext, Depends(source_manager)],
    session: Annotated[Session, Depends(get_session)],
) -> Source:
    existing = session.scalar(
        select(Source.id).where(
            Source.organization_id == context.organization_id,
            Source.name == body.name,
        )
    )
    if existing:
        raise HTTPException(status_code=409, detail="source name already exists")
    source = Source(
        id=uuid.uuid4(),
        organization_id=context.organization_id,
        name=body.name,
        kind=body.kind,
    )
    session.add(source)
    try:
        session.flush()
    except IntegrityError:
        # A concurrent create can win after the preflight lookup.
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail="source name already exists",
        ) from None
    session.add(
        SourceHealth(
            source_id=source.id,
            organization_id=context.organization_id,
            accepted_events=0,
            duplicate_events=0,
            rejected_events=0,
            consecutive_failures=0,
        )
    )
    _audit(
        session,
        context,
        action="source.created",
        resource_type="source",
        resource_id=source.id,
        details={"kind": source.kind},
    )
    return source


@router.get(
    "/v1/sources/{source_id}/credentials",
    response_model=list[CredentialResponse],
    tags=["credentials"],
)
def list_credentials(
    source_id: uuid.UUID,
    context: Annotated[OrganizationContext, Depends(credential_manager)],
    session: Annotated[Session, Depends(get_session)],
) -> list[CredentialResponse]:
    _source_in_organization(session, context.organization_id, source_id)
    rows = session.scalars(
        select(SourceCredential)
        .where(
            SourceCredential.organization_id == context.organization_id,
            SourceCredential.source_id == source_id,
        )
        .order_by(SourceCredential.created_at, SourceCredential.id)
    )
    return [_credential_response(row) for row in rows]


@router.post(
    "/v1/sources/{source_id}/credentials",
    response_model=NewCredentialResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["credentials"],
)
def create_credential(
    source_id: uuid.UUID,
    body: CredentialCreate,
    context: Annotated[OrganizationContext, Depends(credential_manager)],
    session: Annotated[Session, Depends(get_session)],
) -> NewCredentialResponse:
    _source_in_organization(session, context.organization_id, source_id)
    now = datetime.now(timezone.utc)
    expires = body.expires_at
    if expires is not None:
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        expires = expires.astimezone(timezone.utc)
        if expires <= now:
            raise HTTPException(status_code=422, detail="expires_at must be in the future")

    generated = new_collector_credential()
    credential = SourceCredential(
        id=uuid.uuid4(),
        organization_id=context.organization_id,
        source_id=source_id,
        name=body.name,
        credential_prefix=generated.prefix,
        credential_digest=generated.digest,
        created_by_user_id=context.user.id,
        expires_at=expires,
    )
    session.add(credential)
    _audit(
        session,
        context,
        action="source_credential.created",
        resource_type="source_credential",
        resource_id=credential.id,
        details={"source_id": str(source_id), "prefix": generated.prefix},
    )
    session.flush()
    base = _credential_response(credential)
    return NewCredentialResponse(**base.model_dump(), token=generated.token)


@router.delete(
    "/v1/sources/{source_id}/credentials/{credential_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["credentials"],
)
def revoke_credential(
    source_id: uuid.UUID,
    credential_id: uuid.UUID,
    context: Annotated[OrganizationContext, Depends(credential_manager)],
    session: Annotated[Session, Depends(get_session)],
) -> Response:
    _source_in_organization(session, context.organization_id, source_id)
    credential = session.scalar(
        select(SourceCredential).where(
            SourceCredential.id == credential_id,
            SourceCredential.organization_id == context.organization_id,
            SourceCredential.source_id == source_id,
        )
    )
    if credential is None:
        raise HTTPException(status_code=404, detail="credential not found")
    if credential.revoked_at is None:
        credential.revoked_at = datetime.now(timezone.utc)
        _audit(
            session,
            context,
            action="source_credential.revoked",
            resource_type="source_credential",
            resource_id=credential.id,
            details={"source_id": str(source_id)},
        )
        session.flush()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/v1/audit-events",
    response_model=list[AuditEventResponse],
    tags=["audit"],
)
def list_audit_events(
    context: Annotated[OrganizationContext, Depends(member_manager)],
    session: Annotated[Session, Depends(get_session)],
    limit: int = 100,
) -> list[AuditEventResponse]:
    limit = max(1, min(limit, 500))
    rows = session.scalars(
        select(AuditEvent)
        .where(AuditEvent.organization_id == context.organization_id)
        .order_by(AuditEvent.occurred_at.desc(), AuditEvent.id.desc())
        .limit(limit)
    )
    return [
        AuditEventResponse(
            id=row.id,
            actor_user_id=row.actor_user_id,
            action=row.action,
            resource_type=row.resource_type,
            resource_id=row.resource_id,
            details=row.details,
            occurred_at=row.occurred_at,
        )
        for row in rows
    ]


@router.get(
    "/v1/collector/whoami",
    response_model=CollectorIdentityResponse,
    tags=["collectors"],
)
def collector_identity(
    principal: Annotated[CollectorPrincipal, Depends(current_collector)],
) -> CollectorIdentityResponse:
    return CollectorIdentityResponse(
        organization_id=principal.organization_id,
        source_id=principal.source_id,
        credential_id=principal.credential_id,
    )


def _safe_validation_reason(error: ValidationError) -> str:
    first = error.errors(include_input=False, include_context=False)[0]
    location = ".".join(str(part) for part in first.get("loc", ())) or "event"
    kind = str(first.get("type", "invalid"))
    return f"validation_error:{location}:{kind}"[:256]


async def _ingest_batch(request: Request) -> IngestBatchEnvelope:
    try:
        raw: Any = await request.json()
    # Python's decoder also raises a plain ValueError when a JSON integer
    # exceeds the interpreter's safety digit limit. It is malformed client
    # input, not an internal service failure.
    except ValueError:
        raise HTTPException(status_code=400, detail="body must be valid JSON") from None
    try:
        return IngestBatchEnvelope.model_validate(raw)
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=_safe_validation_reason(error)) from None


def _source_health(
    session: Session,
    organization_id: uuid.UUID,
    source_id: uuid.UUID,
) -> SourceHealth:
    health = session.scalar(
        select(SourceHealth)
        .where(
            SourceHealth.source_id == source_id,
            SourceHealth.organization_id == organization_id,
        )
        .with_for_update()
    )
    if health is None:
        health = SourceHealth(
            source_id=source_id,
            organization_id=organization_id,
            accepted_events=0,
            duplicate_events=0,
            rejected_events=0,
            consecutive_failures=0,
        )
        session.add(health)
    return health


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@router.post(
    "/v1/ingest/events",
    response_model=IngestBatchResponse,
    tags=["ingestion"],
)
def ingest_events(
    batch: Annotated[IngestBatchEnvelope, Depends(_ingest_batch)],
    principal: Annotated[CollectorPrincipal, Depends(current_collector)],
    request: Request,
    session: Annotated[Session, Depends(get_session)],
) -> IngestBatchResponse:
    settings = request.app.state.settings
    if len(batch.events) > settings.ingest_max_batch_events:
        raise HTTPException(
            status_code=422,
            detail=f"batch cannot exceed {settings.ingest_max_batch_events} events",
        )

    now = datetime.now(timezone.utc)
    health = _source_health(
        session,
        principal.organization_id,
        principal.source_id,
    )
    health.last_received_at = now
    results: list[IngestItemResult] = []
    accepted_ids: list[str] = []
    accepted = duplicates = rejected = 0

    for index, raw_event in enumerate(batch.events):
        event_id = (
            raw_event.get("event_id")
            if isinstance(raw_event, dict)
            and isinstance(raw_event.get("event_id"), str)
            and len(raw_event["event_id"]) <= 256
            else None
        )
        try:
            event_bytes = len(
                json.dumps(
                    raw_event,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            )
        except (TypeError, ValueError):
            event_bytes = settings.ingest_max_event_bytes + 1
        if event_bytes > settings.ingest_max_event_bytes:
            rejected += 1
            results.append(
                IngestItemResult(
                    index=index,
                    event_id=event_id,
                    status="rejected",
                    reason="validation_error:event:event_too_large_or_nonfinite",
                )
            )
            continue
        try:
            event = IngestEventV1.model_validate(raw_event)
        except ValidationError as error:
            rejected += 1
            results.append(
                IngestItemResult(
                    index=index,
                    event_id=event_id,
                    status="rejected",
                    reason=_safe_validation_reason(error),
                )
            )
            continue

        sent_at = _utc(event.sent_at)
        latest_sent = (
            _utc(health.last_event_sent_at)
            if health.last_event_sent_at is not None
            else None
        )
        if latest_sent is None or sent_at >= latest_sent:
            health.last_event_sent_at = sent_at
            health.last_ingest_lag_seconds = max(0.0, (now - sent_at).total_seconds())

        existing = session.scalar(
            select(IngestEvent.id).where(
                IngestEvent.source_id == principal.source_id,
                IngestEvent.event_id == event.event_id,
            )
        )
        if existing is not None:
            duplicates += 1
            results.append(
                IngestItemResult(
                    index=index,
                    event_id=event.event_id,
                    status="duplicate",
                )
            )
            continue

        payload = event.model_dump(mode="json", by_alias=True)
        if payload["usage"].get("cache_creation") is None:
            payload["usage"].pop("cache_creation", None)
        stored = IngestEvent(
            id=uuid.uuid4(),
            organization_id=principal.organization_id,
            source_id=principal.source_id,
            event_id=event.event_id,
            schema_version=1,
            payload=payload,
            received_at=now,
        )
        try:
            with session.begin_nested():
                session.add(stored)
                session.flush()
        except IntegrityError:
            # Another copy of the same event may win between the lookup and
            # insert. The source/event unique key is the idempotency boundary.
            existing = session.scalar(
                select(IngestEvent.id).where(
                    IngestEvent.source_id == principal.source_id,
                    IngestEvent.event_id == event.event_id,
                )
            )
            if existing is None:
                raise
            duplicates += 1
            results.append(
                IngestItemResult(
                    index=index,
                    event_id=event.event_id,
                    status="duplicate",
                )
            )
            continue

        accepted += 1
        accepted_ids.append(str(stored.id))
        results.append(
            IngestItemResult(
                index=index,
                event_id=event.event_id,
                status="accepted",
            )
        )

    health.accepted_events += accepted
    health.duplicate_events += duplicates
    health.rejected_events += rejected

    job_id = None
    if accepted_ids:
        job = Job(
            id=uuid.uuid4(),
            organization_id=principal.organization_id,
            source_id=principal.source_id,
            kind="analyze_source",
            state="queued",
            attempt=0,
            max_attempts=settings.job_max_attempts,
            payload={
                "accepted_event_ids": accepted_ids,
                "through_received_at": now.isoformat(),
            },
            scheduled_for=now,
        )
        session.add(job)
        session.flush()
        job_id = job.id

    # Metrics describe durable outcomes, not work that might still roll back in
    # the request dependency's final commit. A second commit in get_session is
    # harmless and keeps the shared dependency behavior unchanged.
    session.commit()
    request.app.state.metrics.ingest_batch(
        accepted=accepted,
        duplicates=duplicates,
        rejected=rejected,
    )

    return IngestBatchResponse(
        accepted=accepted,
        duplicates=duplicates,
        rejected=rejected,
        job_id=job_id,
        items=results,
    )


def _health_response(source_id: uuid.UUID, health: SourceHealth | None) -> SourceHealthResponse:
    if health is None:
        return SourceHealthResponse(
            source_id=source_id,
            accepted_events=0,
            duplicate_events=0,
            rejected_events=0,
            last_received_at=None,
            last_event_sent_at=None,
            last_ingest_lag_seconds=None,
            last_success_at=None,
            last_error_at=None,
            last_error_type=None,
            last_job_duration_ms=None,
            consecutive_failures=0,
        )
    return SourceHealthResponse.model_validate(health, from_attributes=True)


@router.get(
    "/v1/sources/{source_id}/health",
    response_model=SourceHealthResponse,
    tags=["ingestion"],
)
def source_health(
    source_id: uuid.UUID,
    context: Annotated[OrganizationContext, Depends(source_reader)],
    session: Annotated[Session, Depends(get_session)],
) -> SourceHealthResponse:
    _source_in_organization(session, context.organization_id, source_id)
    return _health_response(source_id, session.get(SourceHealth, source_id))


def _job_response(job: Job) -> JobResponse:
    return JobResponse.model_validate(job, from_attributes=True)


def _job_in_organization(
    session: Session,
    organization_id: uuid.UUID,
    job_id: uuid.UUID,
    *,
    lock: bool = False,
) -> Job:
    statement = select(Job).where(
        Job.id == job_id,
        Job.organization_id == organization_id,
    )
    if lock:
        statement = statement.with_for_update()
    job = session.scalar(statement)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@router.post(
    "/v1/sources/{source_id}/jobs",
    response_model=JobResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["jobs"],
)
def create_analysis_job(
    source_id: uuid.UUID,
    body: JobCreate,
    context: Annotated[OrganizationContext, Depends(job_runner)],
    request: Request,
    session: Annotated[Session, Depends(get_session)],
) -> JobResponse:
    _source_in_organization(session, context.organization_id, source_id)
    scheduled_for = _utc(body.scheduled_for) if body.scheduled_for else datetime.now(timezone.utc)
    job = Job(
        id=uuid.uuid4(),
        organization_id=context.organization_id,
        source_id=source_id,
        kind="analyze_source",
        state="queued",
        attempt=0,
        max_attempts=request.app.state.settings.job_max_attempts,
        payload={
            "trigger": "manual",
            "through_received_at": scheduled_for.isoformat(),
        },
        scheduled_for=scheduled_for,
    )
    session.add(job)
    _audit(
        session,
        context,
        action="analysis_job.created",
        resource_type="job",
        resource_id=job.id,
        details={"source_id": str(source_id)},
    )
    session.flush()
    return _job_response(job)


@router.get("/v1/jobs", response_model=list[JobResponse], tags=["jobs"])
def list_jobs(
    context: Annotated[OrganizationContext, Depends(source_reader)],
    session: Annotated[Session, Depends(get_session)],
) -> list[JobResponse]:
    jobs = session.scalars(
        select(Job)
        .where(Job.organization_id == context.organization_id)
        .order_by(Job.created_at.desc(), Job.id.desc())
        .limit(200)
    )
    return [_job_response(job) for job in jobs]


@router.post("/v1/jobs/{job_id}/cancel", response_model=JobResponse, tags=["jobs"])
def cancel_job(
    job_id: uuid.UUID,
    context: Annotated[OrganizationContext, Depends(job_runner)],
    session: Annotated[Session, Depends(get_session)],
) -> JobResponse:
    job = _job_in_organization(session, context.organization_id, job_id, lock=True)
    if job.state not in {"queued", "running"}:
        raise HTTPException(status_code=409, detail="job can no longer be cancelled")
    job.state = "cancelled"
    job.finished_at = datetime.now(timezone.utc)
    job.lease_expires_at = None
    job.leased_by = None
    _audit(
        session,
        context,
        action="analysis_job.cancelled",
        resource_type="job",
        resource_id=job.id,
    )
    session.flush()
    return _job_response(job)


@router.post("/v1/jobs/{job_id}/retry", response_model=JobResponse, tags=["jobs"])
def retry_job(
    job_id: uuid.UUID,
    context: Annotated[OrganizationContext, Depends(job_runner)],
    session: Annotated[Session, Depends(get_session)],
) -> JobResponse:
    job = _job_in_organization(session, context.organization_id, job_id, lock=True)
    if job.state not in {"failed", "dead_letter"}:
        raise HTTPException(status_code=409, detail="only failed jobs can be retried")
    job.state = "queued"
    job.attempt = 0
    job.scheduled_for = datetime.now(timezone.utc)
    job.started_at = None
    job.finished_at = None
    job.lease_expires_at = None
    job.leased_by = None
    job.error_type = None
    _audit(
        session,
        context,
        action="analysis_job.retried",
        resource_type="job",
        resource_id=job.id,
    )
    session.flush()
    return _job_response(job)


@router.get(
    "/v1/sources/{source_id}/analyses/latest",
    response_model=AnalysisRecordResponse,
    tags=["analyses"],
)
def latest_analysis(
    source_id: uuid.UUID,
    context: Annotated[OrganizationContext, Depends(source_reader)],
    session: Annotated[Session, Depends(get_session)],
) -> AnalysisRecordResponse:
    _source_in_organization(session, context.organization_id, source_id)
    analysis = session.scalar(
        select(AnalysisResult)
        .where(
            AnalysisResult.organization_id == context.organization_id,
            AnalysisResult.source_id == source_id,
        )
        .order_by(AnalysisResult.created_at.desc(), AnalysisResult.id.desc())
    )
    if analysis is None:
        raise HTTPException(status_code=404, detail="analysis not found")
    return AnalysisRecordResponse.model_validate(analysis, from_attributes=True)


def _percentile_ms(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    value = ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
    return round(value * 1_000, 3)


def _timing(values: list[float]) -> TimingSummary:
    return TimingSummary(
        count=len(values),
        p50_ms=_percentile_ms(values, 0.50),
        p95_ms=_percentile_ms(values, 0.95),
    )


def _operations_bucket_minutes(window_hours: int) -> int:
    if window_hours <= 48:
        return 60
    if window_hours <= 24 * 14:
        return 360
    return 1_440


def _bucket_start(value: datetime, bucket_minutes: int) -> datetime:
    seconds = bucket_minutes * 60
    timestamp = int(_utc(value).timestamp())
    return datetime.fromtimestamp(timestamp - (timestamp % seconds), timezone.utc)


@router.get(
    "/v1/sources/{source_id}/operations",
    response_model=SourceOperationsResponse,
    tags=["operations"],
)
def source_operations(
    source_id: uuid.UUID,
    context: Annotated[OrganizationContext, Depends(source_reader)],
    request: Request,
    session: Annotated[Session, Depends(get_session)],
    window_hours: Annotated[int, Query(ge=1, le=720)] = 168,
) -> SourceOperationsResponse:
    """Return bounded, prompt-free aggregates for the selected source."""

    _source_in_organization(session, context.organization_id, source_id)
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=window_hours)
    max_events = request.app.state.settings.dashboard_max_events
    rows = session.execute(
        select(IngestEvent.payload, IngestEvent.received_at)
        .where(
            IngestEvent.organization_id == context.organization_id,
            IngestEvent.source_id == source_id,
            IngestEvent.received_at >= cutoff,
        )
        .order_by(IngestEvent.received_at.desc(), IngestEvent.id.desc())
        .limit(max_events + 1)
    ).all()
    truncated = len(rows) > max_events
    rows = list(reversed(rows[:max_events]))

    bucket_minutes = _operations_bucket_minutes(window_hours)
    bucket_counts: dict[datetime, Counter[str]] = {}
    statuses: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    latency: list[float] = []
    first_token: list[float] = []
    invalid_events = 0

    for payload, received_at in rows:
        try:
            event = IngestEventV1.model_validate(payload)
        except ValidationError:
            invalid_events += 1
            continue
        outcome = event.status.outcome
        statuses[outcome] += 1
        bucket = bucket_counts.setdefault(
            _bucket_start(received_at, bucket_minutes),
            Counter(),
        )
        bucket["requests"] += 1
        if outcome == "error":
            bucket["errors"] += 1
            errors[event.status.error_type or "unknown"] += 1
        elif outcome == "cancelled":
            bucket["cancelled"] += 1
        elif outcome == "unknown":
            bucket["unknown"] += 1
        if event.completed_at is not None:
            latency.append((event.completed_at - event.sent_at).total_seconds())
        if event.first_token_at is not None:
            first_token.append(
                (event.first_token_at - event.sent_at).total_seconds()
            )

    first_received = _utc(rows[0][1]) if rows else None
    last_received = _utc(rows[-1][1]) if rows else None
    return SourceOperationsResponse(
        source_id=source_id,
        generated_at=now,
        window_hours=window_hours,
        bucket_minutes=bucket_minutes,
        events_examined=len(rows),
        valid_events=len(rows) - invalid_events,
        invalid_events=invalid_events,
        truncated=truncated,
        first_received_at=first_received,
        last_received_at=last_received,
        status=OperationsStatusCounts(
            success=statuses["success"],
            error=statuses["error"],
            cancelled=statuses["cancelled"],
            unknown=statuses["unknown"],
        ),
        latency=_timing(latency),
        time_to_first_token=_timing(first_token),
        errors=[
            OperationsErrorCount(error_type=error_type, count=count)
            for error_type, count in sorted(
                errors.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
        buckets=[
            OperationsBucket(
                started_at=started_at,
                requests=counts["requests"],
                errors=counts["errors"],
                cancelled=counts["cancelled"],
                unknown=counts["unknown"],
            )
            for started_at, counts in sorted(bucket_counts.items())
        ],
    )
