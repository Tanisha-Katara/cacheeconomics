from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from .rbac import Role


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UserResponse(StrictModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)

    id: uuid.UUID
    email: Optional[str]
    display_name: Optional[str]


class OrganizationCreate(StrictModel):
    name: str = Field(min_length=1, max_length=160)
    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,78}[a-z0-9]$", max_length=80)


class OrganizationResponse(StrictModel):
    id: uuid.UUID
    name: str
    slug: str
    role: Role


class MeResponse(StrictModel):
    user: UserResponse
    organizations: list[OrganizationResponse]


class MembershipCreate(StrictModel):
    user_id: uuid.UUID
    role: Role


class MembershipUpdate(StrictModel):
    role: Role


class MembershipResponse(StrictModel):
    user: UserResponse
    role: Role
    created_at: datetime


class SourceCreate(StrictModel):
    name: str = Field(min_length=1, max_length=160)
    kind: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9_-]*$")


class SourceResponse(StrictModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)

    id: uuid.UUID
    organization_id: uuid.UUID
    name: str
    kind: str
    enabled: bool
    created_at: datetime


class CredentialCreate(StrictModel):
    name: str = Field(min_length=1, max_length=160)
    expires_at: Optional[datetime] = None


class CredentialResponse(StrictModel):
    id: uuid.UUID
    source_id: uuid.UUID
    name: str
    prefix: str
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    revoked_at: Optional[datetime]


class NewCredentialResponse(CredentialResponse):
    token: str
    warning: str = "Save this token now. It is not stored and will not be shown again."


class CollectorIdentityResponse(StrictModel):
    organization_id: uuid.UUID
    source_id: uuid.UUID
    credential_id: uuid.UUID


class AuditEventResponse(StrictModel):
    id: uuid.UUID
    actor_user_id: Optional[uuid.UUID]
    action: str
    resource_type: str
    resource_id: Optional[uuid.UUID]
    details: dict
    occurred_at: datetime


class IngestUsage(StrictModel):
    input_tokens: StrictInt = Field(ge=0)
    cache_read_input_tokens: StrictInt = Field(ge=0)
    cache_creation_input_tokens: StrictInt = Field(ge=0)
    output_tokens: StrictInt = Field(ge=0)
    cache_creation: Optional[dict[str, StrictInt]] = None

    @field_validator("cache_creation")
    @classmethod
    def cache_creation_counts_are_safe(cls, value):
        if value is None:
            return None
        allowed = {"ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"}
        if set(value) - allowed or any(count < 0 for count in value.values()):
            raise ValueError("cache_creation contains an unsupported count")
        return value


class IngestStatus(StrictModel):
    outcome: Literal["success", "error", "cancelled", "unknown"]
    code: Optional[StrictInt] = Field(ge=100, le=599)
    error_type: Optional[
        Literal[
            "authentication_error",
            "authorization_error",
            "bad_request",
            "cancelled",
            "network_error",
            "provider_error",
            "rate_limit",
            "timeout",
            "unavailable",
            "unknown",
        ]
    ]

    @model_validator(mode="after")
    def outcome_matches_details(self):
        if self.outcome == "success":
            if self.error_type is not None:
                raise ValueError("a successful event cannot carry an error type")
            if self.code is not None and not 200 <= self.code <= 299:
                raise ValueError("a successful event must carry a 2xx status code")
        elif self.outcome == "error":
            if self.error_type is None:
                raise ValueError("an error event must carry a normalized error type")
            if self.code is not None and 200 <= self.code <= 299:
                raise ValueError("an error event cannot carry a 2xx status code")
        elif self.code is not None and 200 <= self.code <= 299:
            raise ValueError(
                "a cancelled or unknown event cannot carry a 2xx status code"
            )
        return self


class IngestSegment(StrictModel):
    id: str = Field(pattern=r"^hmac:[0-9a-f]{64}$")
    role: str = Field(min_length=1, max_length=64)
    label: str = Field(default="", max_length=256)
    tokens: StrictInt = Field(ge=0)
    cache_marked: StrictBool
    index: StrictInt = Field(ge=0)
    ttl: Optional[str] = Field(max_length=64)


class IngestCollector(StrictModel):
    name: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    source_type: Literal[
        "normalized_trace",
        "litellm",
        "request_bodies",
        "claude_code",
    ]


class IngestEventV1(StrictModel):
    schema_name: Literal["cacheeconomics.ingest-event"] = Field(alias="schema")
    schema_version: Literal[1]
    event_id: str = Field(min_length=1, max_length=256)
    request_id: Optional[str] = Field(default=None, max_length=256)
    sent_at: datetime
    first_token_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    model: str = Field(min_length=1, max_length=256)
    target_id: str = Field(min_length=1, max_length=256)
    agent: str = Field(default="unknown", max_length=256)
    session: Optional[str] = Field(default=None, max_length=256)
    workload_tenant: Optional[str] = Field(default=None, max_length=256)
    ttl_requested: Optional[str] = Field(default=None, max_length=64)
    tokens_counted: StrictBool
    status: IngestStatus
    usage: IngestUsage
    segments: list[IngestSegment] = Field(max_length=256)
    collector: IngestCollector

    @model_validator(mode="after")
    def timestamps_and_segments_are_ordered(self):
        sent = _utc(self.sent_at)
        first = _utc(self.first_token_at) if self.first_token_at else None
        completed = _utc(self.completed_at) if self.completed_at else None
        if first is not None and first < sent:
            raise ValueError("first_token_at cannot precede sent_at")
        if completed is not None and completed < sent:
            raise ValueError("completed_at cannot precede sent_at")
        if first is not None and completed is not None and completed < first:
            raise ValueError("completed_at cannot precede first_token_at")
        indexes = [segment.index for segment in self.segments]
        if indexes != list(range(len(indexes))):
            raise ValueError("segment indexes must be contiguous and ordered from zero")
        self.sent_at = sent
        self.first_token_at = first
        self.completed_at = completed
        return self


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamps must include an explicit UTC offset")
    return value.astimezone(timezone.utc)


class IngestBatchEnvelope(StrictModel):
    events: list[Any] = Field(min_length=1)


class IngestItemResult(StrictModel):
    index: int
    event_id: Optional[str]
    status: Literal["accepted", "duplicate", "rejected"]
    reason: Optional[str] = None


class IngestBatchResponse(StrictModel):
    accepted: int
    duplicates: int
    rejected: int
    job_id: Optional[uuid.UUID]
    items: list[IngestItemResult]


class SourceHealthResponse(StrictModel):
    source_id: uuid.UUID
    accepted_events: int
    duplicate_events: int
    rejected_events: int
    last_received_at: Optional[datetime]
    last_event_sent_at: Optional[datetime]
    last_ingest_lag_seconds: Optional[float]
    last_success_at: Optional[datetime]
    last_error_at: Optional[datetime]
    last_error_type: Optional[str]
    last_job_duration_ms: Optional[int]
    consecutive_failures: int


class JobCreate(StrictModel):
    scheduled_for: Optional[datetime] = None


class JobResponse(StrictModel):
    id: uuid.UUID
    source_id: uuid.UUID
    kind: str
    state: Literal[
        "queued",
        "running",
        "succeeded",
        "failed",
        "dead_letter",
        "cancelled",
    ]
    attempt: int
    max_attempts: int
    scheduled_for: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    error_type: Optional[str]
    created_at: datetime


class AnalysisRecordResponse(StrictModel):
    id: uuid.UUID
    source_id: uuid.UUID
    job_id: Optional[uuid.UUID]
    event_count: int
    result: dict[str, Any]
    created_at: datetime


class TimingSummary(StrictModel):
    count: int
    p50_ms: Optional[float]
    p95_ms: Optional[float]


class OperationsStatusCounts(StrictModel):
    success: int
    error: int
    cancelled: int
    unknown: int


class OperationsErrorCount(StrictModel):
    error_type: str
    count: int


class OperationsBucket(StrictModel):
    started_at: datetime
    requests: int
    errors: int
    cancelled: int
    unknown: int


class SourceOperationsResponse(StrictModel):
    source_id: uuid.UUID
    generated_at: datetime
    window_hours: int
    bucket_minutes: int
    events_examined: int
    valid_events: int
    invalid_events: int
    truncated: bool
    first_received_at: Optional[datetime]
    last_received_at: Optional[datetime]
    status: OperationsStatusCounts
    latency: TimingSummary
    time_to_first_token: TimingSummary
    errors: list[OperationsErrorCount]
    buckets: list[OperationsBucket]


class DashboardConfigResponse(StrictModel):
    oidc_enabled: bool
    authorization_endpoint: Optional[str]
    token_endpoint: Optional[str]
    client_id: Optional[str]
    audience: Optional[str]
    scopes: str
    redirect_uri: Optional[str]
    allow_development_token: bool


class ErrorResponse(StrictModel):
    detail: str
