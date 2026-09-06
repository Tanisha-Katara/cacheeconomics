"""Translate the prompt-free ingest contract into the local analysis engine."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from cacheeconomics.analyzer import analyze
from cacheeconomics.contracts import analysis_payload
from cacheeconomics.trace import Request, Segment, Tier, TraceSet, _billed_input

from .api_schemas import IngestEventV1


def _request_status(event: IngestEventV1) -> int:
    if event.status.outcome == "success":
        return 200
    if event.status.outcome == "cancelled":
        return 499
    if event.status.outcome == "unknown":
        return 500
    if event.status.code is not None:
        return event.status.code
    return 500


def _input_weight(event: IngestEventV1) -> int:
    usage = event.usage.model_dump(exclude_none=True)
    return max(1, _billed_input(usage))


def trace_from_event_payloads(
    payloads: Iterable[Mapping[str, Any]],
    *,
    source: str,
    history_truncated: bool = False,
) -> TraceSet:
    """Build a trace without accepting raw prompt or completion fields."""

    events = [IngestEventV1.model_validate(payload) for payload in payloads]
    requests: list[Request] = []
    structured_weight = 0
    structured_count = 0
    counted_weight = 0
    token_sums_reconciled = True

    for event in events:
        weight = _input_weight(event)
        if event.segments:
            structured_count += 1
            structured_weight += weight
            if event.tokens_counted:
                counted_weight += weight
                measured_input = _billed_input(
                    event.usage.model_dump(exclude_none=True)
                )
                segment_total = sum(segment.tokens for segment in event.segments)
                if measured_input and segment_total != measured_input:
                    token_sums_reconciled = False

        usage = event.usage.model_dump(exclude_none=True)
        requests.append(
            Request(
                request_id=event.request_id or event.event_id,
                sent_at=event.sent_at,
                first_token_at=event.first_token_at,
                model=event.model,
                target_id=event.target_id,
                agent=event.agent,
                session=event.session,
                tenant=event.workload_tenant,
                ttl_requested=event.ttl_requested,
                status=_request_status(event),
                usage=usage,
                segments=[
                    Segment(
                        id=segment.id,
                        role=segment.role,
                        label=segment.label,
                        tokens=segment.tokens,
                        cache_marked=segment.cache_marked,
                        index=segment.index,
                        ttl=segment.ttl,
                    )
                    for segment in event.segments
                ],
            )
        )

    structured = any(event.segments for event in events)
    notes = [
        "Analyzed from the prompt-free cacheeconomics ingest contract; no prompt "
        "or completion content was stored by the service."
    ]
    structural_coverage = structured_count / len(events) if events else 0.0
    if structured and structural_coverage < 1.0:
        notes.append(
            "Some requests contain usage only. Structural recommendations use only "
            "the requests that include locally keyed segment fingerprints."
        )
    if not structured:
        notes.append(
            "No prompt structure was supplied. The result is limited to usage-based "
            "diagnosis; structural counterfactuals are not available."
        )
    if history_truncated:
        notes.append(
            "Older source events were outside this worker's configured analysis "
            "window. Findings describe only the newest retained events in the job snapshot."
        )

    return TraceSet(
        requests=requests,
        tier=Tier.INSTRUMENTED if structured else Tier.USAGE_ONLY,
        source=source,
        notes=notes,
        structural_coverage=structural_coverage,
        tokens_counted=(counted_weight / structured_weight if structured_weight else 0.0),
        token_sums_reconciled=token_sums_reconciled,
        token_sums_publishable=token_sums_reconciled,
    )


def analyze_event_payloads(
    payloads: Iterable[Mapping[str, Any]],
    *,
    source: str,
    history_truncated: bool = False,
) -> dict[str, Any]:
    """Run the unchanged local engine and return its versioned safe contract."""

    trace = trace_from_event_payloads(
        payloads,
        source=source,
        history_truncated=history_truncated,
    )
    result = analyze(trace)
    return analysis_payload(result, source=source)
