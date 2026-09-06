from __future__ import annotations

import pytest

from cacheeconomics_control_plane.analysis_engine import trace_from_event_payloads

from test_ingestion import ingest_event


def _segment(tokens: int) -> list[dict]:
    return [
        {
            "id": "hmac:" + "a" * 64,
            "role": "system",
            "label": "policy",
            "tokens": tokens,
            "cache_marked": True,
            "index": 0,
            "ttl": "5m",
        }
    ]


def test_split_only_cache_writes_count_toward_the_token_count_gate():
    split_only = ingest_event(
        "split-only",
        tokens_counted=False,
        usage={
            "input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "output_tokens": 1,
            "cache_creation": {"ephemeral_5m_input_tokens": 9_000},
        },
        segments=_segment(9_000),
    )
    counted = ingest_event(
        "counted",
        tokens_counted=True,
        usage={
            "input_tokens": 100,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "output_tokens": 1,
        },
        segments=_segment(100),
    )

    trace = trace_from_event_payloads(
        [split_only, counted],
        source="test",
    )

    assert trace.tokens_counted == pytest.approx(100 / 9_100)
    assert trace.tokens_are_counted is False


def test_split_only_cache_writes_participate_in_segment_reconciliation():
    event = ingest_event(
        "split-mismatch",
        tokens_counted=True,
        usage={
            "input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "output_tokens": 1,
            "cache_creation": {"ephemeral_1h_input_tokens": 9_000},
        },
        segments=_segment(1_000),
    )

    trace = trace_from_event_payloads([event], source="test")

    assert trace.token_sums_reconciled is False
    assert trace.token_sums_publishable is False


def test_structural_coverage_counts_rows_not_tokens():
    large_structured = ingest_event(
        "large-structured",
        usage={
            "input_tokens": 1_000_000,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "output_tokens": 1,
        },
        segments=_segment(1_000_000),
    )
    small_usage_only = [
        ingest_event(
            f"usage-only-{index}",
            tokens_counted=False,
            usage={
                "input_tokens": 1,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "output_tokens": 1,
            },
            segments=[],
        )
        for index in range(9)
    ]

    trace = trace_from_event_payloads(
        [large_structured, *small_usage_only],
        source="test",
    )

    assert trace.structural_coverage == pytest.approx(0.1)
    assert trace.structural_coverage_billed > 0.99


@pytest.mark.parametrize(
    ("outcome", "expected_status"),
    [("cancelled", 499), ("unknown", 500)],
)
def test_non_success_outcomes_do_not_enter_successful_analysis(
    outcome, expected_status
):
    event = ingest_event(
        f"{outcome}-event",
        status={"outcome": outcome, "code": None, "error_type": None},
    )

    trace = trace_from_event_payloads([event], source="test")

    assert trace.requests[0].status == expected_status
    assert trace.analysable == []
