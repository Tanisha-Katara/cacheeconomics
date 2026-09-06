"""Versioned, privacy-safe contracts for applications built around cacheeconomics.

This module only transforms local objects into plain Python data. It performs no
I/O beyond reading the package's bundled registry files and never opens sockets.
Hosted services should depend on this contract; the core package must not depend
on a hosted service.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional

from . import money, registry
from .analyzer import Analysis
from .trace import Tier


ANALYSIS_SCHEMA = "cacheeconomics.analysis-result"
ANALYSIS_SCHEMA_VERSION = 1
INGEST_SCHEMA = "cacheeconomics.ingest-event"
INGEST_SCHEMA_VERSION = 1


def _engine_version() -> str:
    # Import lazily so this module can remain independent of package init order.
    from . import __version__

    return __version__


def _finite_number(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return value if math.isfinite(value) else None


def _plain_value(value: Any) -> Any:
    """Convert supported analysis values into strict JSON-compatible values."""

    if isinstance(value, money.Figure):
        return figure_payload(value)
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, float):
        return _finite_number(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError("unsupported contract value: %s" % type(value).__name__)


def figure_payload(figure: money.Figure) -> Dict[str, Any]:
    """Serialize a monetary figure without bypassing its release gate.

    A withheld figure deliberately has no numeric ``amount_usd``. Callers get a
    display string and the reason it was withheld, but never the hidden amount.
    """

    if not isinstance(figure, money.Figure):
        raise TypeError("figure must be a cacheeconomics.money.Figure")

    released = figure.released
    return {
        "display": str(figure),
        "amount_usd": _finite_number(figure.amount) if released else None,
        "basis": figure.basis,
        "released": released,
        "release_state": figure.released_as if released else "withheld",
        "projected": figure.projected,
        "withheld_because": None if released else figure.withheld_because,
    }


def registry_snapshot() -> Dict[str, Any]:
    """Identify the exact bundled registry used for an analysis."""

    digest = hashlib.sha256()
    for name in ("providers.json", "pricing.json"):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        with open(os.path.join(registry.REGISTRY_DIR, name), "rb") as handle:
            digest.update(handle.read())
        digest.update(b"\0")

    providers = registry.providers()
    pricing = registry.pricing()
    return {
        "sha256": digest.hexdigest(),
        "providers_generated": providers.get("generated"),
        "pricing_generated": pricing.get("generated"),
    }


def _analysis_figures(analysis: Analysis) -> Iterable[money.Figure]:
    for value in analysis.spend.values():
        if isinstance(value, money.Figure):
            yield value
    for finding in analysis.findings:
        if isinstance(finding.avoidable_usd_window, money.Figure):
            yield finding.avoidable_usd_window
        if isinstance(finding.avoidable_usd_month, money.Figure):
            yield finding.avoidable_usd_month
    if analysis.reconciliation:
        for value in analysis.reconciliation.values():
            if isinstance(value, money.Figure):
                yield value
    yield analysis.total_avoidable_month


def analysis_payload(
    analysis: Analysis,
    *,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the complete, safe v1 representation of a local analysis.

    This is intended for a future service boundary and dashboard. It is separate
    from the existing CLI JSON so current users keep compatible output unless
    they explicitly adopt this contract.
    """

    tier_value = (
        analysis.tier.value
        if isinstance(analysis.tier, Tier)
        else str(analysis.tier).lower()
    )
    figures = list(_analysis_figures(analysis))

    findings = []
    for finding in analysis.findings:
        findings.append(
            {
                "code": finding.code,
                "title": finding.title,
                "severity": finding.severity,
                "confidence": finding.confidence,
                "evidence_class": finding.evidence_class,
                "quality_risk": finding.quality_risk,
                "detail": finding.detail,
                "fix": finding.fix,
                "affected_requests": finding.affected_requests,
                "structural": finding.structural,
                "avoidable_usd_window": (
                    figure_payload(finding.avoidable_usd_window)
                    if finding.avoidable_usd_window is not None
                    else None
                ),
                "avoidable_usd_month": (
                    figure_payload(finding.avoidable_usd_month)
                    if finding.avoidable_usd_month is not None
                    else None
                ),
            }
        )

    return {
        "schema": ANALYSIS_SCHEMA,
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "engine": {"name": "cacheeconomics", "version": _engine_version()},
        "registry": registry_snapshot(),
        "analysis": {
            "tier": tier_value,
            "source": source,
            "coverage": _plain_value(analysis.coverage),
            "window_days": _finite_number(analysis.window_days),
            "ratios": _plain_value(analysis.ratios),
            "spend": _plain_value(analysis.spend),
            "total_avoidable_usd_month": figure_payload(
                analysis.total_avoidable_month
            ),
            "reconciliation": _plain_value(analysis.reconciliation),
            "findings": findings,
            "notes": list(analysis.notes),
            "caveats": list(analysis.blocking_notes),
            "tokens_counted": bool(analysis.tokens_counted),
            "draft": any(figure.released_as == "draft" for figure in figures),
            "has_withheld_figures": any(not figure.released for figure in figures),
        },
    }


def analysis_payload_json(
    analysis: Analysis,
    *,
    source: Optional[str] = None,
    indent: Optional[int] = None,
) -> str:
    """Encode :func:`analysis_payload` as strict JSON (never NaN/Infinity)."""

    return json.dumps(
        analysis_payload(analysis, source=source),
        indent=indent,
        sort_keys=False,
        allow_nan=False,
    )
