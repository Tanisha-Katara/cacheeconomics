"""Segment request bodies a customer already logs, and score how well it worked.

The recorder gives the high-confidence tier, but it needs a code change inside
someone's agent. Nobody makes that change to evaluate a vendor. What they will
do is export the logs their gateway already keeps -- Langfuse, Helicone and the
LiteLLM proxy all store full request bodies -- which is why the inferred tier is
the one a first engagement actually runs on.

The segmentation itself is `recorder.segments_from_request`: the same function,
so a body segmented here and a request recorded live produce identical ids for
identical content. Sharing it is the point. Two implementations would drift, and
the drift would show up as phantom volatility.

What this module adds is the honest part: **an alignment score**.

The trace schema has always reserved a field for it and it has always been
None, with a note saying structural findings were unvalidated. That was the
truthful thing to write while there was nothing to measure against. Now there
is. `score_alignment` re-segments the bodies of a workload that was also
recorded instrumented, and reports how much of the structure the inference
actually recovered. A tier that carries a measured score is a different claim
from one that carries a promise.
"""

from __future__ import annotations

import json
from datetime import datetime

from ..segment import (_billed_input, _requested_ttl, _scale_to_measured,
                       segments_from_request, usage_from_response)

from ..trace import (Segment, Tier, TraceSet, _parse_ts, request_from_row,
                     resolve_tenant)

# Where the request body and the response live in each export shape. Checked in
# order; the first that yields a dict with `messages` wins. Adding a format is a
# row here, not a new code path.
_BODY_PATHS = (
    ("request",),                    # LiteLLM proxy logs, generic
    ("request_body",),               # Helicone
    ("input",),                      # Langfuse observations
    ("body",),
    ("kwargs",),                     # LiteLLM callback payloads
)
_RESPONSE_PATHS = (
    ("response",),
    ("response_body",),
    ("output",),
    ("completion_response",),
)


def _dig(row: dict, path: tuple):
    cur = row
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _find_body(row: dict) -> dict | None:
    """The Anthropic-shaped request in this row, whatever the exporter called it."""
    for path in _BODY_PATHS:
        cand = _dig(row, path)
        if isinstance(cand, str):
            try:
                cand = json.loads(cand)
            except (ValueError, TypeError):
                cand = None
        if isinstance(cand, dict) and ("messages" in cand or "system" in cand):
            return cand
    # Some exporters flatten the body onto the row itself.
    if "messages" in row or "system" in row:
        return row
    return None


def _find_response(row: dict):
    """The object carrying this row's accounting, not merely the first one that
    is shaped like a response.

    Exports disagree about where usage lives. Some put it inside the response
    object, some alongside it with only metadata under `response`. Taking the
    first dict found meant a row with `response: {"id": ...}` and a top-level
    `usage` handed back the metadata, `usage_from_response` correctly reported
    no accounting fields, and eleven thousand billed write tokens were dropped
    -- the request became unanalysable and shrank spend, cache ratios and the
    coverage denominator while the ingest looked clean.

    So candidates are ranked by whether they actually carry accounting. Shape is
    the tie-breaker, never the test.
    """
    fallback = None
    for path in _RESPONSE_PATHS:
        cand = _dig(row, path)
        if isinstance(cand, str):
            try:
                cand = json.loads(cand)
            except (ValueError, TypeError):
                continue
        if not isinstance(cand, dict):
            continue
        if usage_from_response(cand):
            return cand
        if fallback is None:
            fallback = cand
    if "usage" in row and usage_from_response(row):
        return row
    return fallback


# One tolerant parser, shared with the normalised loader. Two copies of the
# same leniency is how one of them ended up without it.
_ts = _parse_ts


def _first_ts(row: dict, *names):
    for n in names:
        got = _ts(row.get(n))
        if got:
            return got
    return None


def load_bodies(path: str, key: bytes, *, tenant: str | None = None,
                target_id: str = "anthropic/direct") -> TraceSet:
    """Load a JSONL export of logged request bodies as an INFERRED trace.

    Requires a key for the same reason the recorder does: a short low-entropy
    segment cannot then be recovered by dictionary attack.

    Keying does not hide cross-tenant equality and never did -- one shared key
    means identical content yields an identical id whoever sent it. `tenant` is
    what scopes that, and it is passed into segmentation below.
    """
    if not key:
        raise ValueError(
            "segmenting bodies needs an HMAC key, for the same reason the recorder "
            "does: a bare digest of a short segment is guessable by anyone holding "
            "a candidate prompt. Pass --tenant as well on a multi-tenant export -- "
            "the key does not scope ids, the tenant does.")

    requests, notes = [], []
    skipped_no_body = unparseable = dropped_no_body = 0
    renamed: dict = {}
    with open(path) as f:
        rows = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if not isinstance(parsed, dict):
                    # Valid JSON that is not an object is not a row. Counted
                    # with the unparseable ones rather than a new counter: to a
                    # reader they are the same gap -- a line the loader could
                    # not use -- and the coverage note already names that.
                    unparseable += 1
                    continue
                rows.append(parsed)
            except json.JSONDecodeError:
                # Counted, not silently dropped. A truncated line in a gateway
                # export used to vanish from both the analysis and the
                # denominator, so the remaining rows were presented as if they
                # were the whole file.
                unparseable += 1

    for i, row in enumerate(rows):
        body = _find_body(row)
        resp = _find_response(row)
        usage = usage_from_response(resp) if resp else {}
        if not body:
            # Kept, not dropped. A row with usage counters and no body is a real
            # request that cost real money; discarding it shrank the denominator
            # so a mixed export could report 100% coverage over the subset that
            # happened to carry bodies, understating ratios and spend while a
            # note took the blame. It survives with no segments, which is what
            # `structural_coverage` is for.
            skipped_no_body += 1
            if not usage:
                # Only *this* branch is a row the loader could not use. The
                # no-body rows that carry usage are kept and priced a few lines
                # down, and counting them as skipped made the reconciliation
                # gate refuse a report whose spend was complete -- an
                # over-block, which is its own kind of wrong answer.
                dropped_no_body += 1
                continue
            segs = []
        else:
            # Resolved with the same rule `request_from_row` uses below, and
            # *before* the ids are hashed. Passing the caller's `tenant`
            # straight through was the bug: an export whose rows each name
            # their own tenant got correct Request.tenant values over ids
            # computed with no tenant at all, so two tenants sending identical
            # prompts shared an id -- the leak the scoping exists to close,
            # reintroduced by the commit that closed it.
            segs = segments_from_request(body, key, resolve_tenant(row, tenant))
            _scale_to_measured(segs, _billed_input(usage) if usage else None)
        requests.append(request_from_row(
            row,
            [Segment(id=sg["id"], role=sg["role"], tokens=sg["tokens"],
                     index=sg["index"], label=sg["label"],
                     cache_marked=sg["cache_marked"], ttl=sg["ttl"])
             for sg in segs],
            renamed=renamed, default_target=target_id, index=i,
            default_tenant=tenant,
            model_override=(body or {}).get("model"), usage_override=usage,
            # `cache_control: {"type": "ephemeral"}` with no ttl is the
            # provider's five-minute default, and the body proves it.
            ttl_fallback=_requested_ttl(body) if body else None))

    if renamed:
        notes.append(
            "Model ids normalised (date snapshots stripped so they price against the "
            "registry): " + ", ".join(f"{k} -> {v}" for k, v in sorted(renamed.items())))

    surfaces = sorted({r.target_id for r in requests})
    if len(surfaces) > 1:
        notes.append(
            f"This export spans {len(surfaces)} API surfaces ({', '.join(surfaces)}). "
            f"Caches do not span them, so each is analysed as its own pool; a comparison "
            f"across them is not a comparison of one cache.")

    segmented = sum(1 for r in requests if r.segments)
    notes.append(
        f"{segmented:,} of {len(requests):,} requests carried a body and were segmented "
        f"post hoc. Segment boundaries are inferred from the logged body, not recorded "
        f"at source.")
    notes.append(
        "Token counts are proportional estimates scaled to the billed input total, not "
        "counted. Structural findings inherit that.")
    if unparseable:
        notes.append(
            f"{unparseable} line(s) could not be parsed as JSON and are not represented "
            f"anywhere below. Figures describe the {len(requests):,} rows that did parse; "
            f"if that shortfall is material, re-export before relying on them.")
    if skipped_no_body:
        notes.append(
            f"{skipped_no_body} row(s) carried no recognisable request body. Those with "
            f"usage counters are still counted -- they cost money and belong in the "
            f"denominator -- but they cannot take part in a counterfactual.")

    return TraceSet(requests=requests, tier=Tier.INFERRED, alignment=None,
                    source=path, notes=notes,
                    structural_coverage=(segmented / len(requests)) if requests else 0.0,
                    skipped_rows=unparseable + dropped_no_body)


# --- alignment -------------------------------------------------------------

def score_alignment(truth: TraceSet, inferred: TraceSet) -> dict:
    """How much of the real structure the post-hoc segmentation recovered.

    Compares an instrumented capture against the same workload segmented from
    its bodies. Reported per request and averaged, using the Jaccard overlap of
    the segment-id sets: it penalises both a boundary the inference invented and
    one it missed, which a simple count would not.

    This is the number the INFERRED tier has always reserved a field for and
    never had. Without it, every structural finding on inferred data rests on an
    assumption nobody measured; with it, the finding carries its own confidence.
    """
    truth_by_id = {r.request_id: r for r in truth.requests}
    inf_by_id = {r.request_id: r for r in inferred.requests}

    # Score over the union of request ids, not the inferred side alone.
    # Iterating the inferred requests meant an export containing one row could
    # match one row of a fifty-row instrumented capture and report 1.0 -- the
    # rows it never covered simply were not in the denominator. Measured: that
    # exact case returned a perfect score.
    all_ids = set(truth_by_id) | set(inf_by_id)
    scores, marker_scores, matched, compared = [], [], 0, 0
    # Billed input tokens per request, so alignment can be weighted by the money
    # each request represents as well as averaged per row. A plain mean treats a
    # 200-token request and a 900,000-token one as equal evidence: measured, nine
    # perfectly-segmented tiny requests beside one large one segmented entirely
    # wrong scored exactly 0.90 and cleared the floor, while the request that was
    # ~100% of the bill had zero overlap. Same defect as row-counted structural
    # coverage, one module along.
    weights = []

    def _seg_keys(r):
        # Position and container are part of identity here, and the tuple is
        # per-index, so a segmentation that collapses two identical adjacent
        # segments into one no longer looks equal. Comparing bare id sets
        # discarded both multiplicity and position, and scored that collapse
        # as perfect.
        return {(sg.index, sg.role, sg.id) for sg in r.segments}

    for rid in sorted(all_ids):
        t, r = truth_by_id.get(rid), inf_by_id.get(rid)
        side = t if t is not None else r
        weights.append(_billed_input(side.usage)
                       if side is not None and isinstance(side.usage, dict) else 0)
        if t is None or r is None:
            # Present on one side only: nothing was recovered for that request,
            # which is a zero rather than an absence.
            scores.append(0.0)
            marker_scores.append(0.0)
            continue
        compared += 1
        a, b = _seg_keys(t), _seg_keys(r)
        overlap = 1.0 if not (a or b) else len(a & b) / len(a | b)
        scores.append(overlap)
        if overlap == 1.0:
            matched += 1

        # Markers are scored separately because segment identity deliberately
        # excludes cache_control -- a marker is an instruction to the cache,
        # not content, and hashing it would make configuration changes read as
        # prompt drift. The consequence is that an identity score cannot see
        # marker loss: a gateway that strips cache_control produces bodies whose
        # segments match perfectly and whose caching configuration is gone.
        ma = {(sg.index, sg.ttl) for sg in t.segments if sg.cache_marked}
        mb = {(sg.index, sg.ttl) for sg in r.segments if sg.cache_marked}
        marker_scores.append(1.0 if not (ma or mb) else len(ma & mb) / len(ma | mb))

    mean = sum(scores) / len(scores) if scores else None
    marker_mean = sum(marker_scores) / len(marker_scores) if marker_scores else None
    coverage = (compared / len(all_ids)) if all_ids else 0.0
    # The same scores, weighted by what each request cost. Where no request
    # carries billed input there is no money to misdescribe, so the plain mean
    # stands rather than the weighting inventing a verdict from nothing.
    billed = sum(weights)
    weighted = (sum(s * w for s, w in zip(scores, weights)) / billed
                if billed else mean)
    combined = None if mean is None else min(
        mean, weighted if weighted is not None else mean,
        marker_mean if marker_mean is not None else mean, coverage)
    return {
        "compared": compared,
        "request_coverage": coverage,
        "segment_alignment": mean,
        "segment_alignment_billed": weighted,
        "marker_alignment": marker_mean,
        # A structural claim is only as good as its weakest dimension, so this
        # is the number findings should inherit.
        "mean_alignment": combined,
        "exact_matches": matched,
        "unmatched_requests": len(all_ids) - compared,
        "note": ("Scored over the union of request ids, and reported as the lowest of "
                 "four dimensions rather than the friendliest. Per request: Jaccard overlap "
                 "of (index, role, id) segment keys, and separately of (index, ttl) for "
                 "marked segments. The reported alignment is the lowest of segment "
                 "identity, marker recovery, request coverage and the same identity score "
                 "weighted by billed tokens, because each can be perfect while another "
                 "is not: a partial export covers few requests flawlessly, a body whose "
                 "markers were stripped in logging matches on identity while carrying "
                 "none of the caching configuration, and a mean over requests can sit "
                 "above the floor while the one request holding the money was segmented "
                 "wrong."),
    }


def with_alignment(inferred: TraceSet, score: dict) -> TraceSet:
    """Attach a measured alignment score, so findings can inherit it."""
    inferred.alignment = score.get("mean_alignment")
    if inferred.alignment is not None:
        seg, mark = score.get("segment_alignment"), score.get("marker_alignment")
        cov = score.get("request_coverage")
        inferred.notes.append(
            f"Segmentation alignment {inferred.alignment:.1%}, measured against an "
            f"instrumented capture ({score['compared']} of "
            f"{score['compared'] + score['unmatched_requests']} requests matched: "
            f"segment identity {seg:.1%}, cache markers "
            f"{'n/a' if mark is None else format(mark, '.1%')}, request coverage "
            f"{cov:.1%}).")
        if cov is not None and cov < 1.0:
            inferred.notes.append(
                "Some requests appear on only one side of the comparison, so the "
                "segmentation has not been checked against the whole workload.")
        if mark is not None and mark < 1.0:
            inferred.notes.append(
                "Cache markers were not fully recovered from the logged bodies. The "
                "as-shipped arm of any bake-off describes the markers this export "
                "preserved, not necessarily the ones that were sent.")
    return inferred
