"""Read Claude Code session transcripts as a trace.

The first real workload this tool has seen. Everything before it was synthetic
with the failures planted by hand, which proves the code runs and proves nothing
about whether the findings mean anything.

Why this source: it is a genuine long-running tool-calling agent, it is the
user's own data, it never leaves the machine, and it is a *hard* target rather
than a flattering one — Claude Code is already cache-optimised by the people who
built the cache. Beating a tuned baseline is the credible test; beating an
untuned one is a demo.

What this tier can and cannot answer. Transcripts record the conversation, not
the wire request: the system prompt and tool definitions — the large stable
prefix that caching is mostly about — are absent. So this is USAGE_ONLY.
Measured ratios, real spend, real cadence, real lifetime split. No counterfactual
and no bake-off, because a counterfactual needs structure that is not here. The
honest read is that this validates the measurement half of the tool, not the
allocator half.

Privacy, stated precisely rather than flatteringly. An earlier version of this
docstring claimed prompt text was "never loaded into memory at all". That was
false: `json.loads` parses each transcript line in full, so `message.content` --
prompt text and tool output -- is materialised in memory before the fields below
are selected.

What is actually true, and is what matters for a local-first diagnostic:

  - Content is read and immediately discarded. Only usage counters, timestamps,
    model ids and record types are retained on the Request objects.
  - Nothing is hashed, written to disk, or transmitted. This module has no
    network calls and writes no files.
  - Analysis runs entirely on the user's own machine, against their own
    transcripts.

The exposure is therefore the parse itself, which is unavoidable without a
selective JSON reader. If that boundary matters for a given engagement, the
answer is a pre-scrubbed export rather than a stronger claim about this path.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from glob import glob

from ..registry import normalize_model
from ..trace import Request, Tier, TraceSet, _parse_ts

DEFAULT_ROOT = os.path.expanduser("~/.claude/projects")

# How many lines one record may span before the reader gives up on joining them.
# Transcript records carry raw newlines inside string values, so a record is not
# always a line; without a cap, one genuinely broken line would absorb every line
# after it and report the whole file as a single unreadable record.
_MAX_RECORD_LINES = 200

# The only fields retained from a transcript record. The record is parsed in
# full -- see the module docstring -- but nothing outside this list survives the
# call, and none of it is written or transmitted.
READ_FIELDS = ("type", "timestamp", "requestId", "sessionId", "isSidechain",
               "slug", "gitBranch", "message.model", "message.usage")


def _usage_only(u: dict) -> dict:
    """Copy the accounting fields and nothing else.

    `iterations` is dropped deliberately: it repeats the same totals per
    internal step, and summing it would double-count a single billed request.
    """
    return {
        "input_tokens": u.get("input_tokens", 0) or 0,
        "cache_read_input_tokens": u.get("cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": u.get("cache_creation_input_tokens", 0) or 0,
        "output_tokens": u.get("output_tokens", 0) or 0,
        "cache_creation": dict(u.get("cache_creation") or {}),
    }


def load_sessions(root: str = DEFAULT_ROOT, project: str | None = None,
                  limit: int | None = None) -> TraceSet:
    """Every assistant turn across the transcripts under `root`.

    One assistant record is one billed request. User records, attachments,
    file-history snapshots and the rest carry no usage and are skipped.
    """
    # Recursive: subagent transcripts live in per-session subdirectories, and a
    # flat glob silently missed 73 of 122 files here — which would have dropped
    # exactly the subagent traffic that is most worth looking at separately.
    pattern = os.path.join(root, project or "**", "**", "*.jsonl")
    paths = sorted(set(glob(pattern, recursive=True)))
    if limit:
        paths = paths[-limit:]

    requests, notes, skipped, sessions = [], [], 0, set()
    corrupt = 0
    synthetic, untimed, renamed = 0, 0, {}
    for path in paths:
        try:
            fh = open(path)
        except OSError:
            skipped += 1
            continue
        with fh:
            # Read up front so the *last* line can be told from the rest. Only a
            # trailing unparseable line is plausibly a live partial write; one in
            # the middle is a corrupt or truncated record, and if it was a billed
            # assistant turn its spend vanished from usage, from coverage and
            # from the reconciliation denominator -- letting a parsed subset
            # inside the 5% window release figures over an unknown total.
            #
            # These transcripts are the one input this tool reads while the
            # writer may still be appending, which is why the trailing case gets
            # the benefit of the doubt and no other line does.
            lines = fh.read().splitlines()
        # Records can span several lines: the writer emits raw newlines inside
        # string values, which is not strict JSON but is perfectly readable if
        # the lines are joined. A line-at-a-time reader sees the first fragment
        # as "unterminated string" and each continuation as garbage.
        #
        # Counting those fragments as lost data was an over-block I shipped: on
        # this machine's own transcripts it flagged hundreds of lines, of which
        # the recoverable ones turned out to be `user` turns carrying no usage
        # at all. A gate that fires on correct input is not a safer gate.
        #
        # So: accumulate until a record parses, and only count what cannot be
        # recovered even by joining. `strict=False` is what tolerates the
        # embedded newline. The window is capped so one genuinely broken line
        # cannot swallow the rest of the file.
        buf: list = []
        for line in lines:
            if not line.strip() and not buf:
                continue
            # A record starts at `{`. If one starts while the buffer is still
            # open, whatever was in the buffer never completed and is not
            # recoverable -- that is the corruption worth counting, and it is
            # decided by structure rather than by a line budget. Capping instead
            # meant a truly broken line only counted in a long enough file.
            if buf and line.lstrip().startswith("{"):
                corrupt += 1
                buf = []
            buf.append(line)
            try:
                rec = json.loads("\n".join(buf), strict=False)
            except json.JSONDecodeError:
                if len(buf) >= _MAX_RECORD_LINES:
                    corrupt += 1
                    buf = []
                continue
            buf = []
            if rec.get("type") != "assistant":
                continue
            msg = rec.get("message") or {}
            usage = msg.get("usage")
            if not usage:
                continue
            # Synthetic records are locally generated assistant turns —
            # interrupts, errors, cancellations — that were never sent and
            # never billed. Counting them would dilute every ratio with
            # requests the provider never saw.
            raw_model = msg.get("model", "unknown")
            if raw_model == "<synthetic>":
                synthetic += 1
                continue

            model, stamp = normalize_model(raw_model)
            if stamp:
                renamed[raw_model] = model

            # Tolerant, and shared with the other loaders. This used to
            # skip any billed row with no timestamp and call
            # datetime.fromisoformat directly on whatever was there --
            # so a row the provider charged for either vanished from the
            # denominator with nothing in the coverage line to say so, or
            # one malformed stamp aborted the entire load. An untimed row
            # is still a real cost; it just cannot take part in anything
            # that needs ordering, and the notes say how many.
            sent_at = _parse_ts(rec.get("timestamp"))
            if sent_at is None:
                untimed += 1

            sessions.add(rec.get("sessionId"))
            requests.append(Request(
                request_id=rec.get("requestId") or rec.get("uuid", ""),
                sent_at=sent_at,
                model=model,
                usage=_usage_only(usage),
                segments=[],                  # no wire structure in a transcript
                agent=_agent_of(rec),
                session=rec.get("sessionId"),
                target_id="anthropic/direct",
            ))

    if untimed:
        notes.append(
            f"{untimed:,} billed turns carry no usable timestamp. They are kept, "
            f"because the provider charged for them and dropping them would "
            f"understate spend, but nothing that needs ordering -- cadence, TTL "
            f"selection, fan-out, rebuild counting -- can use them.")
    notes.append(
        f"{len(requests):,} assistant turns from {len(sessions):,} sessions across "
        f"{len(paths):,} transcripts. Each is one billed request.")
    notes.append(
        "Transcripts record the conversation, not the wire request. The system prompt "
        "and tool definitions are absent, so prefix structure cannot be recovered and "
        "no counterfactual is derivable. Ratios, spend and cadence are measured.")
    if synthetic:
        notes.append(
            f"{synthetic} synthetic assistant record(s) excluded: locally generated "
            f"interrupts and errors that were never sent to the API and never billed.")
    if renamed:
        notes.append(
            "Model ids normalised (date snapshots stripped so they price against the "
            "registry): " + ", ".join(f"{k} -> {v}" for k, v in sorted(renamed.items())))
    if skipped:
        notes.append(f"{skipped} transcript(s) could not be read and were skipped.")

    # A transcript that could not be opened is traffic this report does not
    # describe, exactly like an unparseable JSONL line. The count existed and
    # went only into `notes`, which the reconciliation gate cannot read -- so a
    # readable subset matching the invoice released figures while the notes
    # beside them said a whole transcript was missing.
    return TraceSet(requests=requests, tier=Tier.USAGE_ONLY,
                    source=pattern, notes=notes,
                    skipped_rows=skipped + corrupt)


def _agent_of(rec: dict) -> str:
    """Group by what was actually doing the work.

    Subagents are the interesting case: they run their own context, so their
    cache behaviour is independent of the main loop's and a blended number hides
    them.
    """
    if rec.get("isSidechain"):
        return f"subagent:{rec.get('slug') or 'unnamed'}"
    return "main-loop"
