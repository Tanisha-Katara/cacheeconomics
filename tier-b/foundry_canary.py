"""Measure prompt-cache behaviour for Claude on Microsoft Foundry.

The registry describes eight surfaces and Foundry is not one of them, so a client
running there gets no answer at all. This is the smallest experiment that turns
that into a sourced row: does caching work, is the per-lifetime split reported,
does the 1h lifetime exist, and what is the minimum cacheable prefix.

**It does not assume the API shape.** Foundry can front a model with the
provider's native API or with Azure's own inference API, and which one applies to
Claude is exactly the sort of thing this project refuses to guess about. So the
first thing the canary does is *discover* which request shape the deployment
accepts, and report it. A run that discovers nothing is still a result: it says
the shape is none of the ones tried, and names what it sent.

**It never prints the key.** The value is read from the environment and passed to
`urllib`; it is not logged, not echoed, and not written to the raw output. HTTP
error bodies are truncated and scrubbed for anything key-shaped before display,
because a 401 body has been known to quote the credential back.

**Nothing here writes to the registry.** The output is a *draft* row for a human
to check against documentation before it becomes a fact anyone publishes. A
measured row still needs a source; that is the whole rule.

Usage:

    export AZURE_AI_ENDPOINT=...      # in ~/.zshenv, not on the command line
    export AZURE_AI_API_KEY=...
    export AZURE_AI_DEPLOYMENT=...

    python3 tier-b/foundry_canary.py --plan     # what it would send, and cost
    python3 tier-b/foundry_canary.py --run      # actually call the API
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ANTHROPIC_VERSION = "2023-06-01"

# Long enough to clear every published minimum (the largest is 4096) with room
# to spare, and small enough that the whole run costs cents.
BIG_PREFIX_TOKENS = 6000

# Deliberately under the smallest published minimum, to see whether the surface
# silently declines to cache rather than erroring -- the failure this whole
# project exists to surface.
TINY_PREFIX_TOKENS = 200

_BYTES_PER_TOKEN = 3.6

# Azure rejects an unknown `api-version` with a 400 rather than ignoring it, and
# a 400 "API version not supported" is *good news*: it means the path resolved
# and only the version was wrong. Distinguishing that from a 404 is the whole
# value of trying several. Newest first; `None` omits the parameter entirely,
# which some surfaces require.
_API_VERSIONS = (None, "2024-10-21", "2025-01-01-preview", "2026-05-01")


def _scrub(text: str, key: str | None) -> str:
    """Remove anything that looks like the credential from text we may print.

    A 401 body has been observed quoting the offending key back, and this output
    goes to a terminal, a log and possibly a screenshot.
    """
    out = text
    if key:
        out = out.replace(key, "[redacted]")
        # Also catch a prefix of it, which some services echo.
        if len(key) > 12:
            out = out.replace(key[:12], "[redacted]")
    return re.sub(r"[A-Za-z0-9_\-]{40,}", "[redacted]", out)


def _prefix(tokens: int, run_id: str) -> str:
    """Filler that is stable within a run and unique across runs.

    Unique across runs because a prefix reused from a previous run could still be
    cached, which would report a hit the run did not earn.
    """
    head = f"Foundry cache canary run {run_id}. "
    body = "The quick brown fox jumps over the lazy dog. "
    return head + body * max(1, int(tokens * _BYTES_PER_TOKEN / len(body)))


# Each candidate is (name, url suffix, headers builder, payload builder, usage
# extractor). Tried in order; the first that returns 200 wins.
def _candidates(endpoint: str, deployment: str, key: str):
    base = endpoint.rstrip("/")
    # Endpoints arrive in several shapes and the version segment is sometimes
    # already in them. Appending `/v1/messages` to a root that already ends in
    # `/openai/v1` produced `/openai/v1/v1/messages` and a 404 that looked like
    # "the surface does not support this API" rather than "I built a bad URL".
    #
    # So: try the endpoint as given, and also each meaningful truncation of it.
    # Order matters only in that the first 200 wins; every attempt is recorded.
    roots = [base]
    for suffix in ("/openai/v1", "/openai", "/v1"):
        if base.endswith(suffix):
            roots.append(base[: -len(suffix)])
    if "/api/projects/" in base:
        roots.append(base.split("/api/projects/")[0])
    deduped, seen = [], set()
    for r in roots:
        if r not in seen:
            seen.add(r)
            deduped.append(r)
    roots = deduped

    def anthropic_payload(prefix, ttl, max_tokens):
        cc = {"type": "ephemeral"}
        if ttl == "1h":
            cc["ttl"] = "1h"
        return {
            "model": deployment,
            "max_tokens": max_tokens,
            "system": [{"type": "text", "text": prefix, "cache_control": cc}],
            "messages": [{"role": "user", "content": "."}],
        }

    def openai_payload(prefix, ttl, max_tokens):
        # Azure's inference API is OpenAI-shaped and has no cache_control field.
        # Sent anyway, because "the surface accepts the request but ignores the
        # marker" is a finding: it means caching cannot be controlled here.
        return {
            "model": deployment,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": prefix},
                         {"role": "user", "content": "."}],
        }

    def q(ver):
        return f"?api-version={ver}" if ver else ""

    for root in roots:
      for ver in _API_VERSIONS:
          # Both with and without the version segment, since `root` may already
          # carry it.
          yield ("anthropic-native", f"{root}/messages{q(ver)}",
                 {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION,
                  "content-type": "application/json"},
                 anthropic_payload)
          yield ("anthropic-native-v1", f"{root}/v1/messages{q(ver)}",
                 {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION,
                  "content-type": "application/json"},
                 anthropic_payload)
          yield ("anthropic-native-bearer", f"{root}/v1/messages{q(ver)}",
                 {"Authorization": f"Bearer {key}",
                  "anthropic-version": ANTHROPIC_VERSION,
                  "content-type": "application/json"},
                 anthropic_payload)
          yield ("azure-inference", f"{root}/models/chat/completions{q(ver)}",
                 {"api-key": key, "content-type": "application/json"},
                 openai_payload)
          yield ("azure-openai-deployment",
                 f"{root}/deployments/{deployment}/chat/completions{q(ver)}",
                 {"api-key": key, "content-type": "application/json"},
                 openai_payload)
          yield ("azure-chat", f"{root}/chat/completions{q(ver)}",
                 {"api-key": key, "content-type": "application/json"},
                 openai_payload)


def _post(url: str, headers: dict, payload: dict, key: str) -> tuple:
    """Returns (status, body_or_error). Never surfaces the key."""
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, {"error": _scrub(e.read().decode("utf-8", "replace")[:500], key)}
    except Exception as e:                                        # noqa: BLE001
        return 0, {"error": _scrub(f"{type(e).__name__}: {e}", key)}


def _usage(body: dict) -> dict:
    """Normalise whichever usage shape came back, without inventing one."""
    u = body.get("usage") or {}
    if not isinstance(u, dict):
        return {}
    out = {
        "input_tokens": u.get("input_tokens"),
        "cache_read_input_tokens": u.get("cache_read_input_tokens"),
        "cache_creation_input_tokens": u.get("cache_creation_input_tokens"),
        "cache_creation": u.get("cache_creation"),
        # OpenAI-shaped surfaces report a cached count and nothing else.
        "prompt_tokens": u.get("prompt_tokens"),
        "prompt_tokens_details": u.get("prompt_tokens_details"),
    }
    return {k: v for k, v in out.items() if v is not None}


def discover(endpoint, deployment, key, records):
    """Find a request shape this deployment accepts. Returns it, or None."""
    probe = _prefix(TINY_PREFIX_TOKENS, "discovery")
    for name, url, headers, build in _candidates(endpoint, deployment, key):
        status, body = _post(url, headers, build(probe, "5m", 1), key)
        records.append({"phase": "discover", "shape": name,
                        "url": _scrub(url, key), "status": status,
                        "error": body.get("error"),
                        "ts": datetime.now(timezone.utc).isoformat()})
        shown = url.split("?")[0]
        if status == 200:
            print(f"  {name:26} {shown}  -> 200 OK")
            return name, url, headers, build
        # Flattened: a pretty-printed error body wrapped across lines and made
        # the one informative response look like three unrelated ones.
        err = " ".join(str(body.get("error") or "").split())[:80]
        print(f"  {name:26} {shown}  -> {status or 'no response'}"
              f"{' ' + err if err else ''}")
    return None


def probe(shape, key, prefix, ttl, tag, records, max_tokens=1):
    name, url, headers, build = shape
    sent = time.time()
    status, body = _post(url, headers, build(prefix, ttl, max_tokens), key)
    rec = {"phase": "probe", "tag": tag, "ttl": ttl, "shape": name,
           "status": status, "latency_s": round(time.time() - sent, 3),
           "ts": datetime.now(timezone.utc).isoformat()}
    if status == 200:
        rec["usage"] = _usage(body)
    else:
        rec["error"] = body.get("error")
    records.append(rec)
    return rec


def run(endpoint, deployment, key):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    records: list = []

    print("Discovering which request shape this deployment accepts:")
    shape = discover(endpoint, deployment, key, records)
    if shape is None:
        print("\nNo shape accepted. That is the result, not a crash: the raw "
              "attempts are recorded below and the registry row stays unwritten.")
        return run_id, records, None

    big = _prefix(BIG_PREFIX_TOKENS, run_id)
    tiny = _prefix(TINY_PREFIX_TOKENS, run_id)

    print("\nProbing cache behaviour:")
    plan = [
        ("write-5m", big, "5m", "first send of a large prefix: does it write?"),
        ("read-5m", big, "5m", "same prefix immediately: does it read back?"),
        ("write-1h", big + " 1h", "1h", "is the one-hour lifetime accepted?"),
        ("read-1h", big + " 1h", "1h", "does the 1h entry read back?"),
        ("below-min", tiny, "5m", "under every published minimum: silent no-cache?"),
    ]
    for tag, text, ttl, why in plan:
        rec = probe(shape, key, text, ttl, tag, records)
        u = rec.get("usage") or {}
        if rec["status"] != 200:
            print(f"  {tag:10} FAILED {str(rec.get('error'))[:70]}")
            continue
        print(f"  {tag:10} in={u.get('input_tokens', u.get('prompt_tokens', '?'))} "
              f"write={u.get('cache_creation_input_tokens', '-')} "
              f"read={u.get('cache_read_input_tokens', '-')} "
              f"split={'yes' if u.get('cache_creation') else 'no'}   ({why})")
    return run_id, records, shape


def draft_row(records) -> dict:
    """A registry row for a human to check. Never written automatically."""
    ok = [r for r in records if r.get("phase") == "probe" and r.get("status") == 200]
    by_tag = {r["tag"]: (r.get("usage") or {}) for r in ok}

    def wrote(tag):
        return (by_tag.get(tag, {}).get("cache_creation_input_tokens") or 0) > 0

    def read(tag):
        return (by_tag.get(tag, {}).get("cache_read_input_tokens") or 0) > 0

    ttls = []
    if wrote("write-5m") or read("read-5m"):
        ttls.append("5m")
    if wrote("write-1h") or read("read-1h"):
        ttls.append("1h")

    return {
        "id": "microsoft-foundry/claude",
        "capabilities": {
            "explicit_breakpoints": bool(ttls),
            "supported_ttls": ttls,
            "automatic_prefix_cache": None,      # not probed
            "caching_observed": bool(ttls),
            "per_lifetime_split_reported": any(
                by_tag.get(t, {}).get("cache_creation") for t in by_tag),
            "below_minimum_silently_uncached":
                (not wrote("below-min")) if "below-min" in by_tag else None,
        },
        "provenance": {
            "checked_on": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "verified_by": {"docs": False, "fixture": False, "live_canary": True},
            "confidence": "medium",
            "contested": False,
            "note": "Measured by tier-b/foundry_canary.py against one deployment. "
                    "A single deployment is not the surface: minimums and TTL "
                    "support can differ per model, and this probed one. Confirm "
                    "against Microsoft's documentation before publishing, and "
                    "record that source here.",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="store_true",
                    help="actually call the API (costs a few cents)")
    ap.add_argument("--plan", action="store_true",
                    help="show what would be sent, and stop")
    args = ap.parse_args()

    missing = [v for v in ("AZURE_AI_ENDPOINT", "AZURE_AI_API_KEY",
                           "AZURE_AI_DEPLOYMENT") if not os.environ.get(v)]
    if missing:
        # Names only. The point of reading these from the environment is that
        # they never appear in a terminal, and an error message is a terminal.
        return _fail(f"not set: {', '.join(missing)}. Export them in ~/.zshenv.")

    endpoint = os.environ["AZURE_AI_ENDPOINT"]
    deployment = os.environ["AZURE_AI_DEPLOYMENT"]
    key = os.environ["AZURE_AI_API_KEY"]

    print(f"endpoint   {endpoint}")
    print(f"deployment {deployment}")
    print(f"api key    set ({len(key)} chars, not shown)")

    if args.plan or not args.run:
        print("\nWould send, in order:")
        print("  1. up to 6 discovery requests, ~200 tokens each, to find a "
              "request shape this deployment accepts")
        print(f"  2. five probes of ~{BIG_PREFIX_TOKENS:,} tokens "
              f"(one ~{TINY_PREFIX_TOKENS} below the minimum)")
        print("\nRoughly 30k input tokens total, max_tokens=1 on every call, so "
              "output cost is negligible. Under $0.20 at Opus rates and far less "
              "on Sonnet or Haiku.")
        print("\nRe-run with --run to execute.")
        return 0

    run_id, records, shape = run(endpoint, deployment, key)

    out = os.path.join(HERE, f"foundry_run_{run_id}.jsonl")
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\nraw records: {out}  (gitignored)")

    if shape is None:
        return 1
    print("\nDraft registry row — check against Microsoft's docs before "
          "publishing any of it:\n")
    print(json.dumps(draft_row(records), indent=2))
    return 0


def _fail(msg: str) -> int:
    print(f"foundry-canary: {msg}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
