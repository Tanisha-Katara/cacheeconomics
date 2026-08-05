"""Command line entry point.

Everything this package does was previously reachable only by writing Python
against it, which is fine for the person who wrote it and useless on somebody
else's machine. These subcommands are the same calls the tests make.

Two properties this file is responsible for keeping:

*Nothing here reaches the network.* No subcommand opens a socket; the registry
is packaged data and every input is a local path. That is the claim the whole
analysis is sold on -- prompt content never has to leave the machine -- and a
CLI is the obvious place for it to quietly stop being true.

*The HMAC key never appears in an argument.* There is no `--key` flag and there
will not be one: argv is visible in `ps` to every user on the box and lands in
shell history. The key comes from a file or the environment, and the file's
permissions are checked before it is read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__, checks, registry, report, simulate
from .analyzer import analyze
from .trace import UNATTRIBUTED, TraceSet, load_jsonl

KEY_ENV = "CACHEECONOMICS_HMAC_KEY"

# Short keys make the segment ids guessable, which defeats the point of hashing
# them: an attacker holding a candidate prompt can confirm it by recomputing the
# id. 16 bytes is the floor, 32 is what the recorder's own tests use.
MIN_KEY_BYTES = 16

# The surface assumed for rows that name none and an operator who said nothing.
# Named once: `--target-id` now defaults to None so a deliberate choice can be
# told from silence, and three loaders needed the same fallback spelled the same
# way.
# What a trace row means when it names no surface: nothing, deliberately.
# This was "anthropic/direct", so a gateway export whose format never carried a
# provider earned Anthropic first-party rates -- measured at $2,924/month on a
# twelve-request Bedrock-fronting capture. The registry's own rate_scope entry
# already described this as fixed; the loader default was the door it could not
# see. `--target-id` states the surface, `--effective-rate` prices it from the
# customer's own bill.
DEFAULT_TARGET = UNATTRIBUTED


class Fail(Exception):
    """A user-facing failure. Printed without a traceback."""


def _read_key(args) -> bytes | None:
    """The HMAC key, from a file or the environment. Never from argv.

    Returns None when no key was supplied, which is legal: a trace whose
    segments already carry `hmac:` ids does not need one. It is only required to
    *generate* ids from content, which is what the body adapter does.
    """
    if args.key_file:
        path = args.key_file
        if not os.path.exists(path):
            raise Fail(f"key file not found: {path}")
        # A world-readable key file is a key on a shared machine. Refusing is
        # ruder than warning and correct: the alternative is a secret that
        # everyone with an account can read while the tool says nothing.
        mode = os.stat(path).st_mode
        if mode & 0o077:
            raise Fail(
                f"{path} is readable by other users (mode {mode & 0o777:o}). "
                f"Run: chmod 600 {path}")
        with open(path, "rb") as f:
            key = f.read().strip()
    elif os.environ.get(KEY_ENV):
        key = os.environ[KEY_ENV].encode()
    else:
        return None
    if len(key) < MIN_KEY_BYTES:
        raise Fail(
            f"the HMAC key is {len(key)} bytes; {MIN_KEY_BYTES} is the minimum. "
            f"Segment ids are keyed hashes of prompt content, and a short key "
            f"lets someone holding a candidate prompt confirm it by recomputing "
            f"the id. Generate one with: openssl rand -hex 32")
    return key


def _load(args) -> TraceSet:
    """Ingest, by whichever adapter the caller named."""
    key = _read_key(args)
    if args.source == "trace":
        if not os.path.exists(args.path):
            raise Fail(f"no such file: {args.path}")
        return load_jsonl(args.path, key, default_tenant=args.tenant,
                          default_target=args.target_id or DEFAULT_TARGET)
    if args.source == "litellm":
        # No key: this adapter reads counters and identity fields only, never
        # prompt content, so there is nothing to hash and nothing to protect.
        if not os.path.exists(args.path):
            raise Fail(f"no such file: {args.path}")
        from .adapters.litellm import load_litellm
        # Passed only when the operator actually said so: this adapter can read
        # the surface off `custom_llm_provider`, and an unconditional default
        # would override rows that already know better.
        return load_litellm(args.path, default_tenant=args.tenant,
                            default_target=args.target_id)
    if args.source == "bodies":
        # Required here, unlike the trace path: this adapter hashes prompt
        # content to make ids, and `segment_id` refuses to do that unkeyed
        # rather than emit a bare SHA-256 of somebody's prompt.
        if key is None:
            raise Fail(
                f"--from bodies needs an HMAC key: it derives segment ids from "
                f"prompt content, and an unkeyed digest of a prompt is "
                f"reversible by anyone holding a guess. Set {KEY_ENV} or pass "
                f"--key-file.")
        from .adapters.bodies import load_bodies
        if not os.path.exists(args.path):
            raise Fail(f"no such file: {args.path}")
        # Passed only when the operator actually said so, like the litellm
        # path above. Injecting DEFAULT_TARGET here defeated the adapter's
        # fail-closed: a body export states the API shape, never who invoices
        # it, so an unstated surface has to stay unstated all the way down.
        return load_bodies(args.path, key, tenant=args.tenant,
                           target_id=args.target_id)
    raise Fail(f"unknown source: {args.source}")


def _coverage_line(ts: TraceSet) -> str:
    c = ts.coverage
    if not c["total"]:
        return "no requests"
    out = [f"{c['analysed']:,} of {c['total']:,} requests analysable "
           f"({c['fraction']:.0%}), tier {ts.tier.name}"]
    for reason, n in sorted(c["excluded"].items()):
        out.append(f"  excluded: {n:,} {reason}")
    return "\n".join(out)


def _json_safe(value):
    """Replace non-finite numbers with null, recursively.

    `--format json` is the machine-readable surface, and Python emits bare
    `NaN`/`Infinity` tokens that no strict parser accepts -- so the output
    became unparseable on exactly the failure path automation needs to inspect:
    `--invoice-usd nan` is recorded as an invalid invoice and then written as
    `"invoice_usd": NaN`. Null is the honest encoding: the value was supplied
    and is not a number, which is what `invalid_invoice` already says in words.
    """
    if isinstance(value, float) and (value != value or value in (
            float("inf"), float("-inf"))):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def analysis_json(a, tier_name: str = "", coverage=None) -> str:
    """The machine-readable analysis, as one callable both the CLI and its
    tests go through.

    Extracted from `cmd_analyze` because it was inline, and an invariant
    written to check "every dollar field in the JSON carries release state"
    could not reach it without a `Namespace` and a loaded trace. So the test
    grew a mirror of this dict instead -- and then tested the mirror. The
    mirror agreed with itself perfectly while the real output shipped finding
    figures with no release state at all.

    An invariant that cannot reach the real code path will be given a
    convincing substitute. The fix is not a better mirror, it is a seam.
    """
    # Figures are rendered through `str`, so a withheld one serialises as
    # "[withheld: ...]" rather than a number somebody's script would treat
    # as spend. `raw()` is deliberately not reachable from here.
    return json.dumps({
        "tier": tier_name,
        "coverage": coverage,
        "window_days": a.window_days,
        "spend": {k: str(v) for k, v in a.spend.items()},
        # A script reading this saw only strings and could not tell a
        # draft figure from an invoice-checked one. The state is the
        # machine-readable half of the DRAFT banner.
        "release_state": {k: getattr(v, "released_as", "")
                          for k, v in a.spend.items()
                          if hasattr(v, "released_as")},
        "reconciliation": _json_safe(a.reconciliation),
        "findings": [{"code": f.code, "severity": f.severity,
                      "title": f.title, "confidence": f.confidence,
                      "affected_requests": f.affected_requests,
                      "avoidable_usd_month": (str(f.avoidable_usd_month)
                                              if f.avoidable_usd_month
                                              else None)}
                     for f in a.findings],
        "notes": a.notes,
    }, indent=2, default=str, allow_nan=False)


def cmd_analyze(args) -> int:
    ts = _load(args)
    a = analyze(ts, invoice_usd=args.invoice_usd,
                effective_rate=args.effective_rate, on_date=args.on_date,
                allow_unreconciled=args.allow_unreconciled)
    if args.format == "json":
        out = analysis_json(a, tier_name=ts.tier.name, coverage=ts.coverage)
    elif args.format == "html":
        out = report.render_html(a, client=args.client or "",
                                 window_label=args.window_label or "")
    else:
        out = report.render_text(a, detail=args.detail)

    if args.out:
        with open(args.out, "w") as f:
            f.write(out)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(out)
    return 0


def cmd_bakeoff(args) -> int:
    ts = _load(args)
    reqs = ts.analysable
    if not reqs:
        raise Fail("no analysable requests: " + _coverage_line(ts))
    print(_coverage_line(ts), file=sys.stderr)
    if args.by_agent:
        results = simulate.bake_off_by_agent(
            reqs, on_date=args.on_date, effective_rate=args.effective_rate,
            invoice_usd=args.invoice_usd,
            allow_unreconciled=args.allow_unreconciled,
            excluded_billed=ts.excluded_billed)
        if not results:
            raise Fail("no agent has the 3 requests a comparison needs")
        for b in results:
            print(b)
            print()
    else:
        print(simulate.bake_off(reqs, on_date=args.on_date,
                                effective_rate=args.effective_rate,
                                invoice_usd=args.invoice_usd,
                                allow_unreconciled=args.allow_unreconciled,
                                excluded_billed=ts.excluded_billed))
    return 0


def cmd_checks(args) -> int:
    results = checks.run_all(
        prefix_tokens=args.prefix_tokens, model=args.model,
        breakpoints=args.breakpoints,
        ttls_in_order=args.ttls.split(",") if args.ttls else None,
        target_id=args.target_id,
        tokens_are_estimated=not args.tokens_are_exact,
        rolling_marker=args.rolling_marker)
    failed = abstained = 0
    for r in results:
        print(f"[{r.status.name}] {r.check}: {r.summary}")
        if r.detail:
            print(f"    {r.detail}")
        if r.resolve:
            print(f"    fix: {r.resolve}")
        failed += r.status is checks.Status.FAIL
        abstained += r.status is checks.Status.ABSTAIN
    # The exit code carries the verdict, so this is usable as a pipeline gate.
    #
    # ABSTAIN gets its own code rather than folding into success. "I could not
    # evaluate this" is not "this passed": the registry abstains on a model it
    # has no dated minimum for, and a first mapping here sent that to 0 -- a
    # green build in which the minimum-cacheable check, the one that catches
    # markers the provider silently ignores, never ran at all. Reading absence of
    # a failure as evidence of a pass is the exact shape of defect this package
    # exists to find in other people's caching.
    if failed:
        return 2
    if abstained:
        return 3
    return 0


def _claude_code_target(args) -> str:
    """The surface for a Claude Code run: stated, assumed on purpose, or refused.

    A transcript records the conversation rather than the wire request and
    carries no provider field anywhere -- checked across 190 of them -- so this
    surface can never be read from the data. It used to be *fabricated* instead:
    `args.target_id or "anthropic/direct"`, which is the default path of the
    command, so the ordinary invocation priced transcripts at Anthropic
    first-party rates whatever they actually ran against. Measured on a
    40-request fixture with `--allow-unreconciled`: $6.66 input spend, $38.46
    uncached, $31.80 saved, $123/mo, all released as DRAFT under a rate table
    nobody had chosen. On a Bedrock- or Vertex-routed deployment every one of
    those is against the wrong table, and the only thing standing in the way was
    a prose caveat further down the report.

    Defaulting to UNATTRIBUTED instead was measured before it was rejected, and
    it does not merely withhold the dollars -- it empties the report. `analyze`
    recomputes its ratios over the *priced* requests, so with none priceable the
    same fixture reports 0 requests, `input_from_cache` None, `prefix_efficiency`
    None and no findings at all, in place of 40 requests at 94% and 94%. Those
    two ratios are computed from the provider's own usage counters and do not
    depend on the rate table in any way, so silently trading them away to avoid
    naming a surface would be an over-block: a command that answers nothing gets
    switched off, and then it catches nothing.

    So the choice is made explicit rather than either fabricated or removed.
    Anthropic direct is still very likely correct -- Claude Code talks to
    Anthropic unless CLAUDE_CODE_USE_BEDROCK or CLAUDE_CODE_USE_VERTEX is set --
    and `--assume-anthropic-direct` is how you say so. It stays an assumption:
    the adapter's blocking note still fires and still holds release.

    Refusing is louder and cheaper than the alternatives: one flag, at start-up,
    instead of a wrong rate table discovered from a bill. Same shape as
    `litellm_handler(mutate=True)`, which refuses at construction for the same
    reason.
    """
    if args.target_id and args.assume_anthropic_direct:
        raise Fail(
            "pass either --target-id or --assume-anthropic-direct, not both: "
            f"--target-id says the surface is {args.target_id!r} and "
            f"--assume-anthropic-direct says to assume anthropic/direct.")
    if args.target_id:
        return args.target_id
    if args.assume_anthropic_direct:
        return "anthropic/direct"
    raise Fail(
        "this command needs to be told which provider surface these sessions "
        "ran against, because a Claude Code transcript records the conversation "
        "rather than the wire request and carries no provider field to read it "
        "from. The surface decides which rate table applies, and Bedrock and "
        "Vertex rates are not Anthropic's.\n"
        "  --assume-anthropic-direct   Claude Code talks to Anthropic directly "
        "unless CLAUDE_CODE_USE_BEDROCK or CLAUDE_CODE_USE_VERTEX is set on the "
        "machine that produced these transcripts. This is usually the right "
        "answer; the report still states it as an assumption and still withholds "
        "figures until they reconcile.\n"
        "  --target-id <surface>       name it exactly, e.g. "
        "amazon-bedrock/converse or google-cloud/vertex.")


# The surface half of a DRAFT reason, written to be *appended* to whatever else
# already made this a draft rather than to stand alone. It deliberately does not
# open with "DRAFT" or promise that naming the surface reconciles anything: with
# no invoice supplied it would not, and an earlier version of this string said so
# anyway while sitting in front of the analyzer's true "no invoice was supplied".
ASSUMED_SURFACE_CLAUSE = (
    "The provider surface was assumed rather than measured: "
    "--assume-anthropic-direct supplied anthropic/direct, and every dollar "
    "figure here is priced from that surface's rate table. An invoice can check "
    "that the total adds up; it cannot check that the rate table it was added up "
    "from is the right one, because a Claude Code transcript carries no provider "
    "field to compare against. Pass --target-id to state the surface from "
    "knowledge instead.")

ASSUMED_SURFACE_DRAFT_NOTE = "DRAFT — " + ASSUMED_SURFACE_CLAUSE


def _draft_because_the_surface_was_assumed(a):
    """Re-release every figure in `a` as DRAFT rather than RECONCILED.

    The floor under an assumed surface. Reconciliation checks a *total* against
    an invoice; it cannot check the *rate table* that total was computed from,
    and with `--assume-anthropic-direct` that table came from an assumption. So
    a report could carry `released_as='reconciled'` -- the label meaning an
    invoice verified this -- over dollars whose provenance was a guess.
    Reproduced on a 40-request fixture: an invoice equal to computed spend
    reconciled at 0.0% and released input_usd, if_uncached_usd,
    caching_saved_usd and monthly_input_usd all as 'reconciled'.

    That is this project's central failure in one place: an assumption
    published with the provenance of a measurement. The assumption *was*
    disclosed, but only as free text a renderer adds later -- in the text report
    a costed finding and the total appear above the caveat, and in HTML the
    Input spend KPI appears above the standing notes. Neither a reader skimming
    nor a script reading the JSON `release_state` is reached by prose.

    DRAFT is the existing vocabulary for "released, and not invoice-checked", so
    this reuses it rather than inventing a third state that every renderer and
    consumer would then have to learn. `report._is_draft` reads release state
    off the figures rather than off the notes, so both renderers stamp this
    without being told separately, and `Analysis.total_avoidable_month` derives
    DRAFT from its parts on its own.

    Deliberately a downgrade and never an upgrade: a figure that is withheld
    stays withheld, and one already DRAFT stays DRAFT. `Figure.release` keeps an
    explicit `as_`, so this cannot launder a withheld figure into a released one.

    This is a floor, not the whole fix. It covers the surface *this flag*
    assumed. The general form -- structured surface provenance on the figures
    themselves, and `ts.blocking_notes` feeding the release decision so that any
    assumed input blocks reconciled release whatever produced it -- belongs in
    `analyzer.analyze` and is not done here.
    """
    import dataclasses

    from . import money

    def draft(v):
        # Only touch what is actually published as invoice-checked. A withheld
        # figure has no release state to downgrade and must keep its reason.
        if not isinstance(v, money.Figure) or not v.released:
            return v
        return v.release(True, as_=money.DRAFT)

    findings = [dataclasses.replace(f, avoidable_usd_month=draft(f.avoidable_usd_month))
                if f.avoidable_usd_month is not None else f
                for f in a.findings]
    spend = {k: draft(v) for k, v in a.spend.items()}

    # The banner is driven by the same evidence `report._is_draft` reads -- a
    # figure actually released as DRAFT -- and not by the fact that this function
    # ran. Two failures came out of prepending it unconditionally:
    #
    # A report whose reconciliation *failed* has every figure withheld, so there
    # is no draft to announce. The note went in anyway, and because `render_text`
    # looks for a note beginning "DRAFT" while `render_html` calls `_is_draft`,
    # the text report stamped a draft banner and the HTML did not. Measured on a
    # $999 invoice against $1.16 of spend: `_is_draft` False, HTML gate div
    # absent, text banner present. Two renderers disagreeing about the same
    # verdict is the defect this repo has the longest history with, and it had
    # just been reintroduced by a fix written to close a provenance hole.
    #
    # So: derive from the figures, exactly as the renderers' own predicate does.
    released_draft = any(isinstance(v, money.Figure) and v.released
                         and v.released_as == money.DRAFT for v in spend.values())

    notes = list(a.notes)
    if released_draft:
        # Composed with whatever else already made this a draft, rather than
        # winning by being first. With `--allow-unreconciled` and no invoice the
        # analyzer inserts its own "figures released without invoice
        # reconciliation" at notes[0]; prepending in front of it made
        # `_draft_reason` return the surface note instead -- which told the
        # reader that passing --target-id would make these reconciled figures.
        # With no invoice supplied that is simply false, and it had replaced a
        # true explanation with it. Both reasons are real here and the reader
        # needs both.
        existing = next((n for n in notes if n.startswith("DRAFT")), None)
        if existing is None:
            notes.insert(0, ASSUMED_SURFACE_DRAFT_NOTE)
        elif ASSUMED_SURFACE_CLAUSE not in existing:
            notes[notes.index(existing)] = existing.rstrip() + " " + ASSUMED_SURFACE_CLAUSE
    return dataclasses.replace(
        a,
        spend=spend,
        reconciliation=({k: draft(v) for k, v in a.reconciliation.items()}
                        if a.reconciliation else a.reconciliation),
        findings=findings,
        notes=notes,
        # Unconditional, unlike the banner: the assumption qualifies these
        # figures whether or not any of them cleared the gate, and that is what
        # this list is for. The clause without the "DRAFT — " prefix, so the word
        # never appears on a report where nothing was released as a draft.
        blocking_notes=list(a.blocking_notes) + [ASSUMED_SURFACE_CLAUSE])


def cmd_claude_code(args) -> int:
    """Analyse local Claude Code transcripts.

    Reads only usage counters and prompt *shape* from the session files. It
    still touches transcripts, so the output can carry counts and timings
    derived from real work -- worth knowing before piping it anywhere.

    The surface has to be chosen, one way or the other. See `_claude_code_target`.
    """
    from .adapters.claude_code import load_sessions
    target_id = _claude_code_target(args)
    ts = load_sessions(root=args.root, project=args.project, limit=args.limit,
                       target_id=target_id,
                       # Passed, not re-derived from `target_id`. The adapter
                       # cannot tell an assumed anthropic/direct from a stated
                       # one -- it is the same string -- and it used to guess
                       # from the value, so `--target-id anthropic/direct` was
                       # told its own surface was an assumption.
                       surface_assumed=bool(args.assume_anthropic_direct))
    a = analyze(ts, invoice_usd=args.invoice_usd,
                effective_rate=args.effective_rate, on_date=args.on_date,
                allow_unreconciled=args.allow_unreconciled)
    # Keyed on the flag rather than on the resulting surface id: `--target-id
    # anthropic/direct` is the same string arrived at by knowledge, and there is
    # nothing assumed about it.
    if args.assume_anthropic_direct:
        a = _draft_because_the_surface_was_assumed(a)
    print(_coverage_line(ts), file=sys.stderr)
    print(report.render_text(a, detail=args.detail))
    return 0


def cmd_registry(args) -> int:
    """What the registry actually knows, so a gap is visible before a run."""
    print(f"registry: {registry.REGISTRY_DIR}")
    providers = registry._load("providers.json")
    pricing = registry._load("pricing.json")
    print(f"generated: {providers.get('generated', 'unknown')}")
    priced = set(pricing.get("models", {}))
    print("\nsurfaces:   rates=list: billed at the recorded Anthropic rates. "
          "rates=invoice-only:\n"
          "            operated and invoiced by the cloud provider, so pass "
          "--effective-rate.")
    for t in providers.get("targets", []):
        cap = t.get("capabilities", {})
        ttls = ",".join(cap.get("supported_ttls") or []) or "none"
        mins = t.get("min_cacheable_tokens", {})
        # Which models on this surface this build can actually price. A surface
        # the registry describes but cannot price is the difference between "we
        # support that" and "we can answer a question about it", and reading it
        # off the two files beats finding out mid-engagement.
        n = len(priced & set(mins))
        # A missing key and an explicit null both mean "this surface has no
        # breakpoint budget", and printing "-" for one and "None" for the other
        # reads as two different states.
        bp = cap.get("max_breakpoints")
        # Whether the recorded rates apply here at all, which is a different
        # question from how many models are described and the one that decides
        # if a trace on this surface produces dollars. Bedrock reads as fully
        # capable on every other column.
        rate = "list" if registry.rates_apply_to(t.get("id", "")) else "invoice-only"
        print(f"  {t.get('id', '?'):<34} breakpoints={'-' if bp is None else bp:<4} "
              f"ttls={ttls:<8} priced_models={n}/{len(mins):<4} rates={rate}")
    print("\npriced models:")
    for m in sorted(priced):
        print(f"  {m}")
    return 0


def _ingest_args(p, need_path=True):
    if need_path:
        p.add_argument("path", help="trace file to read (JSONL)")
    p.add_argument("--from", dest="source", default="trace",
                   choices=("trace", "bodies", "litellm"),
                   help="ingest adapter: a normalised trace (default), request "
                        "bodies you already log, or LiteLLM proxy logs "
                        "(StandardLoggingPayload JSONL)")
    p.add_argument("--key-file", metavar="PATH",
                   help=f"file holding the HMAC key for segment ids. Must not "
                        f"be readable by other users. Alternative: the {KEY_ENV} "
                        f"environment variable. There is deliberately no --key "
                        f"flag: argv is visible in ps and saved to shell history")
    p.add_argument("--tenant",
                   help="tenant id for rows that do not carry one. It is part "
                        "of cache isolation and of segment identity, so setting "
                        "it on a single-tenant export is worth doing")
    # Defaults to None rather than the surface itself, so "the operator chose
    # anthropic/direct" and "the operator said nothing" stay distinguishable.
    # The LiteLLM adapter reads the surface off each row and must only be
    # overridden when a choice was actually made.
    p.add_argument("--target-id", default=None,
                   help="provider surface for rows that do not name one. There "
                        "is no default: a row naming no surface stays "
                        "unattributed and unpriced rather than being assumed "
                        "first-party. Honoured on every ingest mode; a row that "
                        "names its own surface still wins")


def _pricing_args(p):
    p.add_argument("--on-date", metavar="YYYY-MM-DD",
                   help="price every request on this date instead of the day it "
                        "was sent. Pricing is date-effective; the default is "
                        "almost always what you want")
    p.add_argument("--effective-rate", type=float, metavar="USD_PER_MTOK",
                   help="the input rate to price against, in USD per million "
                        "tokens. Use it for a negotiated price the public table "
                        "does not carry, and for Bedrock and Vertex traffic -- "
                        "those are invoiced by the cloud provider, so the "
                        "recorded Anthropic rates do not apply and the analyzer "
                        "abstains until you supply the rate off that bill")


def _detail_arg(p):
    """Both report commands, one definition.

    The findings table is the short version of each finding. This turns the
    reasoning back on. Defined here rather than twice because `analyze` and
    `claude-code` render through the same function and a flag on only one of
    them is the twin-path shape this codebase keeps getting bitten by.
    """
    p.add_argument("--detail", action="store_true",
                   help="print the reasoning behind each finding, the counts "
                        "it rests on, and what it excluded")


def _release_args(p):
    p.add_argument("--invoice-usd", type=float, metavar="USD",
                   help="the provider invoice for this window. Dollar figures "
                        "stay withheld until they reconcile against one")
    p.add_argument("--allow-unreconciled", action="store_true",
                   help="release figures with no invoice, for internal drafts. "
                        "Named rather than default so publishing an "
                        "unreconciled number is always someone's decision")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cacheeconomics",
        description="Prompt-cache cost analysis. Runs locally; nothing here "
                    "makes a network call.",
        epilog="Dollar figures are withheld unless they reconcile against a "
               "real invoice (--invoice-usd) or you explicitly ask for a draft "
               "(--allow-unreconciled).")
    p.add_argument("--version", action="version",
                   version=f"cacheeconomics {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="find what a trace is paying for")
    _ingest_args(a)
    _pricing_args(a)
    _release_args(a)
    _detail_arg(a)
    a.add_argument("--format", default="text", choices=("text", "html", "json"))
    a.add_argument("--out", metavar="PATH", help="write here instead of stdout")
    a.add_argument("--client", help="client name, for the HTML header")
    a.add_argument("--window-label", help="window description, for the HTML header")
    a.set_defaults(func=cmd_analyze)

    b = sub.add_parser("bakeoff",
                       help="compare placement policies over the same trace")
    _ingest_args(b)
    _pricing_args(b)
    # The same two flags `analyze` carries, and for the same reason: these are
    # dollar amounts, and this tool does not publish an unreconciled one because
    # an argument was left out.
    _release_args(b)
    b.add_argument("--by-agent", action="store_true",
                   help="one comparison per agent; a blended number hides the "
                        "interesting cases")
    b.set_defaults(func=cmd_bakeoff)

    c = sub.add_parser("checks",
                       help="lint a cache configuration before you ship it")
    c.add_argument("--prefix-tokens", type=int, required=True,
                   help="size of the prefix you intend to cache")
    c.add_argument("--model", required=True)
    c.add_argument("--breakpoints", type=int, default=1)
    c.add_argument("--ttls", help="comma-separated lifetimes in wire order, "
                                  "e.g. 1h,5m")
    # No default surface. Minimums, breakpoint budgets and supported lifetimes
    # are all per-surface -- 512 tokens on anthropic/direct is 1,024 on
    # openai/direct -- so a default here answers for a provider the operator
    # never chose. Measured: `checks --prefix-tokens 768 --model claude-opus-5
    # --tokens-are-exact` with no surface exited 0 with three PASSes against
    # Anthropic's 512 minimum, while the same prefix on openai/direct FAILs.
    #
    # The checks are tri-state, so nothing has to be refused here: all three
    # abstain, and `cmd_checks` already maps abstention to exit 3 rather than 0
    # precisely so that "I could not evaluate this" never reads as a pass.
    c.add_argument("--target-id", default=DEFAULT_TARGET,
                   help="the provider surface this configuration ships to "
                        "(e.g. anthropic/direct, openai/direct, "
                        "amazon-bedrock/converse). Without it every check "
                        "abstains and the command exits 3, because these "
                        "thresholds differ by surface")
    c.add_argument("--tokens-are-exact", action="store_true",
                   help="the prefix size is measured, not estimated")
    c.add_argument("--rolling-marker", action="store_true",
                   help="the marker moves with the conversation")
    c.epilog = ("Exit codes: 0 all checks passed; 2 at least one failed; "
                "3 nothing failed but at least one could not be evaluated "
                "(usually a model the registry has no dated entry for); "
                "1 the tool itself broke.")
    c.set_defaults(func=cmd_checks)

    cc = sub.add_parser("claude-code",
                        help="analyse local Claude Code transcripts")
    cc.add_argument("--root", default=None,
                    help="transcript root (default: ~/.claude/projects)")
    cc.add_argument("--project", help="one project directory only")
    cc.add_argument("--limit", type=int, help="most recent N sessions only")
    # A transcript carries no provider field, so the surface here can only be
    # stated or assumed -- never read. One of these two is required; see
    # `_claude_code_target` for why it is not simply defaulted either way.
    cc.add_argument("--target-id", default=None,
                    help="the provider surface these sessions ran against, "
                         "e.g. amazon-bedrock/converse or google-cloud/vertex, "
                         "whose rates are not Anthropic's")
    cc.add_argument("--assume-anthropic-direct", action="store_true",
                    help="assume these sessions ran against anthropic/direct, "
                         "which is true unless CLAUDE_CODE_USE_BEDROCK or "
                         "CLAUDE_CODE_USE_VERTEX was set. Required if "
                         "--target-id is not given: the surface decides which "
                         "rate table applies and cannot be read from a "
                         "transcript. The report still states it as an "
                         "assumption")
    _pricing_args(cc)
    _release_args(cc)
    _detail_arg(cc)
    cc.set_defaults(func=cmd_claude_code)

    r = sub.add_parser("registry", help="what the registry knows")
    r.set_defaults(func=cmd_registry)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "root", "sentinel") is None:
        from .adapters.claude_code import DEFAULT_ROOT
        args.root = DEFAULT_ROOT
    try:
        return args.func(args)
    except Fail as e:
        print(f"cacheeconomics: {e}", file=sys.stderr)
        return 1
    except registry.RegistryError as e:
        # The registry refusing to guess is a designed outcome, not a crash, so
        # it gets a plain message rather than a traceback.
        print(f"cacheeconomics: {e}", file=sys.stderr)
        return 1
    except BrokenPipeError:
        # `cacheeconomics analyze ... | head` should not print a traceback.
        return 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
