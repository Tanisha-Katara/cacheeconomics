"""Reading LiteLLM proxy logs.

The whole value of this adapter is that a client changes nothing and gets an
answer, which means every mistake here is invisible to them. Two are worth
guarding above all others, and both come from LiteLLM's own semantics rather
than from anything this package does:

  `prompt_tokens` is the *inclusive* billed total, because the Anthropic
  transform adds cache creation and cache read onto Anthropic's `input_tokens`.
  Mapping it onto `input_tokens` would count every cached token twice.

  The per-lifetime split is present, under `cache_creation_token_details`.
  Losing it makes every write's price unprovable, which excludes it from the
  dollar figures entirely -- a silent 100% understatement of write spend.

Verified against litellm at commit cad32fd9, whose provenance is pinned in
landscape/SOURCES.md.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cacheeconomics.adapters.litellm import (load_litellm,  # noqa: E402
                                             request_from_payload,
                                             usage_from_payload)
from cacheeconomics.analyzer import analyze                          # noqa: E402
from cacheeconomics.trace import _billed_input, write_tokens         # noqa: E402


def payload(i=0, fresh=300, read=200_000, write=0, ttl="5m", **over):
    prompt_tokens = fresh + read + write
    row = {
        "id": f"c{i}", "trace_id": "sess-1", "litellm_call_id": f"lc{i}",
        "call_type": "acompletion", "status": "success",
        "custom_llm_provider": "anthropic", "model": "claude-opus-5",
        "startTime": 1785000000 + 120 * i,
        "endTime": 1785000000 + 120 * i + 4,
        "completionStartTime": 1785000000 + 120 * i + 1,
        "prompt_tokens": prompt_tokens, "completion_tokens": 120,
        "total_tokens": prompt_tokens + 120,
        "prompt_tokens_details": {
            "cached_tokens": read,
            "cache_write_tokens": write,
            "cache_creation_tokens": write,
            "text_tokens": fresh,
            "cache_creation_token_details": {
                "ephemeral_5m_input_tokens": write if ttl == "5m" else 0,
                "ephemeral_1h_input_tokens": write if ttl == "1h" else 0}},
        "metadata": {"user_api_key_team_id": "team-a", "session_id": "sess-1",
                     "requester_metadata": {"agent": "research"}},
        "request_tags": [], "response_cost": 0.5,
    }
    row.update(over)
    return row


def written(rows):
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    with open(path, "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows))
    return path


class TestTheBilledTotalIsPreserved(unittest.TestCase):
    """The invariant the adapter lives or dies on: whatever shape the payload
    used, the tokens we think were billed must equal the tokens LiteLLM says
    were billed."""

    def test_billed_input_equals_prompt_tokens(self):
        for fresh, read, write in ((300, 200_000, 0), (300, 0, 200_000),
                                   (9_000, 0, 0), (100, 50_000, 50_000)):
            with self.subTest(fresh=fresh, read=read, write=write):
                row = payload(fresh=fresh, read=read, write=write)
                u = usage_from_payload(row)
                self.assertEqual(_billed_input(u), row["prompt_tokens"])

    def test_it_holds_when_only_the_per_lifetime_split_is_present(self):
        """The shape the fixture never produced, because `payload()` always sets
        both mirrored aggregates alongside the details.

        With neither aggregate, `written` stayed zero, so the write tokens
        remained inside the uncached remainder *and* were emitted again as the
        split. Measured: prompt_tokens 1,000 with a 700-token 5m write
        reconstructed as 1,700 billed -- a 70% overstatement of the one number
        this adapter exists to preserve."""
        for w5, w1h in ((700, 0), (0, 700), (400, 300)):
            with self.subTest(m5=w5, h1=w1h):
                row = {"model": "claude-opus-5", "prompt_tokens": 1_000,
                       "prompt_tokens_details": {
                           "cached_tokens": 0,
                           "cache_creation_token_details": {
                               "ephemeral_5m_input_tokens": w5,
                               "ephemeral_1h_input_tokens": w1h}}}
                u = usage_from_payload(row)
                self.assertEqual(_billed_input(u), 1_000)
                self.assertEqual(u["input_tokens"], 1_000 - w5 - w1h)
                self.assertEqual(write_tokens(u), w5 + w1h)

    def test_an_aggregate_still_wins_over_the_split_sum(self):
        """They are the same number reported twice; the aggregate is what the
        provider states."""
        row = {"model": "claude-opus-5", "prompt_tokens": 1_000,
               "prompt_tokens_details": {
                   "cached_tokens": 0, "cache_write_tokens": 700,
                   "cache_creation_token_details": {
                       "ephemeral_5m_input_tokens": 700,
                       "ephemeral_1h_input_tokens": 0}}}
        u = usage_from_payload(row)
        self.assertEqual(u["cache_creation_input_tokens"], 700)
        self.assertEqual(_billed_input(u), 1_000)

    def test_the_uncached_portion_is_not_the_inclusive_total(self):
        """The trap. `prompt_tokens` is 200,300 on a request whose fresh input
        was 300; reading it as `input_tokens` would price 200,000 cached tokens
        at the full uncached rate -- a 668x overstatement here, always in the
        direction that makes caching look pointless."""
        u = usage_from_payload(payload(fresh=300, read=200_000))
        self.assertEqual(u["input_tokens"], 300)
        self.assertEqual(u["cache_read_input_tokens"], 200_000)

    def test_it_survives_a_missing_text_tokens(self):
        """Older payloads omit it, so the uncached portion is recomputed by
        subtraction. It must never go negative -- a negative counter is refused
        at the pricing boundary, which would drop the row entirely."""
        row = payload(fresh=300, read=200_000)
        row["prompt_tokens_details"].pop("text_tokens")
        u = usage_from_payload(row)
        self.assertEqual(u["input_tokens"], 300)

        row = payload(fresh=300, read=200_000)
        row["prompt_tokens_details"].pop("text_tokens")
        row["prompt_tokens"] = 1_000          # inconsistent payload
        self.assertGreaterEqual(usage_from_payload(row)["input_tokens"], 0)


class TestTheWriteLifetimeSurvives(unittest.TestCase):
    """Without the split, a write's price is unprovable and the request is
    excluded from every dollar figure. The split is genuinely in the payload, so
    failing to read it would be a self-inflicted 100% understatement."""

    def test_the_split_is_carried_through(self):
        u = usage_from_payload(payload(read=0, write=9_000, ttl="1h"))
        self.assertEqual(u["cache_creation"]["ephemeral_1h_input_tokens"], 9_000)
        self.assertEqual(u["cache_creation"]["ephemeral_5m_input_tokens"], 0)

    def test_either_spelling_of_the_write_count_is_read(self):
        """LiteLLM mirrors `cache_write_tokens` and `cache_creation_tokens` onto
        each other, so a payload serialised through either path must work."""
        for name in ("cache_write_tokens", "cache_creation_tokens"):
            with self.subTest(field=name):
                row = payload(read=0, write=9_000)
                row["prompt_tokens_details"].pop("cache_write_tokens")
                row["prompt_tokens_details"].pop("cache_creation_tokens")
                row["prompt_tokens_details"][name] = 9_000
                self.assertEqual(write_tokens(usage_from_payload(row)), 9_000)

    def test_a_write_is_priced_rather_than_excluded(self):
        rows = [payload(i, read=0, write=200_000, ttl="1h") for i in range(12)]
        path = written(rows)
        try:
            a = analyze(load_litellm(path), allow_unreconciled=True)
            self.assertTrue(a.spend["input_usd"].raw() > 0)
            self.assertFalse(any("unprovable" in n for n in a.notes),
                             "the split makes the lifetime provable")
        finally:
            os.unlink(path)


class TestPayloadShapes(unittest.TestCase):

    def test_a_raw_provider_usage_object_wins(self):
        """When the proxy kept it, it states the counts outright rather than
        being derived from them."""
        row = payload(fresh=300, read=200_000)
        row["metadata"]["usage_object"] = {
            "input_tokens": 42, "cache_read_input_tokens": 7,
            "cache_creation_input_tokens": 0, "cache_creation": {}}
        u = usage_from_payload(row)
        self.assertEqual(u["input_tokens"], 42)
        self.assertEqual(u["cache_read_input_tokens"], 7)

    def test_a_wrapped_payload_is_unwrapped(self):
        path = written([{"standard_logging_object": payload(i)} for i in range(4)])
        try:
            self.assertEqual(len(load_litellm(path).requests), 4)
        finally:
            os.unlink(path)

    def test_millisecond_timestamps_are_recognised(self):
        """Some exporters emit ms. Read as seconds that is the year 58,500, and
        a workload spanning millennia silently disables every cadence rule."""
        row = payload(startTime=1785000000000, completionStartTime=1785000001000)
        r = request_from_payload(row)
        self.assertEqual(r.sent_at.year, 2026)

    def test_a_failed_call_is_not_counted_as_a_normal_request(self):
        r = request_from_payload(payload(status="failure"))
        self.assertNotEqual(r.status, 200)

    def test_the_provider_maps_to_a_registry_surface(self):
        self.assertEqual(request_from_payload(payload()).target_id,
                         "anthropic/direct")
        self.assertEqual(
            request_from_payload(payload(custom_llm_provider="bedrock")).target_id,
            "amazon-bedrock/converse")

    def test_an_unmapped_provider_is_not_assumed_to_be_anthropic(self):
        """Assuming Anthropic multipliers for an unchecked surface is how a
        confident number gets built on a guess. It keeps the provider name and
        fails closed at pricing."""
        r = request_from_payload(payload(custom_llm_provider="cohere"))
        self.assertEqual(r.target_id, "cohere")

    def test_identity_fields_are_mapped(self):
        r = request_from_payload(payload())
        self.assertEqual(r.session, "sess-1")
        self.assertEqual(r.agent, "research")
        self.assertEqual(r.tenant, "team-a")

    def test_an_agent_tag_is_read_when_metadata_has_none(self):
        row = payload()
        row["metadata"].pop("requester_metadata")
        row["request_tags"] = ["env:prod", "agent:planner"]
        self.assertEqual(request_from_payload(row).agent, "planner")

    def test_no_agent_anywhere_pools_honestly(self):
        row = payload()
        row["metadata"].pop("requester_metadata")
        self.assertEqual(request_from_payload(row).agent, "unknown")


class TestMalformedInputDoesNotCrash(unittest.TestCase):
    """A log file is somebody else's output. It will contain surprises."""

    def test_bad_lines_are_counted_not_fatal(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            with open(path, "w") as f:
                f.write("not json\n")
                f.write(json.dumps({"no": "model"}) + "\n")
                f.write(json.dumps(payload(1)) + "\n")
                f.write("\n")
                f.write(json.dumps([1, 2, 3]) + "\n")
            ts = load_litellm(path)
            self.assertEqual(len(ts.requests), 1)
            self.assertTrue(any("not valid JSON" in n for n in ts.notes))
            self.assertTrue(any("no model name" in n for n in ts.notes))
        finally:
            os.unlink(path)

    def test_nonsense_counters_do_not_reach_the_usage(self):
        for bad in (-5, True, "1000", float("nan"), None, {"a": 1}):
            with self.subTest(value=bad):
                row = payload()
                row["prompt_tokens_details"]["cached_tokens"] = bad
                u = usage_from_payload(row)
                self.assertIsInstance(u["cache_read_input_tokens"], int)
                self.assertGreaterEqual(u["cache_read_input_tokens"], 0)

    def test_a_missing_details_block_is_uncovered_not_assumed_uncached(self):
        """`prompt_tokens` is inclusive of reads and writes -- the first thing
        this module's docstring says. So with no split there is no evidence
        which class those tokens fell into, and this used to reconstruct them
        as *entirely uncached*: on a well-cached Anthropic workload that prices
        0.1x reads at 1x and reports the cache as absent, telling the client to
        start caching what they already cache.

        The previous version of this test asserted
        `u["input_tokens"] == row["prompt_tokens"]`, which is that assumption
        written down as a contract. Tolerance was the right instinct -- the row
        must not crash or vanish -- but it is bought by leaving the row
        honestly uncovered, not by inventing the split."""
        row = payload()
        row.pop("prompt_tokens_details")
        self.assertEqual(usage_from_payload(row), {})

    def test_the_missing_split_is_reported_rather_than_silent(self):
        path = written([{k: v for k, v in payload(i).items()
                         if k != "prompt_tokens_details"} for i in range(4)])
        try:
            ts = load_litellm(path)
            self.assertTrue([n for n in ts.notes if "prompt_tokens_details" in n])
        finally:
            os.unlink(path)

    def test_the_tier_is_usage_only_and_says_why(self):
        """It must not claim structural confidence it does not have."""
        path = written([payload(i) for i in range(4)])
        try:
            ts = load_litellm(path)
            self.assertFalse(ts.tier.supports_counterfactual)
            self.assertTrue(any("not reconstructed" in n for n in ts.notes))
        finally:
            os.unlink(path)


class TestModelIdsAreNormalised(unittest.TestCase):
    """A proxy is exactly where non-bare model ids come from.

    LiteLLM routes on `anthropic/claude-opus-5`; pinned deployments send
    `claude-opus-5-20250101`. The registry is keyed on bare ids, so both were
    refused by `cost.price` -- every request on them dropped out of spend, the
    ratios and reconciliation, on the ingest path whose whole pitch is that the
    client changes nothing.
    """

    def _priced(self, model):
        from cacheeconomics import cost, registry
        r = request_from_payload(payload(model=model))
        try:
            return cost.price(cost.Usage(uncached_input=1_000_000), r.model,
                              target_id=r.target_id, on_date="2026-07-30").usd
        except registry.RegistryError:
            return None

    def test_a_provider_prefixed_id_prices(self):
        self.assertEqual(self._priced("anthropic/claude-opus-5"), 5.0)

    def test_a_date_stamped_id_prices(self):
        self.assertEqual(self._priced("claude-opus-5-20250101"), 5.0)

    def test_a_bare_id_is_unchanged(self):
        self.assertEqual(self._priced("claude-opus-5"), 5.0)

    def test_an_unknown_model_still_fails_closed(self):
        """Normalisation must not invent a match. A genuinely unknown model has
        to keep being refused, or the registry's refusal to guess is worthless."""
        self.assertIsNone(self._priced("not-a-real-model"))
        self.assertIsNone(self._priced("anthropic/not-a-real-model"))


class TestSessionComesFromWhereverTheIdIs(unittest.TestCase):
    """`trace_id` is the documented top-level field, but callers routinely set
    correlation ids under `metadata` -- the runtime plugin reads all three for
    that reason. Reading only the top level meant `session=None`, which turns off
    in-session rebuild detection: REB-1 is the highest-value finding a usage-only
    trace can produce, and it would have gone quiet on logs that contained the
    key needed to find it."""

    def test_metadata_fields_are_read(self):
        for field in ("session_id", "conversation_id"):
            with self.subTest(field=field):
                row = payload()
                row.pop("trace_id")
                row["metadata"] = {field: "from-metadata"}
                self.assertEqual(request_from_payload(row).session, "from-metadata")

    def test_a_trace_id_is_not_a_conversation(self):
        """Corrected twice, and this is the reasoning that settles it.

        I first asserted the top-level `trace_id` outranks `metadata.session_id`,
        then that it was a weaker-but-usable fallback. Both were wrong. LiteLLM's
        schema defines it as spanning "multiple LLM calls belonging to same
        overall request (e.g. fallbacks/retries)", so each *turn* carries a
        different one.

        Measured on a realistic 14-request log: fourteen singleton sessions,
        REB-1 cannot fire because no session has a second request to compare
        against, and REB-0 -- whose entire job is to report that rebuild
        detection is unavailable -- is suppressed because sessions are non-null.
        A weak signal would be worse than nothing here; this one is worse than
        that, because it is silent in both directions."""
        row = payload(trace_id="top-level")
        row["metadata"].pop("session_id", None)
        self.assertIsNone(request_from_payload(row).session)

    def test_no_id_anywhere_is_none_not_invented(self):
        row = payload()
        row.pop("trace_id")
        row["metadata"] = {}
        self.assertIsNone(request_from_payload(row).session)

    def test_rebuild_detection_works_off_metadata_only_logs(self):
        """The reason this matters, asserted end to end."""
        rows = []
        for i in range(14):
            r = payload(i, fresh=400, read=0, write=200_000)
            r.pop("trace_id")
            r["metadata"]["session_id"] = "s1"
            rows.append(r)
        path = written(rows)
        try:
            a = analyze(load_litellm(path), allow_unreconciled=True)
            self.assertIn("REB-1", [f.code for f in a.findings])
        finally:
            os.unlink(path)


class TestALossyExportCannotReconcile(unittest.TestCase):
    """An invoice covers the traffic that ran, not the traffic that parsed.

    Rows the loader could not read were mentioned in `notes` and nowhere the
    reconciliation gate could see, so a malformed export whose surviving subset
    happened to match the invoice passed the gate and released figures over an
    unknown denominator.
    """

    def _trace(self, good=2, bad=1):
        rows = []
        for i in range(good):
            r = payload(i, fresh=1_000_000, read=0, write=0)
            rows.append(json.dumps(r))
        rows += ["{ not json" for _ in range(bad)]
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        with open(path, "w") as f:
            f.write("\n".join(rows))
        return path

    def test_skipped_rows_are_carried_structurally(self):
        path = self._trace()
        try:
            ts = load_litellm(path)
            self.assertEqual(len(ts.requests), 2)
            self.assertEqual(ts.skipped_rows, 1)
        finally:
            os.unlink(path)

    def test_an_exactly_matching_invoice_still_does_not_pass(self):
        path = self._trace()
        try:
            a = analyze(load_litellm(path), invoice_usd=10.0)
            r = a.reconciliation
            self.assertEqual(r["delta_pct"], 0.0, "the parsed subset matches exactly")
            self.assertFalse(r["within_ship_gate"])
            self.assertEqual(r["blockers"]["skipped_rows"], 1)
            self.assertFalse(a.spend["input_usd"].released)
            self.assertIn("could not read", a.spend["input_usd"].withheld_because)
        finally:
            os.unlink(path)

    def test_a_clean_export_still_reconciles(self):
        """The gate has to let a complete export through."""
        path = self._trace(good=2, bad=0)
        try:
            a = analyze(load_litellm(path), invoice_usd=10.0)
            self.assertTrue(a.reconciliation["within_ship_gate"])
            self.assertTrue(a.spend["input_usd"].released)
        finally:
            os.unlink(path)


class TestMissingAccountingIsNotZeroAccounting(unittest.TestCase):
    """Candidates are tried, not accepted on key presence.

    A `metadata.usage_object` of all zeros beat a top-level
    `prompt_tokens_details` holding 300 fresh and 200,000 cached tokens, so the
    row priced at zero and its whole cache activity vanished. And a payload with
    a model but no counters at all produced a full set of zeroed fields, which
    `has_usage` accepts -- so a row whose cost is simply unknown was analysed as
    a $0 request and diluted every ratio it touched.
    """

    def test_a_payload_with_no_counters_carries_no_usage(self):
        self.assertEqual(usage_from_payload({"model": "claude-opus-5"}), {})

    def test_a_zero_usage_object_does_not_shadow_real_details(self):
        row = payload(fresh=300, read=200_000)
        row["metadata"]["usage_object"] = {
            "input_tokens": 0, "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0}
        u = usage_from_payload(row)
        self.assertEqual(u["input_tokens"], 300)
        self.assertEqual(u["cache_read_input_tokens"], 200_000)

    def test_a_real_usage_object_still_wins(self):
        row = payload(fresh=300, read=200_000)
        row["metadata"]["usage_object"] = {
            "input_tokens": 42, "cache_read_input_tokens": 7,
            "cache_creation_input_tokens": 0}
        self.assertEqual(usage_from_payload(row)["input_tokens"], 42)

    def test_a_split_only_candidate_counts_as_input(self):
        """Writes reported per lifetime with no aggregate are still accounting."""
        row = {"model": "claude-opus-5", "metadata": {"usage_object": {
            "input_tokens": 0, "cache_read_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 9_000,
                               "ephemeral_1h_input_tokens": 0}}}}
        self.assertTrue(usage_from_payload(row))

    def test_such_a_row_is_uncovered_rather_than_free(self):
        rows = [payload(i) for i in range(3)] + [{"model": "claude-opus-5",
                                                 "startTime": 1785009999,
                                                 "id": "blind"}]
        path = written(rows)
        try:
            ts = load_litellm(path)
            self.assertEqual(len(ts.analysable), 3)
            self.assertEqual(ts.coverage["excluded"].get("no usage fields"), 1)
        finally:
            os.unlink(path)


class TestUnknownCostBlocksTheGate(unittest.TestCase):
    """`analyze` works over `ts.analysable`, so rows with no usage were dropped
    before any count could see them -- while still being real requests that
    really cost money. Measured: a two-request trace where one row had no usage
    reconciled its priced half against the invoice exactly, passed the gate at
    0.0%, and released the figure, with coverage reading 1/2 directly above."""

    def _ts(self):
        from datetime import datetime, timedelta, timezone
        from cacheeconomics.trace import Request, Tier, TraceSet
        t0 = datetime(2026, 7, 29, 9, tzinfo=timezone.utc)
        return TraceSet(requests=[
            Request(request_id="paid", sent_at=t0, model="claude-opus-5",
                    agent="a", session="s", ttl_requested="5m",
                    usage={"input_tokens": 2_000_000,
                           "cache_read_input_tokens": 0,
                           "cache_creation_input_tokens": 0}, segments=[]),
            Request(request_id="blind", sent_at=t0 + timedelta(seconds=60),
                    model="claude-opus-5", agent="a", session="s",
                    usage={}, segments=[], status=200)],
            tier=Tier.USAGE_ONLY, source="x")

    def test_a_matching_invoice_over_a_partial_denominator_stays_withheld(self):
        a = analyze(self._ts(), invoice_usd=10.0)
        r = a.reconciliation
        self.assertEqual(r["delta_pct"], 0.0, "the priced subset matches exactly")
        self.assertFalse(r["within_ship_gate"])
        self.assertEqual(r["blockers"]["no_usage"], 1)
        self.assertFalse(a.spend["input_usd"].released)
        self.assertIn("no usage fields", a.spend["input_usd"].withheld_because)

    def test_a_failed_call_is_not_counted_as_unknown_spend(self):
        """Non-200 rows are excluded for a different reason: they populated no
        cache entry. Counting them here would withhold every report that
        contains a single 500."""
        from cacheeconomics.trace import Tier, TraceSet
        ts = self._ts()
        ts.requests[1].status = 500
        a = analyze(TraceSet(requests=ts.requests, tier=Tier.USAGE_ONLY,
                             source="x"), invoice_usd=10.0)
        self.assertEqual(a.reconciliation["blockers"]["no_usage"], 0)
        self.assertTrue(a.reconciliation["within_ship_gate"])


class TestAnInvalidEffectiveRateIsRefused(unittest.TestCase):
    """The one number a client supplies by hand, read off an invoice, through a
    CLI flag that parses any float. It multiplies every token class.

    Measured, all four reached a published figure: NaN gave `$nan`, infinity
    gave `$nan`, -5.0 gave -$5.00, and 0.0 made the whole trace free. In
    `--format json` a bare NaN is not even valid JSON."""

    def test_each_bad_rate_is_refused(self):
        from cacheeconomics import cost
        for rate in (float("nan"), float("inf"), float("-inf"), -5.0, 0.0,
                     True, "2.5", None if False else -0.001):
            with self.subTest(rate=rate):
                with self.assertRaises(ValueError):
                    cost.price(cost.Usage(uncached_input=1_000),
                               "claude-opus-5", "anthropic/direct", on_date="2026-07-30",
                               effective_rate=rate)

    def test_a_real_negotiated_rate_still_works(self):
        from cacheeconomics import cost
        s = cost.price(cost.Usage(uncached_input=1_000_000), "claude-opus-5", "anthropic/direct",
                       on_date="2026-07-30", effective_rate=2.5)
        self.assertEqual(s.usd, 2.5)

    def test_omitting_it_still_uses_list_price(self):
        from cacheeconomics import cost
        s = cost.price(cost.Usage(uncached_input=1_000_000), "claude-opus-5", "anthropic/direct",
                       on_date="2026-07-30")
        self.assertEqual(s.usd, 5.0)


class TestAnAllZeroRowIsAPlaceholder(unittest.TestCase):
    """`has_usage` is what decides `analysable`, so it is where the
    positive-counter rule actually has to live.

    `usage_from_response` and the LiteLLM adapter both learned it; this did not,
    so the partial-denominator hole reopened through a different door one round
    after it was closed. Measured: a two-row trace with one real $10 request and
    one all-zero row reported 100% coverage, passed the ship gate against a $10
    invoice, and released the figure.
    """

    def _reqs(self, tail_usage):
        from datetime import datetime, timedelta, timezone
        from cacheeconomics.trace import Request
        t0 = datetime(2026, 7, 29, 9, tzinfo=timezone.utc)
        return [Request(request_id="paid", sent_at=t0, model="claude-opus-5",
                        agent="a", session="s", ttl_requested="5m",
                        usage={"input_tokens": 2_000_000,
                               "cache_read_input_tokens": 0,
                               "cache_creation_input_tokens": 0}, segments=[]),
                Request(request_id="tail", sent_at=t0 + timedelta(seconds=60),
                        model="claude-opus-5", agent="a", session="s",
                        usage=tail_usage, segments=[], status=200)]

    def test_all_zero_scalars_are_not_accounting(self):
        from cacheeconomics.trace import Tier, TraceSet
        ts = TraceSet(requests=self._reqs(
            {"input_tokens": 0, "cache_read_input_tokens": 0,
             "cache_creation_input_tokens": 0}),
            tier=Tier.USAGE_ONLY, source="x")
        self.assertFalse(ts.requests[1].has_usage)
        self.assertEqual(ts.coverage["analysed"], 1)
        a = analyze(ts, invoice_usd=10.0)
        self.assertFalse(a.reconciliation["within_ship_gate"])
        self.assertFalse(a.spend["input_usd"].released)

    def test_one_positive_counter_is_enough(self):
        from cacheeconomics.trace import Tier, TraceSet
        ts = TraceSet(requests=self._reqs(
            {"input_tokens": 0, "cache_read_input_tokens": 5_000,
             "cache_creation_input_tokens": 0}),
            tier=Tier.USAGE_ONLY, source="x")
        self.assertTrue(ts.requests[1].has_usage)
        self.assertEqual(ts.coverage["analysed"], 2)

    def test_a_positive_split_is_enough(self):
        from cacheeconomics.trace import Tier, TraceSet
        ts = TraceSet(requests=self._reqs(
            {"cache_creation": {"ephemeral_5m_input_tokens": 9_000,
                                "ephemeral_1h_input_tokens": 0}}),
            tier=Tier.USAGE_ONLY, source="x")
        self.assertTrue(ts.requests[1].has_usage)


class TestTheDraftOverrideRespectsTheSameBlockers(unittest.TestCase):
    """`--allow-unreconciled` covers a missing invoice and nothing else. The
    previous round added skipped and no-usage rows to the invoice gate and not
    to its escape hatch, so a draft run released $10.00 over a trace that had
    dropped a row or could not see one's cost. Fixing a gate and not its
    override is the same defect as fixing a guard and not its twin."""

    def _paid(self):
        from datetime import datetime, timezone
        from cacheeconomics.trace import Request
        return Request(request_id="paid",
                       sent_at=datetime(2026, 7, 29, 9, tzinfo=timezone.utc),
                       model="claude-opus-5", agent="a", session="s",
                       ttl_requested="5m",
                       usage={"input_tokens": 2_000_000,
                              "cache_read_input_tokens": 0,
                              "cache_creation_input_tokens": 0}, segments=[])

    def test_a_dropped_row_blocks_the_draft(self):
        from cacheeconomics.trace import Tier, TraceSet
        a = analyze(TraceSet(requests=[self._paid()], tier=Tier.USAGE_ONLY,
                             source="x", skipped_rows=1),
                    allow_unreconciled=True)
        self.assertFalse(a.spend["input_usd"].released)

    def test_a_no_usage_row_blocks_the_draft(self):
        from datetime import datetime, timedelta, timezone
        from cacheeconomics.trace import Request, Tier, TraceSet
        t0 = datetime(2026, 7, 29, 9, tzinfo=timezone.utc)
        blind = Request(request_id="blind", sent_at=t0 + timedelta(seconds=60),
                        model="claude-opus-5", agent="a", session="s",
                        usage={}, segments=[], status=200)
        a = analyze(TraceSet(requests=[self._paid(), blind],
                             tier=Tier.USAGE_ONLY, source="x"),
                    allow_unreconciled=True)
        self.assertFalse(a.spend["input_usd"].released)

    def test_a_clean_trace_still_drafts(self):
        """The override has to keep working, or it is not an override."""
        from cacheeconomics.trace import Tier, TraceSet
        a = analyze(TraceSet(requests=[self._paid()], tier=Tier.USAGE_ONLY,
                             source="x"), allow_unreconciled=True)
        self.assertTrue(a.spend["input_usd"].released)


class TestAPricedRowIsNotASkippedRow(unittest.TestCase):
    """The opposite failure, and mine: the previous round counted every no-body
    row as skipped, but rows carrying usage are kept and priced. So a report
    whose spend was complete was refused by the gate. An over-block is its own
    kind of wrong answer, and a gate nobody can pass gets switched off."""

    def test_a_no_body_row_with_usage_does_not_block(self):
        from cacheeconomics.adapters.bodies import load_bodies
        row = {"sent_at": "2026-07-29T09:00:00Z", "model": "claude-opus-5",
               "usage": {"input_tokens": 2_000_000,
                         "cache_read_input_tokens": 0,
                         "cache_creation_input_tokens": 0}}
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            with open(path, "w") as f:
                f.write(json.dumps(row))
            ts = load_bodies(path, b"k" * 32, target_id="anthropic/direct")
            self.assertEqual(len(ts.analysable), 1)
            self.assertEqual(ts.skipped_rows, 0, "it was priced, not skipped")
            a = analyze(ts, invoice_usd=10.0)
            self.assertTrue(a.reconciliation["within_ship_gate"])
        finally:
            os.unlink(path)

    def test_a_no_body_row_without_usage_still_blocks(self):
        """That one really is unusable, and must keep counting."""
        from cacheeconomics.adapters.bodies import load_bodies
        rows = [{"sent_at": "2026-07-29T09:00:00Z", "model": "claude-opus-5",
                 "usage": {"input_tokens": 2_000_000,
                           "cache_read_input_tokens": 0,
                           "cache_creation_input_tokens": 0}},
                {"sent_at": "2026-07-29T09:01:00Z", "note": "no body, no usage"}]
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            with open(path, "w") as f:
                f.write("\n".join(json.dumps(r) for r in rows))
            ts = load_bodies(path, b"k" * 32, target_id="anthropic/direct")
            self.assertEqual(ts.skipped_rows, 1)
            a = analyze(ts, invoice_usd=10.0)
            self.assertFalse(a.reconciliation["within_ship_gate"])
            self.assertEqual(a.reconciliation["blockers"]["skipped_rows"], 1)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
