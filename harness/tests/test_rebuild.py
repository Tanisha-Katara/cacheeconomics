"""Rebuild detection, and the verdict that stops it being misread.

These two exist because of a specific, common, confidently-stated field
diagnosis: "most of our Anthropic bill is cache reads and writes, so caching is
the problem." It is backwards. Reads bill at a tenth of the input rate, so they
dominate a bill only when the volume behind them is large — which is caching
working. What costs money is rebuilding a prefix instead of extending it, and
that is invisible in a usage dashboard because both show up as "cache writes".

Both checks need only usage counters and a session id, which is the whole point:
they work on the trace most teams can actually produce.
"""

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cacheeconomics import analyzer, monitor                      # noqa: E402
from cacheeconomics.trace import Request, Segment, Tier, TraceSet  # noqa: E402

T0 = datetime(2026, 7, 29, 9, tzinfo=timezone.utc)


def req(i, *, read, write, session="s1", model="claude-opus-5", gap=60):
    return Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=gap * i),
                   model=model, session=session,
                   usage={"input_tokens": 0, "cache_read_input_tokens": read,
                          "cache_creation_input_tokens": write,
                          # Real responses report the lifetime split; without it
                          # the cost model refuses to price the row at all.
                          "cache_creation": {"ephemeral_5m_input_tokens": write,
                                             "ephemeral_1h_input_tokens": 0}})


def session(turns=60, prefix=100_000, grow=2_000, rebuild_every=None,
            switch_every=None, session_id="s1"):
    """A long agent session, optionally tearing down its prefix."""
    out, p = [], prefix
    for t in range(turns):
        cold = t == 0 or (rebuild_every and t % rebuild_every == 0)
        model = ("claude-fable-5" if switch_every and (t // switch_every) % 2
                 else "claude-opus-5")
        if switch_every and t and t % switch_every == 0:
            cold = True
        out.append(req(t, read=0 if cold else p, write=p if cold else grow,
                       session=session_id, model=model))
        p += grow
    return out


def findings(reqs, **kw):
    ts = TraceSet(requests=reqs, tier=Tier.USAGE_ONLY, source="test")
    return {f.code: f for f in analyzer.analyze(ts, allow_unreconciled=True, **kw).findings}


class TestRebuildIsDistinguishedFromExtension(unittest.TestCase):

    def test_a_session_that_only_extends_is_not_reported(self):
        self.assertNotIn("REB-1", findings(session()))

    def test_a_session_rebuilding_every_ten_turns_is(self):
        """Not exactly ten: the first turn of a session has nothing before it to
        compare against, so it is never countable as a rebuild."""
        import re
        f = findings(session(rebuild_every=10))["REB-1"]
        n = int(re.search(r"every (\d+) turns", f.detail).group(1))
        self.assertGreaterEqual(n, 10)
        self.assertLessEqual(n, 13)

    def test_the_severity_tracks_how_often(self):
        self.assertEqual(findings(session(rebuild_every=8))["REB-1"].severity, "high")
        self.assertEqual(findings(session(rebuild_every=25))["REB-1"].severity, "medium")

    def test_rare_rebuilds_are_left_alone(self):
        self.assertNotIn("REB-1", findings(session(turns=200, rebuild_every=100)))

    def test_it_needs_enough_turns_before_it_speaks(self):
        self.assertNotIn("REB-1", findings(session(turns=6, rebuild_every=2)))

    def test_a_model_switch_is_not_reported_as_a_prefix_rebuild(self):
        """Switching model is a separate cache pool by design.

        REB-1 used to fire here and attribute the rebuilds in prose while
        counting them in every number -- `turns`, the interval, the severity and
        affected_requests. So a trace whose rebuilds were entirely explained
        still arrived as a rebuild finding telling the reader to go hunting for
        compaction. Excluded now means excluded, and when nothing is left the
        rule says nothing and SPL-1 owns the trace.
        """
        got = findings(session(turns=60, switch_every=10))
        self.assertNotIn("REB-1", got)
        self.assertIn("SPL-1", got, "nothing names the model switch")

    def test_a_mixed_trace_reports_only_the_unexplained_rebuilds(self):
        """Some rebuilds explained by a switch, some not. The finding has to
        cover the remainder and only the remainder."""
        f = findings(session(turns=120, rebuild_every=10, switch_every=40))["REB-1"]
        self.assertIn("no innocent explanation", f.detail)
        self.assertIn("model change", f.detail)

    def test_rebuilds_with_no_model_change_are_called_unexplained(self):
        f = findings(session(rebuild_every=10))["REB-1"]
        self.assertIn("no innocent explanation", f.detail)
        self.assertIn("changing the prefix itself", f.detail)
        self.assertIn("compaction", f.fix.lower())

    def test_the_count_and_the_cause_come_before_the_exclusions(self):
        """The exclusions used to sit between the count and the conclusion, so
        the sentence saying what is actually wrong landed last. Real caveats,
        wrong position."""
        f = findings(session(turns=120, rebuild_every=10, switch_every=30))["REB-1"]
        self.assertLess(f.detail.index("no innocent explanation"),
                        f.detail.index("Excluded:"),
                        "the cause is buried behind the caveats again")

    def test_it_works_with_no_prompt_structure_at_all(self):
        """The reason this rule matters: it needs three usage counters."""
        reqs = session(rebuild_every=10)
        self.assertTrue(all(not r.segments for r in reqs))
        self.assertIn("REB-1", findings(reqs))

    def test_sessions_are_not_pooled_across_tenants(self):
        """Session ids are not globally unique in a shared gateway export."""
        a = session(session_id="shared")
        b = session(session_id="shared")
        for r in b:
            r.tenant = "other"
        self.assertNotIn("REB-1", findings(a + b))


class TestRebuildNeedsSessionIdentity(unittest.TestCase):
    """Telling an extended prefix from a rebuilt one means knowing which request
    followed which. Assuming unrelated calls are one conversation invents the
    most expensive finding this tool makes."""

    def _sessionless(self, n=80):
        return [Request(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                        model="claude-opus-5",
                        usage={"input_tokens": 0, "cache_read_input_tokens": 0,
                               "cache_creation_input_tokens": 50_000,
                               "cache_creation": {"ephemeral_5m_input_tokens": 50_000,
                                                  "ephemeral_1h_input_tokens": 0}})
                for i in range(n)]

    def test_the_runtime_does_not_report_a_conversation_that_does_not_exist(self):
        m, fired = monitor.Monitor(), []
        for r in self._sessionless():
            fired += m.observe_usage(r)
        self.assertNotIn("RT-REBUILD", {a.code for a in fired})

    def test_the_runtime_says_the_check_is_off_rather_than_going_quiet(self):
        m, fired = monitor.Monitor(), []
        for r in self._sessionless():
            fired += m.observe_usage(r)
        self.assertIn("RT-NOSESSION", {a.code for a in fired})

    def test_the_report_says_the_same_thing(self):
        f = findings(self._sessionless())
        self.assertNotIn("REB-1", f)
        self.assertIn("REB-0", f)
        self.assertIn("which request followed", f["REB-0"].detail)

    def test_both_agree_once_a_session_is_supplied(self):
        reqs = self._sessionless()
        for r in reqs:
            r.session = "s1"
        m, fired = monitor.Monitor(), []
        for r in reqs:
            fired += m.observe_usage(r)
        self.assertIn("RT-REBUILD", {a.code for a in fired})
        self.assertIn("REB-1", findings(reqs))


class TestTheVerdictAnswersTheCommonMisreading(unittest.TestCase):

    def test_a_working_cache_is_reported_as_working(self):
        f = findings(session())["CAC-1"]
        self.assertEqual(f.severity, "low")
        self.assertIn("paying for itself", f.title)

    def test_it_states_the_counterfactual_not_just_the_share(self):
        f = findings(session())["CAC-1"]
        self.assertIn("would cost", f.detail)
        self.assertIn("says nothing about", f.detail)

    def test_a_cache_share_of_the_bill_alone_is_called_out_as_uninformative(self):
        f = findings(session())["CAC-1"]
        self.assertIn("reads bill at a tenth", f.detail)

    def test_it_reports_the_read_share_against_a_healthy_baseline(self):
        f = findings(session())["CAC-1"]
        self.assertIn("75%", f.detail)

    def test_a_cache_that_loses_money_is_not_congratulated(self):
        """Writes that nothing reads bill above the plain input rate."""
        reqs = [req(i, read=0, write=50_000) for i in range(30)]
        f = findings(reqs)["CAC-1"]
        self.assertEqual(f.severity, "high")
        self.assertIn("costs more", f.title)
        self.assertIn("adding", f.detail)

    def test_it_recommends_doing_nothing_when_nothing_is_wrong(self):
        self.assertIn("No action indicated", findings(session())["CAC-1"].fix)

    def test_it_carries_no_dollar_figure(self):
        """It is a ratio between two priced totals. The ratio is the finding."""
        self.assertIsNone(findings(session())["CAC-1"].avoidable_usd_month)


class TestTheRuntimeAgreesWithTheReport(unittest.TestCase):
    """A monitor that disagrees with the report is worse than no monitor."""

    def _alerts(self, reqs):
        m, out = monitor.Monitor(), []
        for r in reqs:
            out += m.observe(r)
        return {a.code: a for a in out}

    def test_both_see_the_same_rebuilding_session(self):
        reqs = session(rebuild_every=10)
        self.assertIn("REB-1", findings(reqs))
        self.assertIn("RT-REBUILD", self._alerts(reqs))

    def test_both_stay_quiet_on_a_healthy_session(self):
        reqs = session()
        self.assertNotIn("REB-1", findings(reqs))
        self.assertNotIn("RT-REBUILD", self._alerts(reqs))

    def test_they_share_one_definition_of_a_rebuild(self):
        self.assertEqual(analyzer.REBUILD_FRACTION, monitor.REBUILD_FRACTION)

    def test_the_runtime_reports_a_comparable_interval(self):
        """The runtime sees a rolling window, so it will not match the batch
        figure exactly. It has to land close enough to mean the same thing."""
        import re
        a = self._alerts(session(rebuild_every=10))["RT-REBUILD"]
        n = int(re.search(r"every (\d+) turns", a.summary).group(1))
        self.assertGreaterEqual(n, 8)
        self.assertLessEqual(n, 13)

    def test_the_usage_only_notice_no_longer_claims_this_check_is_off(self):
        blind = self._alerts(session())["RT-BLIND"]
        self.assertIn("RT-REBUILD still", blind.detail)


class TestRuntimeStateStaysBounded(unittest.TestCase):

    def test_per_session_prefix_sizes_are_capped(self):
        m = monitor.Monitor()
        for i in range(5000):
            m.observe(req(i, read=1000, write=100, session=f"s{i}"))
        st = m._scopes[(None, "anthropic/direct", "claude-opus-5")]
        self.assertLessEqual(len(st.established), monitor.MAX_FIRING)
        self.assertLessEqual(len(st.rebuilds), monitor.WINDOW)




class TestTranscriptIngestKeepsBilledRows(unittest.TestCase):
    """One assistant record is one billed request. A row the provider charged
    for that this drops is money missing from the denominator, and a report that
    silently analyses less than it was given is the failure the coverage line
    exists to prevent."""

    def _load(self, records):
        import json
        import tempfile
        from cacheeconomics.adapters.claude_code import load_sessions
        root = tempfile.mkdtemp()
        proj = os.path.join(root, "proj")
        os.makedirs(proj)
        with open(os.path.join(proj, "a.jsonl"), "w") as f:
            f.write("\n".join(json.dumps(r) for r in records))
        return load_sessions(root=root)

    def _rec(self, stamp, tokens):
        rec = {"type": "assistant", "sessionId": "s1", "uuid": f"u{tokens}",
               "message": {"model": "claude-opus-5",
                           "usage": {"input_tokens": tokens,
                                     "cache_read_input_tokens": 0,
                                     "cache_creation_input_tokens": 0}}}
        if stamp is not None:
            rec["timestamp"] = stamp
        return rec

    def test_a_billed_row_with_no_timestamp_is_kept(self):
        ts = self._load([self._rec("2026-07-29T09:00:00Z", 10), self._rec(None, 20)])
        self.assertEqual(len(ts), 2)
        self.assertEqual(sum(r.usage["input_tokens"] for r in ts.requests), 30)

    def test_a_malformed_timestamp_does_not_abort_the_load(self):
        """It used to call datetime.fromisoformat directly, so one bad stamp
        anywhere took the whole file with it."""
        ts = self._load([self._rec("2026-07-29T09:00:00Z", 10),
                         self._rec("not-a-timestamp", 20)])
        self.assertEqual(len(ts), 2)

    def test_untimed_rows_are_disclosed_not_absorbed(self):
        ts = self._load([self._rec("2026-07-29T09:00:00Z", 10), self._rec(None, 20)])
        self.assertTrue(any("no usable timestamp" in n for n in ts.notes))

    def test_an_untimed_row_carries_no_invented_time(self):
        ts = self._load([self._rec("2026-07-29T09:00:00Z", 10), self._rec(None, 20)])
        self.assertEqual(sum(1 for r in ts.requests if r.sent_at is None), 1)

    def test_a_clean_file_says_nothing_about_timestamps(self):
        ts = self._load([self._rec("2026-07-29T09:00:00Z", 10),
                         self._rec("2026-07-29T09:01:00Z", 20)])
        self.assertFalse(any("no usable timestamp" in n for n in ts.notes))


class TestSchemaDriftDoesNotBuyConfidence(unittest.TestCase):
    """Four ways a malformed or contradictory export used to be accepted as
    stronger evidence than it was."""

    def _load(self, row):
        import json
        import tempfile
        from cacheeconomics.trace import load_jsonl
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        with open(path, "w") as f:
            f.write(json.dumps(row))
        try:
            return load_jsonl(path)
        finally:
            os.unlink(path)

    def _hid(self, s):
        import hashlib
        import hmac
        return "hmac:" + hmac.new(b"k" * 32, s.encode(), hashlib.sha256).hexdigest()

    def _row(self, segments):
        return {"request_id": "r1", "sent_at": "2026-07-29T09:00:00Z",
                "model": "claude-opus-5",
                "usage": {"input_tokens": 9000, "cache_read_input_tokens": 0,
                          "cache_creation_input_tokens": 0},
                "segments": segments}

    def test_ids_without_token_counts_do_not_earn_the_instrumented_tier(self):
        """An id says what a segment is; only a size says what it costs, and a
        zero-token prompt enters the counterfactual as free."""
        ts = self._load(self._row([
            {"id": self._hid("a"), "role": "system", "index": 0},
            {"id": self._hid("b"), "role": "user", "index": 1}]))
        self.assertEqual(ts.tier.name, "USAGE_ONLY")
        self.assertEqual(ts.structural_coverage, 0.0)

    def test_it_says_why_rather_than_silently_downgrading(self):
        ts = self._load(self._row([{"id": self._hid("a"), "role": "system", "index": 0}]))
        self.assertTrue(any("no numeric token count" in n for n in ts.notes))

    def test_measured_usage_survives_the_downgrade(self):
        """Dropping structure must not drop spend."""
        ts = self._load(self._row([{"id": self._hid("a"), "role": "system", "index": 0}]))
        self.assertEqual(ts.requests[0].usage["input_tokens"], 9000)

    def test_a_complete_export_still_gets_full_confidence(self):
        ts = self._load(self._row([
            {"id": self._hid("a"), "role": "system", "index": 0, "tokens": 9000},
            {"id": self._hid("b"), "role": "user", "index": 1, "tokens": 100}]))
        self.assertEqual(ts.tier.name, "INSTRUMENTED")
        self.assertEqual(ts.structural_coverage, 1.0)


class TestContradictoryLifetimesAreUnprovable(unittest.TestCase):
    """The marker is on the block that was sent; the row field is exporter
    metadata about it. Taking the row's word when they disagree priced 1h
    writes at the 5m rate -- the 38% understatement, in the flattering
    direction, that this path exists to prevent."""

    def _req(self, marker_ttl, row_ttl):
        from cacheeconomics.trace import Segment
        return Request(request_id="r", sent_at=T0, model="claude-opus-5",
                       usage={"cache_creation_input_tokens": 10000},
                       segments=[Segment(id="a", role="system", tokens=9000, index=0,
                                         cache_marked=True, ttl=marker_ttl)],
                       ttl_requested=row_ttl)

    def test_a_contradiction_is_refused(self):
        self.assertIsNone(analyzer._declared_ttl(self._req("1h", "5m")))

    def test_agreement_is_accepted(self):
        self.assertEqual(analyzer._declared_ttl(self._req("1h", "1h")), "1h")

    def test_the_marker_is_used_when_the_row_is_silent(self):
        self.assertEqual(analyzer._declared_ttl(self._req("1h", None)), "1h")

    def test_the_row_is_a_fallback_when_there_is_no_marker(self):
        self.assertEqual(analyzer._declared_ttl(self._req(None, "1h")), "1h")

    def test_an_unprovable_lifetime_is_excluded_rather_than_guessed(self):
        f = findings([self._req("1h", "5m")])
        self.assertNotIn("CAC-1", f)


class TestSubagentsAreTheirOwnContext(unittest.TestCase):
    """Claude Code subagents share the parent's sessionId and run their own
    context. Grouping on session alone measured one context's write against
    another's established prefix, so the answer depended on how the two
    interleaved rather than on what either did."""

    # Twenty seconds a turn, not sixty. At sixty the subagent respawned every
    # 420 seconds against a 300-second lifetime, so its cold write was a
    # genuine expiry and the rule now says so -- the fixture was measuring
    # lifetime rather than the context isolation it claims to test. Tight
    # enough that the entry is still alive, so a cold write can only be a
    # rebuild.
    def _q(self, i, agent, read, write):
        return Request(request_id=f"{agent}-{i}", sent_at=T0 + timedelta(seconds=20 * i),
                       model="claude-opus-5", session="same-session", agent=agent,
                       usage={"input_tokens": 0, "cache_read_input_tokens": read,
                              "cache_creation_input_tokens": write,
                              "cache_creation": {"ephemeral_5m_input_tokens": write,
                                                 "ephemeral_1h_input_tokens": 0}})

    def _mixed(self):
        """A main loop that only extends, and subagents that respawn cold."""
        out, t, pm = [], 0, 200_000
        for _ in range(12):
            for _ in range(4):
                out.append(self._q(t, "main-loop", 0 if t == 0 else pm,
                                   pm if t == 0 else 3000))
                pm += 3000
                t += 1
            ps = 40_000
            for k in range(3):
                out.append(self._q(t, "subagent:explore", 0 if k == 0 else ps,
                                   ps if k == 0 else 1500))
                ps += 1500
                t += 1
        return out

    def test_the_repeated_subagent_cold_start_is_still_found(self):
        """Pooling it with a main loop that was merely extending made this
        genuine finding disappear."""
        self.assertIn("REB-1", findings(self._mixed()))

    def test_a_main_loop_that_only_extends_is_not_accused(self):
        main = [r for r in self._mixed() if r.agent == "main-loop"]
        self.assertNotIn("REB-1", findings(main))

    def test_the_grouping_key_carries_the_agent(self):
        from cacheeconomics.analyzer import _sessions
        keys = _sessions(self._mixed()).keys()
        self.assertEqual(len({k[2] for k in keys}), 2)

    def test_the_runtime_isolates_contexts_the_same_way(self):
        m, fired = monitor.Monitor(), []
        for r in self._mixed():
            fired += m.observe_usage(r)
        self.assertIn("RT-REBUILD", {a.code for a in fired})


class TestOnePairIsEnoughForFanOut(unittest.TestCase):
    """Two concurrent requests writing the same prefix is the minimum case and
    the commonest. Requiring two pairs meant the report stayed silent about
    something the runtime monitor had already alerted on."""

    def _w(self, i, secs):
        from cacheeconomics.trace import Segment
        return Request(request_id=f"f{i}", sent_at=T0 + timedelta(seconds=secs),
                       model="claude-opus-5", session="s",
                       usage={"input_tokens": 0, "cache_read_input_tokens": 0,
                              "cache_creation_input_tokens": 80_000,
                              "cache_creation": {"ephemeral_5m_input_tokens": 80_000,
                                                 "ephemeral_1h_input_tokens": 0}},
                       segments=[Segment(id="sys", role="system", tokens=80_000,
                                         index=0, cache_marked=True, ttl="5m")])

    def _f(self, n):
        reqs = [self._w(i, i) for i in range(n)]
        ts = TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="test")
        return {f.code: f for f in
                analyzer.analyze(ts, allow_unreconciled=True).findings}

    def test_two_concurrent_writers_are_reported(self):
        self.assertIn("FAN-1", self._f(2))

    def test_affected_requests_counts_requests_not_pairs(self):
        """Three requests form two adjacent pairs and were reported as four."""
        self.assertEqual(self._f(3)["FAN-1"].affected_requests, 3)
        self.assertEqual(self._f(2)["FAN-1"].affected_requests, 2)

    def test_spaced_writes_are_still_left_alone(self):
        reqs = [self._w(i, 600 * i) for i in range(4)]
        ts = TraceSet(requests=reqs, tier=Tier.INSTRUMENTED, source="test")
        self.assertNotIn("FAN-1", {f.code for f in
                                   analyzer.analyze(ts, allow_unreconciled=True).findings})

    def test_the_report_and_the_runtime_now_agree_on_two(self):
        m, fired = monitor.Monitor(), []
        for r in [self._w(i, i) for i in range(2)]:
            fired += m.observe(r)
        self.assertIn("RT-FANOUT", {a.code for a in fired})
        self.assertIn("FAN-1", self._f(2))


class TestASubagentIsNotAModelSwitch(unittest.TestCase):
    """A subagent runs its own context, and Claude Code's subagents share the
    parent's sessionId. REB-1 has keyed on agent since it was written; SPL-1 did
    not, so a main loop on one model and a subagent on another read as one
    conversation switching mid-flight -- a finding, and a remediation, for two
    contexts that were never one."""

    def _reqs(self, subagent_model):
        out = []
        for i in range(20):
            for agent, model in (("main", "claude-opus-5"),
                                 ("sub", subagent_model)):
                out.append(Request(
                    request_id=f"{agent}{i}", sent_at=T0 + timedelta(seconds=60 * i),
                    model=model, session="shared", agent=agent,
                    usage={"input_tokens": 10, "cache_read_input_tokens": 5_000,
                           "cache_creation_input_tokens": 0}))
        return out

    def test_two_agents_on_different_models_are_not_a_split_session(self):
        self.assertNotIn("SPL-1", findings(self._reqs("claude-haiku-4-5")))

    def test_one_agent_genuinely_switching_still_is(self):
        """The fix must not silence the real finding."""
        reqs = []
        for i in range(20):
            reqs.append(Request(
                request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                model="claude-opus-5" if i % 2 else "claude-fable-5",
                session="s", agent="main",
                usage={"input_tokens": 10, "cache_read_input_tokens": 5_000,
                       "cache_creation_input_tokens": 0}))
        self.assertIn("SPL-1", findings(reqs))

    def test_the_two_rules_key_sessions_the_same_way(self):
        """The divergence itself, asserted rather than inferred."""
        import inspect
        from cacheeconomics import analyzer
        spl = inspect.getsource(analyzer._f_model_split)
        self.assertIn("r.agent", spl)




class TestExpiryIsExcludedFromTheVeryFirstRepeat(unittest.TestCase):
    """The exclusion needs a previous timestamp, and a session's first request
    was returning before one was recorded. So every session's *second* turn had
    nothing to compare against and an expired entry was counted as a rebuild --
    forty short sessions whose second turn arrived ten minutes later reported
    "rebuilt about every 1 turns", which is exactly the misdiagnosis the
    exclusion exists to prevent."""

    def _short_sessions(self, n=40, gap=600):
        def one(sess, at):
            return Request(request_id=f"{sess}-{at}", sent_at=at,
                           model="claude-opus-5", session=sess, agent="main",
                           ttl_requested="5m",
                           usage={"input_tokens": 0, "cache_read_input_tokens": 0,
                                  "cache_creation_input_tokens": 50_000},
                           segments=[Segment(id="sys", role="system", tokens=50_000,
                                             index=0, cache_marked=True, ttl="5m")])
        out = [one(f"s{i}", T0 + timedelta(seconds=30 * i)) for i in range(n)]
        out += [one(f"s{i}", T0 + timedelta(seconds=gap + 30 * i)) for i in range(n)]
        return out

    def _codes(self, reqs):
        m, fired = monitor.Monitor(), []
        for r in reqs:
            fired += m.observe_usage(r)
        return {a.code for a in fired}

    def test_an_expired_second_turn_is_not_a_rebuild(self):
        self.assertNotIn("RT-REBUILD", self._codes(self._short_sessions()))

    def test_a_second_turn_inside_the_lifetime_still_is(self):
        """The exclusion must not swallow the real case: the entry was alive
        and got rewritten anyway."""
        self.assertIn("RT-REBUILD", self._codes(self._short_sessions(gap=60)))

    def _mixed_lifetime_sessions(self, n=40, gap=4000):
        def one(sess, at):
            return Request(
                request_id=f"{sess}-{at}", sent_at=at,
                model="claude-opus-5", session=sess, agent="main",
                ttl_requested="1h",
                usage={"input_tokens": 0, "cache_read_input_tokens": 0,
                       "cache_creation_input_tokens": 50_000,
                       "cache_creation": {"ephemeral_5m_input_tokens": 10_000,
                                          "ephemeral_1h_input_tokens": 40_000}},
                segments=[Segment(id="sys", role="system", tokens=40_000,
                                  index=0, cache_marked=True, ttl="1h"),
                          Segment(id="turn", role="user", tokens=10_000,
                                  index=1, cache_marked=True)])
        out = [one(f"s{i}", T0 + timedelta(seconds=30 * i)) for i in range(n)]
        out += [one(f"s{i}", T0 + timedelta(seconds=gap + 30 * i)) for i in range(n)]
        return out

    def test_mixed_lifetimes_after_the_longest_ttl_are_expiry_not_rebuild(self):
        reqs = self._mixed_lifetime_sessions()
        self.assertNotIn("RT-REBUILD", self._codes(reqs))
        self.assertNotIn("REB-1", findings(reqs))

    def test_mixed_lifetimes_inside_the_longest_ttl_can_still_be_rebuilds(self):
        reqs = self._mixed_lifetime_sessions(gap=60)
        self.assertIn("RT-REBUILD", self._codes(reqs))
        self.assertIn("REB-1", findings(reqs))


if __name__ == "__main__":
    unittest.main()
