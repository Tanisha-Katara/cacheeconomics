"""Live diagnostics: does it fire when it should, stay quiet when it should,
and stay bounded when it runs for a long time.

The last of those matters most. This is the only component designed to sit
inside a request path for weeks, so a leak here is not a wrong number in a
report, it is a process that dies at 3am.
"""

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cacheeconomics import monitor, registry  # noqa: E402
from cacheeconomics.monitor import WINDOW, Monitor  # noqa: E402
from cacheeconomics.trace import Request, Segment  # noqa: E402

T0 = datetime(2026, 7, 29, 9, tzinfo=timezone.utc)


def sg(i, role, tokens, sid, label="", marked=False, ttl=None):
    return Segment(id=sid, role=role, tokens=tokens, index=i, label=label,
                   cache_marked=marked, ttl=ttl)


def req(i, segments, *, gap=60, model="claude-opus-5", tenant=None,
        target="anthropic/direct", ttl="5m", writes=0, reads=0, at=None,
        session=None):
    return Request(
        request_id=f"r{i}", sent_at=at or (T0 + timedelta(seconds=gap * i)),
        model=model,
        usage={"input_tokens": 100, "cache_creation_input_tokens": writes,
               "cache_read_input_tokens": reads},
        segments=segments, agent="a", tenant=tenant, target_id=target,
        ttl_requested=ttl, session=session)


STABLE = [sg(0, "system", 8000, "sys", "instructions", marked=True, ttl="5m"),
          sg(1, "user", 100, "turn", "user_turn")]


def drifting(i):
    return [sg(0, "system", 300, f"hdr{i}", "session_ctx"),
            sg(1, "system", 8000, "body", "instructions", marked=True, ttl="5m"),
            sg(2, "user", 100, f"t{i}", "user_turn")]


class TestDriftDetection(unittest.TestCase):

    def test_a_changing_position_inside_the_prefix_fires(self):
        m, fired = Monitor(), []
        for i in range(10):
            fired += m.observe(req(i, drifting(i)))
        self.assertIn("RT-DRIFT", [a.code for a in fired])

    def test_a_stable_prefix_stays_quiet(self):
        m, fired = Monitor(), []
        for i in range(20):
            fired += m.observe(req(i, [sg(0, "system", 8000, "sys", marked=True, ttl="5m"),
                                       sg(1, "user", 100, f"t{i}")]))
        self.assertNotIn("RT-DRIFT", [a.code for a in fired])

    def test_content_after_the_marker_is_not_drift(self):
        """The trailing turn changes on every request by design."""
        m, fired = Monitor(), []
        for i in range(20):
            fired += m.observe(req(i, [sg(0, "system", 8000, "sys", marked=True, ttl="5m"),
                                       sg(1, "user", 100, f"changes{i}")]))
        self.assertNotIn("RT-DRIFT", [a.code for a in fired])

    def test_one_change_is_not_yet_drift(self):
        """A single new value is usually a deploy, not a pattern."""
        m, fired = Monitor(), []
        for i in range(10):
            hdr = "before" if i < 5 else "after"
            fired += m.observe(req(i, [sg(0, "system", 300, hdr, "hdr"),
                                       sg(1, "system", 8000, "body", marked=True, ttl="5m")]))
        self.assertNotIn("RT-DRIFT", [a.code for a in fired])

    def test_it_names_the_tokens_behind_the_drifting_segment(self):
        m, fired = Monitor(), []
        for i in range(10):
            fired += m.observe(req(i, drifting(i)))
        drift = next(a for a in fired if a.code == "RT-DRIFT")
        self.assertIn("8,000", drift.detail)


class TestTheWorstDriftPattern(unittest.TestCase):
    """A field that flips between two states changes the prefix on every single
    request. Counting distinct values rated that as calmer than a field with
    three values that changes once a week, and said nothing at all."""

    def test_a_value_alternating_every_request_fires(self):
        m, fired = Monitor(), []
        for i in range(20):
            fired += m.observe(req(i, [sg(0, "system", 300, "A" if i % 2 else "B"),
                                       sg(1, "system", 8000, "body", marked=True, ttl="5m")]))
        self.assertIn("RT-DRIFT", [a.code for a in fired])

    def test_it_reports_the_rate_not_the_number_of_values(self):
        m, fired = Monitor(), []
        for i in range(20):
            fired += m.observe(req(i, [sg(0, "system", 300, "A" if i % 2 else "B"),
                                       sg(1, "system", 8000, "body", marked=True, ttl="5m")]))
        self.assertIn("100%", next(a for a in fired if a.code == "RT-DRIFT").detail)

    def test_per_session_stable_headers_do_not_drift(self):
        m, fired = Monitor(), []
        for i in range(20):
            for sess in ("a", "b"):
                fired += m.observe(req(
                    i, [sg(0, "system", 300, f"hdr-{sess}"),
                        sg(1, "system", 8000, "body", marked=True, ttl="5m")],
                    session=sess))
        self.assertNotIn("RT-DRIFT", [a.code for a in fired])

    def test_a_changing_session_header_still_drifts(self):
        m, fired = Monitor(), []
        for i in range(20):
            fired += m.observe(req(
                i, [sg(0, "system", 300, f"hdr-{i}"),
                    sg(1, "system", 8000, "body", marked=True, ttl="5m")],
                session="changing"))
        self.assertIn("RT-DRIFT", [a.code for a in fired])


class TestStrandedTokenCount(unittest.TestCase):
    """The figure in the alert is the reason somebody acts on it."""

    def test_it_counts_the_whole_prefix_not_just_what_follows(self):
        """A timestamp in the middle of a prefix with no marker below it
        strands what precedes it too — that content stops matching as well."""
        def segs(i):
            return [sg(0, "system", 5000, "policy"),
                    sg(1, "system", 50, f"stamp{i}"),
                    sg(2, "system", 4000, "tools", marked=True, ttl="5m")]
        m, fired = Monitor(), []
        for i in range(10):
            fired += m.observe(req(i, segs(i)))
        detail = next(a for a in fired if a.code == "RT-DRIFT").detail
        self.assertIn("9,000", detail)   # 5,000 above it + 4,000 below it

    def test_a_marker_below_the_drift_protects_what_it_covers(self):
        def segs(i):
            return [sg(0, "system", 5000, "policy", marked=True, ttl="1h"),
                    sg(1, "system", 50, f"stamp{i}"),
                    sg(2, "system", 4000, "tools", marked=True, ttl="5m")]
        m, fired = Monitor(), []
        for i in range(10):
            fired += m.observe(req(i, segs(i)))
        detail = next(a for a in fired if a.code == "RT-DRIFT").detail
        self.assertIn("4,000", detail)
        self.assertNotIn("9,000", detail)


class TestTheCaseWhereNothingIsCachedAtAll(unittest.TestCase):
    """Every check that begins by looking for a cache marker is silent on the
    workload that has none — which is the one losing the most money."""

    def blocked(self, i):
        return [sg(0, "system", 300, f"session{i}", "session_id"),
                sg(1, "system", 9000, "policy", "instructions"),
                sg(2, "tools", 6000, "tools", "tool_defs"),
                sg(3, "user", 200, f"t{i}", "user_turn")]

    def test_a_volatile_segment_in_front_of_a_stable_block_fires(self):
        m, fired = Monitor(), []
        for i in range(20):
            fired += m.observe(req(i, self.blocked(i)))
        self.assertIn("RT-BLOCKED", [a.code for a in fired])

    def test_it_names_the_tokens_stranded_behind_it(self):
        m, fired = Monitor(), []
        for i in range(20):
            fired += m.observe(req(i, self.blocked(i)))
        a = next(x for x in fired if x.code == "RT-BLOCKED")
        self.assertIn("15,000", a.detail)

    def test_the_fix_requires_an_eval_because_it_moves_content(self):
        m, fired = Monitor(), []
        for i in range(20):
            fired += m.observe(req(i, self.blocked(i)))
        a = next(x for x in fired if x.code == "RT-BLOCKED")
        self.assertIn("behavioural eval", a.fix)

    def test_it_stays_quiet_once_a_marker_exists(self):
        """With a marker present this is RT-DRIFT's job, and two alerts for one
        problem is how an alert stream becomes something people mute."""
        m, fired = Monitor(), []
        for i in range(20):
            segs = self.blocked(i)
            segs[1] = sg(1, "system", 9000, "policy", "instructions",
                         marked=True, ttl="5m")
            fired += m.observe(req(i, segs))
        codes = [a.code for a in fired]
        self.assertIn("RT-DRIFT", codes)
        self.assertNotIn("RT-BLOCKED", codes)

    def test_a_stable_block_below_the_minimum_is_not_worth_reporting(self):
        m, fired = Monitor(), []
        for i in range(20):
            fired += m.observe(req(i, [sg(0, "system", 300, f"session{i}"),
                                       sg(1, "system", 200, "small"),
                                       sg(2, "user", 100, f"t{i}")]))
        self.assertNotIn("RT-BLOCKED", [a.code for a in fired])

    def test_an_uncached_but_stable_prompt_stays_quiet(self):
        m, fired = Monitor(), []
        for i in range(20):
            fired += m.observe(req(i, [sg(0, "system", 9000, "policy"),
                                       sg(1, "user", 200, f"t{i}")]))
        self.assertNotIn("RT-BLOCKED", [a.code for a in fired])


class TestASectionThatComesAndGoes(unittest.TestCase):
    """Appearing and disappearing moves everything behind it exactly as an edit
    does. The offline loader has counted absence as a change since it was
    written; the runtime appended nothing at all for a position that was not
    there, so its few observations were identical and it scored zero churn."""

    def optional(self, i):
        segs = [sg(0, "tools", 9000, "tools"), sg(1, "system", 6000, "policy")]
        if i % 2:
            segs.append(sg(2, "system", 4000, "optional"))
        segs.append(sg(9, "user", 200, f"turn{i}"))
        return segs

    def test_a_position_that_vanishes_is_not_stable(self):
        m = Monitor()
        for i in range(40):
            m.observe_shape(req(i, self.optional(i)))
        rates = m.change_rates((None, "anthropic/direct", "claude-opus-5"))
        self.assertEqual(rates[2], 1.0)

    def test_a_position_always_present_is_still_stable(self):
        m = Monitor()
        for i in range(40):
            m.observe_shape(req(i, self.optional(i)))
        rates = m.change_rates((None, "anthropic/direct", "claude-opus-5"))
        self.assertEqual(rates[0], 0.0)
        self.assertEqual(rates[1], 0.0)

    def test_the_runtime_and_the_batch_loader_agree(self):
        """They are the same quantity feeding the same allocator. A monitor that
        disagrees with the report is worse than no monitor."""
        from cacheeconomics.allocate import observed_change_rates_by_pool, pool_of
        reqs = [req(i, self.optional(i), gap=240) for i in range(40)]
        m = Monitor()
        for r in reqs:
            m.observe_shape(r)
        offline = observed_change_rates_by_pool(reqs)[pool_of(reqs[0])]
        runtime = m.change_rates((None, "anthropic/direct", "claude-opus-5"))
        self.assertEqual({k: round(v, 6) for k, v in runtime.items()},
                         {k: round(v, 6) for k, v in offline.items()})

    def test_tracked_positions_are_capped(self):
        from cacheeconomics.monitor import MAX_POSITIONS
        m = Monitor()
        for i in range(4):
            m.observe_shape(req(i, [sg(j, "system", 10, f"s{j}")
                                    for j in range(MAX_POSITIONS + 200)]))
        st = m._scopes[(None, "anthropic/direct", "claude-opus-5")]
        self.assertLessEqual(len(st.seg_values), MAX_POSITIONS)

    def test_reading_an_untracked_position_does_not_start_tracking_it(self):
        """Otherwise the cap above is decorative."""
        from cacheeconomics.monitor import MAX_POSITIONS
        m = Monitor()
        for i in range(12):
            m.observe(req(i, [sg(j, "system", 10, f"s{j}")
                              for j in range(MAX_POSITIONS + 200)]))
        st = m._scopes[(None, "anthropic/direct", "claude-opus-5")]
        self.assertLessEqual(len(st.seg_values), MAX_POSITIONS)


class TestAlertsDoNotSpam(unittest.TestCase):

    def test_a_persistent_condition_fires_once(self):
        m, fired = Monitor(), []
        for i in range(50):
            fired += m.observe(req(i, drifting(i)))
        self.assertEqual(sum(1 for a in fired if a.code == "RT-DRIFT"), 1)

    def test_clearing_lets_it_fire_again(self):
        m, fired = Monitor(), []
        for i in range(10):
            fired += m.observe(req(i, drifting(i)))
        m.clear((None, "anthropic/direct", "claude-opus-5"), "RT-DRIFT")
        again = []
        for i in range(10, 20):
            again += m.observe(req(i, drifting(i)))
        self.assertIn("RT-DRIFT", [a.code for a in again])

    def test_scopes_are_independent(self):
        """One tenant drifting must not silence the alert for another."""
        m, fired = Monitor(), []
        for i in range(10):
            fired += m.observe(req(i, drifting(i), tenant="A"))
        for i in range(10):
            fired += m.observe(req(i, drifting(i), tenant="B"))
        self.assertEqual(sum(1 for a in fired if a.code == "RT-DRIFT"), 2)


class TestMinimumAndBudget(unittest.TestCase):

    def test_a_marker_below_the_minimum_fires(self):
        m = Monitor()
        alerts = m.observe(req(0, [sg(0, "system", 100, "tiny", marked=True, ttl="5m")],
                               model="claude-haiku-4-5"))
        codes = [a.code for a in alerts]
        self.assertIn("RT-MIN", codes)

    def test_a_prefix_above_the_minimum_stays_quiet(self):
        m = Monitor()
        alerts = m.observe(req(0, [sg(0, "system", 9000, "big", marked=True, ttl="5m")],
                               model="claude-haiku-4-5"))
        self.assertNotIn("RT-MIN", [a.code for a in alerts])

    def test_an_unregistered_model_abstains_rather_than_guessing(self):
        m = Monitor()
        alerts = m.observe(req(0, [sg(0, "system", 100, "tiny", marked=True, ttl="5m")],
                               model="claude-not-registered"))
        self.assertNotIn("RT-MIN", [a.code for a in alerts])

    def test_using_the_whole_marker_budget_fires(self):
        segs = [sg(i, "system", 3000, f"s{i}", marked=True, ttl="5m") for i in range(4)]
        alerts = Monitor().observe(req(0, segs))
        self.assertIn("RT-BUDGET", [a.code for a in alerts])

    def test_exceeding_the_marker_budget_is_an_error_not_headroom_pressure(self):
        segs = [sg(i, "system", 3000, f"s{i}", marked=True, ttl="5m")
                for i in range(5)]
        alert = next(a for a in Monitor().observe(req(0, segs))
                     if a.code == "RT-BUDGET")
        self.assertEqual("high", alert.severity)
        self.assertIn("exceeds", alert.summary)
        self.assertIn("request-level error", alert.detail)

    def test_headroom_stays_quiet(self):
        segs = [sg(i, "system", 3000, f"s{i}", marked=True, ttl="5m") for i in range(2)]
        self.assertNotIn("RT-BUDGET", [a.code for a in Monitor().observe(req(0, segs))])


class TestCadenceAgainstLifetime(unittest.TestCase):

    def _run(self, gap, ttl):
        m, fired = Monitor(), []
        for i in range(20):
            fired += m.observe(req(i, [sg(0, "system", 8000, "sys", marked=True, ttl=ttl),
                                       sg(1, "user", 100, f"t{i}")], gap=gap, ttl=ttl))
        return [a for a in fired if a.code == "RT-TTL"]

    def test_five_minute_ttl_with_in_band_cadence_fires(self):
        a = self._run(gap=900, ttl="5m")
        self.assertEqual(len(a), 1)
        self.assertIn("one-hour", a[0].fix)

    def test_one_hour_ttl_with_sparse_traffic_fires_the_other_way(self):
        a = self._run(gap=7200, ttl="1h")
        self.assertEqual(len(a), 1)
        self.assertIn("five-minute", a[0].fix)

    def test_fast_traffic_on_five_minutes_stays_quiet(self):
        self.assertEqual(self._run(gap=30, ttl="5m"), [])

    def test_in_band_traffic_already_on_one_hour_stays_quiet(self):
        self.assertEqual(self._run(gap=900, ttl="1h"), [])

    def test_it_waits_for_enough_samples(self):
        """A median over three requests is not a cadence."""
        m, fired = Monitor(), []
        for i in range(3):
            fired += m.observe(req(i, STABLE, gap=900))
        self.assertEqual([a for a in fired if a.code == "RT-TTL"], [])


class TestTheCadenceEvidenceIsTheLifetimesOwn(unittest.TestCase):
    """RT-TTL evaluates 5m and 1h. It abstains on any other lifetime -- but the
    rewrite-gap window it reads was filled by every marked request regardless.

    Measured: ten 30m requests ten minutes apart on `anthropic/direct` built
    nine gaps, and the very next 5m request emitted RT-TTL reporting a median
    "over the last 10" -- off ten observations of traffic the rule says it
    cannot read, and zero observations at 5m. Gating the wording on the
    lifetime while leaving the state pooled is the worse half of the original
    defect: silence became a confident number.
    """

    def _at(self, i, ttl, minutes):
        return req(i, [sg(0, "system", 8000, "sys", marked=True, ttl=ttl),
                       sg(1, "user", 100, f"t{i}")],
                   ttl=ttl, at=T0 + timedelta(minutes=minutes), session="s")

    def test_a_lifetime_it_cannot_evaluate_builds_no_evidence(self):
        m, fired = Monitor(), []
        minutes = 0
        for i in range(10):
            fired += m.observe(self._at(i, "30m", minutes))
            minutes += 10
        late = m.observe(self._at(99, "5m", minutes))
        self.assertEqual([], [a for a in fired + late if a.code == "RT-TTL"])

    def test_the_buckets_stay_empty_for_an_unevaluable_lifetime(self):
        m = Monitor()
        minutes = 0
        for i in range(10):
            m.observe(self._at(i, "30m", minutes))
            minutes += 10
        st = next(s for k, s in m._scopes.items() if len(k) > 3)
        self.assertEqual({k: len(v) for k, v in st.rewrite_gaps.items()},
                         {"5m": 0, "1h": 0})

    def test_one_lifetimes_history_does_not_answer_for_another(self):
        """A scope can legitimately carry both -- a durable prefix under an
        advancing turn is a documented pattern -- so the fix is to partition
        the evidence, not to refuse a mixed scope."""
        m, fired = Monitor(), []
        minutes = 0
        for i in range(14):            # sparse 1h traffic: fires the 1h way
            fired += m.observe(self._at(i, "1h", minutes))
            minutes += 70
        for i in range(14):            # in-band 5m traffic: fires the 5m way
            fired += m.observe(self._at(100 + i, "5m", minutes))
            minutes += 10
        by_subject = {a.subject for a in fired if a.code == "RT-TTL"}
        self.assertEqual({"to-5m", "to-1h"}, by_subject)

    def test_the_alert_says_how_many_observations_at_that_lifetime(self):
        m, fired = Monitor(), []
        minutes = 0
        for i in range(14):
            fired += m.observe(self._at(i, "5m", minutes))
            minutes += 10
        a = next(x for x in fired if x.code == "RT-TTL")
        self.assertIn("at this lifetime", a.detail)

    def test_a_lifetime_the_surface_does_not_offer_is_not_recommended(self):
        """`TTL_SECONDS` is what this module can reason about;
        `registry.supported_ttls` is what the surface will accept. Checking
        only the first told an operator to set a one-hour TTL on
        `openai/direct`, whose row advertises 30m and nothing else -- a switch
        the provider would reject, recommended with full confidence off twenty
        good observations."""
        self.assertEqual(["30m"],
                         registry.supported_ttls("openai/direct", "claude-opus-5"))
        m, fired = Monitor(), []
        minutes = 0
        for i in range(20):
            fired += m.observe(req(
                i, [sg(0, "system", 8000, "sys", marked=True, ttl="5m"),
                    sg(1, "user", 100, f"t{i}")],
                ttl="5m", target="openai/direct",
                at=T0 + timedelta(minutes=minutes), session="s"))
            minutes += 10
        self.assertEqual([], [a for a in fired if a.code == "RT-TTL"])

    def test_a_surface_that_offers_it_still_gets_the_recommendation(self):
        """The gate must not silence the check everywhere. `anthropic/direct`
        offers both lifetimes."""
        self.assertIn("1h", registry.supported_ttls("anthropic/direct",
                                                    "claude-opus-5"))
        m, fired = Monitor(), []
        minutes = 0
        for i in range(14):
            fired += m.observe(self._at(i, "5m", minutes))
            minutes += 10
        self.assertIn("RT-TTL", [a.code for a in fired])

    def test_the_lifetime_it_recommends_must_also_be_offered(self):
        """Proving the lifetime in force is supported says nothing about the
        one being recommended. `amazon-bedrock/converse` + `claude-opus-5` is
        offered 5m and nothing else -- 1h never went GA for that model -- and
        a ten-minute 5m rewrite cadence there still emitted `subject='to-1h'`
        with `fix='Set a one-hour TTL...'`, advice pointing at a combination
        the registry says is rejected.

        Third round in a row where the fix covered one direction of a two-sided
        thing: wording not state, invent not destroy, source not destination.
        """
        self.assertEqual(["5m"], registry.supported_ttls(
            "amazon-bedrock/converse", "claude-opus-5"))
        self.assertEqual([], self._cadence("amazon-bedrock/converse",
                                           "claude-opus-5"))

    def test_a_model_that_is_offered_the_destination_still_gets_it(self):
        """The control, so the gate is not a blanket silence. Same surface,
        same cadence, a model for which 1h did go GA."""
        self.assertIn("1h", registry.supported_ttls(
            "amazon-bedrock/converse", "claude-haiku-4-5"))
        fired = self._cadence("amazon-bedrock/converse", "claude-haiku-4-5")
        self.assertEqual(["to-1h"], [a.subject for a in fired])

    def _cadence(self, target, model, ttl="5m", step=10, n=16):
        """A rewrite cadence inside the window RT-TTL argues about."""
        m, fired = Monitor(), []
        minutes = 0
        for i in range(n):
            fired += m.observe(req(
                i, [sg(0, "system", 8000, "sys", marked=True, ttl=ttl),
                    sg(1, "user", 100, f"t{i}")],
                ttl=ttl, target=target, model=model,
                at=T0 + timedelta(minutes=minutes), session="s"))
            minutes += step
        return [a for a in fired if a.code == "RT-TTL"]

    def test_a_surface_that_cannot_answer_reports_the_gap(self):
        """Abstaining is right; abstaining silently is the defect this whole
        seam exists for. An unnamed surface cannot say which lifetimes it
        takes, so RT-TTL goes quiet and RT-NOSURFACE has to say why."""
        m, fired = Monitor(), []
        minutes = 0
        for i in range(14):
            fired += m.observe(req(
                i, [sg(0, "system", 8000, "sys", marked=True, ttl="5m"),
                    sg(1, "user", 100, f"t{i}")],
                ttl="5m", target=registry.UNATTRIBUTED,
                at=T0 + timedelta(minutes=minutes), session="s"))
            minutes += 10
        self.assertEqual([], [a for a in fired if a.code == "RT-TTL"])
        a = next(x for x in fired if x.code == "RT-NOSURFACE")
        self.assertIn("supported_ttls", a.detail)
        self.assertIn("RT-TTL", a.detail)

    def test_an_unevaluable_lifetime_cannot_erase_evaluable_evidence(self):
        """Partitioning the gaps while leaving one shared span->timestamp map
        moved the defect rather than removing it: a 30m write on a span
        overwrote the 5m timestamp, so the next 5m request had nothing to
        measure from. Measured: a 5m stream that records 19 gaps and fires
        recorded 0 and never fired once identical 30m writes were interleaved
        between the same requests."""
        def run(interleave):
            m, fired = Monitor(), []
            minutes, i = 0, 0
            for _ in range(20):
                fired += m.observe(self._at(i, "5m", minutes))
                i += 1
                minutes += 10
                if interleave:
                    fired += m.observe(self._at(i, "30m", minutes))
                    i += 1
                    minutes += 10
            st = next(s for k, s in m._scopes.items() if len(k) > 3)
            return ([a.code for a in fired].count("RT-TTL"),
                    len(st.rewrite_gaps["5m"]))

        self.assertEqual(run(False), run(True))
        self.assertEqual(1, run(True)[0])

    def test_the_timeline_map_is_partitioned_and_capped_per_lifetime(self):
        """Bounded by `len(TTL_SECONDS) * MAX_FIRING`, and neither bucket may
        evict the other."""
        m = Monitor()
        minutes = 0
        for i in range(monitor.MAX_FIRING * 3):
            # A fresh span every request, so the cap is actually exercised.
            m.observe(req(i, [sg(0, "system", 8000, f"sys{i}", marked=True,
                                 ttl="5m"),
                              sg(1, "user", 100, f"t{i}")],
                          ttl="5m", at=T0 + timedelta(minutes=minutes),
                          session="s"))
            minutes += 10
        st = next(s for k, s in m._scopes.items() if len(k) > 3)
        self.assertEqual(sorted(monitor.TTL_SECONDS), sorted(st.last_marked_at))
        for name, marks in st.last_marked_at.items():
            with self.subTest(lifetime=name):
                self.assertLessEqual(len(marks), monitor.MAX_FIRING)

    def test_the_bucket_map_cannot_grow(self):
        """Keys come only from `_ttl_rt_ttl_can_read`, which returns a member
        of TTL_SECONDS or None. A lifetime nobody evaluates must not create a
        bucket, or a long-lived process grows one per string it ever sees."""
        m = Monitor()
        minutes = 0
        for i, ttl in enumerate(["30m", "12h", "1d", "5m", "1h", None] * 6):
            m.observe(self._at(i, ttl, minutes))
            minutes += 10
        for k, st in m._scopes.items():
            with self.subTest(scope=k):
                self.assertEqual(sorted(monitor.TTL_SECONDS),
                                 sorted(st.rewrite_gaps))
                self.assertEqual(sorted(monitor.TTL_SECONDS),
                                 sorted(st.last_marked_at))


class TestColdFanOut(unittest.TestCase):

    def test_concurrent_writers_of_one_prefix_fire(self):
        m, fired = Monitor(), []
        for i in range(3):
            fired += m.observe(req(i, STABLE, writes=8000,
                                   at=T0 + timedelta(seconds=i)))
        self.assertIn("RT-FANOUT", [a.code for a in fired])

    def test_spaced_writes_stay_quiet(self):
        m, fired = Monitor(), []
        for i in range(3):
            fired += m.observe(req(i, STABLE, writes=8000,
                                   at=T0 + timedelta(seconds=60 * i)))
        self.assertNotIn("RT-FANOUT", [a.code for a in fired])

    def test_it_reports_the_number_of_writers_there_actually_were(self):
        """Counting the current request after adding it to the window claimed
        one more writer than existed. Two concurrent writes are two."""
        m, fired = Monitor(), []
        for i in range(2):
            fired += m.observe(req(i, STABLE, writes=8000,
                                   at=T0 + timedelta(seconds=i)))
        a = next(x for x in fired if x.code == "RT-FANOUT")
        self.assertIn("2 concurrent", a.summary)

    def test_fan_out_on_a_different_prefix_is_a_new_alert(self):
        other = [sg(0, "system", 8000, "other-sys", marked=True, ttl="5m")]
        m, fired = Monitor(), []
        for i in range(2):
            fired += m.observe(req(i, STABLE, writes=8000, at=T0 + timedelta(seconds=i)))
        for i in range(2):
            fired += m.observe(req(i, other, writes=8000, at=T0 + timedelta(seconds=i)))
        self.assertEqual(sum(1 for a in fired if a.code == "RT-FANOUT"), 2)

    def test_it_fires_when_responses_complete_out_of_order(self):
        """Usage arrives when responses come *back*, so the live path observes
        these in completion order. Requiring the earlier write to have been seen
        first made an out-of-order completion read its own concurrent partner as
        being in the future, and two simultaneous requests produced no alert at
        all -- which is precisely the case this check exists for."""
        m, fired = Monitor(), []
        for i in (2, 1):                       # later-sent completes first
            fired += m.observe_usage(req(i, STABLE, writes=8000,
                                         at=T0 + timedelta(seconds=i)))
        self.assertIn("RT-FANOUT", [a.code for a in fired])

    def test_the_count_is_the_same_whichever_order_they_complete_in(self):
        def run(order):
            m, out = Monitor(), []
            for i in order:
                out += m.observe_usage(req(i, STABLE, writes=8000,
                                           at=T0 + timedelta(seconds=i)))
            return next(a.summary for a in out if a.code == "RT-FANOUT")
        self.assertEqual(run([1, 2]), run([2, 1]))

    def test_spaced_writes_stay_quiet_in_either_order(self):
        for order in ([1, 2], [2, 1]):
            m, out = Monitor(), []
            for i in order:
                out += m.observe_usage(req(i, STABLE, writes=8000,
                                           at=T0 + timedelta(seconds=600 * i)))
            self.assertNotIn("RT-FANOUT", [a.code for a in out])

    def test_concurrent_reads_are_not_fan_out(self):
        """Reading concurrently is the cache working, not duplicated writes."""
        m, fired = Monitor(), []
        for i in range(3):
            fired += m.observe(req(i, STABLE, reads=8000,
                                   at=T0 + timedelta(seconds=i)))
        self.assertNotIn("RT-FANOUT", [a.code for a in fired])


class TestItStaysBounded(unittest.TestCase):
    """The only component meant to live inside a request path for weeks."""

    def test_per_scope_history_is_capped(self):
        m = Monitor()
        for i in range(WINDOW * 20):
            m.observe(req(i, [sg(0, "system", 8000, f"changes{i}", marked=True, ttl="5m")]))
        st = m._scopes[(None, "anthropic/direct", "claude-opus-5")]
        self.assertLessEqual(len(st.seg_values[0]), WINDOW)
        self.assertLessEqual(len(st.gaps), WINDOW)
        self.assertLessEqual(len(st.recent_writes), WINDOW)

    def test_the_scope_map_itself_is_capped(self):
        """Bounding each scope's history and then keeping one scope per tenant
        forever is the same leak with extra steps."""
        m = Monitor(max_scopes=32)
        for i in range(5000):
            m.observe(req(i, STABLE, tenant=f"t{i}"))
        self.assertEqual(m.scopes, 32)

    def test_the_live_scope_survives_a_flood_of_one_off_tenants(self):
        m = Monitor(max_scopes=32)
        for i in range(200):
            m.observe(req(i, drifting(i), tenant="steady"))
            m.observe(req(i, STABLE, tenant=f"burst{i}"))
        self.assertIn(("steady", "anthropic/direct", "claude-opus-5"), m._scopes)

    def test_the_dedup_table_is_capped(self):
        m = Monitor()
        for i in range(500):
            m.observe(req(i, [sg(0, "system", 8000, f"prefix{i}", marked=True, ttl="5m")],
                          writes=8000, at=T0 + timedelta(milliseconds=i)))
        st = m._scopes[(None, "anthropic/direct", "claude-opus-5")]
        self.assertLessEqual(len(st.firing), 64)

    def test_a_second_drifting_segment_is_still_reported(self):
        """The first problem found must not permanently silence the next one."""
        m, fired = Monitor(), []
        for i in range(12):
            fired += m.observe(req(i, [sg(0, "system", 300, f"a{i}"),
                                       sg(1, "system", 300, "stable"),
                                       sg(2, "system", 8000, "body", marked=True, ttl="5m")]))
        for i in range(12, 40):
            fired += m.observe(req(i, [sg(0, "system", 300, "a-frozen"),
                                       sg(1, "system", 300, f"b{i}"),
                                       sg(2, "system", 8000, "body", marked=True, ttl="5m")]))
        subjects = {a.subject for a in fired if a.code == "RT-DRIFT"}
        self.assertEqual(subjects, {"seg0", "seg1"})

    def test_a_long_run_does_not_accumulate_alerts(self):
        m, fired = Monitor(), []
        for i in range(2000):
            fired += m.observe(req(i, drifting(i)))
        self.assertLess(len(fired), 10, "a persistent condition must not re-fire forever")

    def test_it_never_emits_a_dollar_figure(self):
        """There is no invoice at runtime, and this project does not publish a
        number nobody has reconciled."""
        m, fired = Monitor(), []
        for i in range(30):
            fired += m.observe(req(i, drifting(i), writes=8000))
        for a in fired:
            self.assertNotIn("$", a.summary + a.detail + a.fix)


class TestSilenceIsNotAPass(unittest.TestCase):
    """On a usage-only stream every structural check abstains. Saying nothing
    at all lets that read as a clean bill of health."""

    def test_a_stream_with_no_structure_says_so(self):
        m, fired = Monitor(), []
        for i in range(50):
            fired += m.observe(req(i, []))
        self.assertIn("RT-BLIND", [a.code for a in fired])

    def test_each_missing_input_is_named_separately(self):
        """No prompt structure and no session id are two different gaps, and a
        reader fixing one wants to know the other is still open.

        `writes=` is now supplied because the session gap only matters to
        somebody paying for cache writes -- see the test below."""
        m, fired = Monitor(), []
        for i in range(50):
            fired += m.observe(req(i, [], writes=200_000))
        self.assertEqual(sorted({a.code for a in fired}), ["RT-BLIND", "RT-NOSESSION"])

    def test_it_does_not_ask_for_a_session_when_nothing_is_cached(self):
        """RT-NOSESSION says "rebuild detection is off". That is worth telling
        somebody who is paying for cache writes, and pure noise for a workload
        that never caches -- there is no rebuild to miss. The batch twin has
        always gated REB-0 on writes existing; this is the runtime agreeing."""
        m, fired = Monitor(), []
        for i in range(50):
            fired += m.observe(req(i, [], writes=0, reads=0))
        self.assertNotIn("RT-NOSESSION", {a.code for a in fired})

    def test_supplying_a_session_closes_that_one(self):
        m, fired = Monitor(), []
        for i in range(50):
            r = req(i, [])
            r.session = "s1"
            fired += m.observe(r)
        self.assertNotIn("RT-NOSESSION", {a.code for a in fired})

    def test_it_does_not_nag_a_stream_that_has_structure(self):
        m, fired = Monitor(), []
        for i in range(20):
            fired += m.observe(req(i, STABLE))
        self.assertNotIn("RT-BLIND", [a.code for a in fired])

    def test_it_does_not_claim_the_stream_is_healthy(self):
        alert = Monitor().observe(req(0, []))[0]
        self.assertIn("unmeasured, not healthy", alert.detail)


class TestItAgreesWithTheBatchAnalyzer(unittest.TestCase):
    """A monitor that disagrees with the report is worse than no monitor."""

    def test_the_demo_fixture_drift_is_seen_by_both(self):
        from cacheeconomics.analyzer import analyze
        from cacheeconomics.trace import load_jsonl
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "fixtures", "demo-traces.jsonl")
        ts = load_jsonl(path)
        m, fired = Monitor(), []
        for r in sorted(ts.analysable, key=lambda r: r.sent_at):
            fired += m.observe(r)
        codes = {a.code for a in fired}
        batch = {f.code for f in analyze(ts, invoice_usd=17.45).findings}
        # The fixture plants a below-minimum marker; both paths must see it.
        self.assertIn("MIN-1", batch)
        self.assertIn("RT-MIN", codes)


class TestAModelSwitchIsStillARebuild(unittest.TestCase):
    """The runtime keyed rebuild state by cache pool, which carries the model.
    So a session that switched model landed in a fresh scope with empty history
    and the rebuild vanished -- while RT-REBUILD's own fix text names "a model
    switch" as one of the causes it reports.

    Measured: a 60-turn session alternating model every 10 turns produced REB-1
    from the analyzer and silence from the monitor. The batch rule has always
    compared consecutive requests within a session regardless of model, because
    switching model *is* a way to pay for the prefix twice.
    """

    def _codes(self, switch_every=None):
        m, fired = Monitor(), []
        for i in range(60):
            model = "claude-opus-5"
            if switch_every and (i // switch_every) % 2:
                model = "claude-sonnet-5"
            r = req(i, [], gap=120, model=model, writes=200_000, session="s")
            fired += m.observe(r)
        return {a.code for a in fired}

    def test_a_switching_session_still_reports_rebuilds(self):
        self.assertIn("RT-REBUILD", self._codes(switch_every=10))

    def test_a_single_model_session_is_unaffected(self):
        self.assertIn("RT-REBUILD", self._codes())

    def test_the_runtime_agrees_with_the_batch_rule(self):
        """The promise the monitor's own docstring makes: replay it against a
        captured trace and get the same answer as the report."""
        from cacheeconomics.analyzer import analyze
        from cacheeconomics.trace import Tier, TraceSet
        reqs = []
        for i in range(60):
            model = "claude-opus-5" if (i // 10) % 2 == 0 else "claude-sonnet-5"
            reqs.append(req(i, [], gap=120, model=model, writes=200_000,
                            session="s"))
        batch = {f.code for f in analyze(
            TraceSet(requests=reqs, tier=Tier.USAGE_ONLY, source="x"),
            allow_unreconciled=True).findings}
        m, fired = Monitor(), []
        for r in reqs:
            fired += m.observe(r)
        runtime = {a.code for a in fired}
        self.assertIn("REB-1", batch)
        self.assertIn("RT-REBUILD", runtime,
                      "the report says rebuild, the monitor must not be silent")


class TestEveryAlertCanBeRendered(unittest.TestCase):
    """An alert that crashes when printed is worse than no alert.

    `Alert.__str__` unpacked `self.scope[:3]` into three names, while rebuild
    tracking deliberately keys on `(tenant, target_id)` -- the model is dropped
    because a rebuild is a property of the prefix, not the pool. So the alerts
    that report a live cache failure raised ValueError the moment anything
    logged or printed one, which is the only thing anybody does with an alert.

    Swept over every scope width the emitters are allowed to produce rather
    than fixed for the two-wide case, because the widths are a design choice
    that can change again.
    """

    def test_scopes_of_every_width_render(self):
        for scope in ((), ("acme",), ("acme", "amazon-bedrock/converse"),
                      ("acme", "anthropic/direct", "claude-opus-5"),
                      (None, "anthropic/direct", None)):
            with self.subTest(scope=scope):
                a = monitor.Alert(code="RT-REBUILD", severity="high", scope=scope,
                                  summary="prefix rebuilt", detail="detail")
                self.assertIn("RT-REBUILD", str(a))

    def test_a_rebuild_alert_from_the_monitor_itself_renders(self):
        """Not a hand-built Alert: the scope has to come from the emitter, or
        the test proves only that the dataclass works."""
        m, fired = Monitor(), []
        for i in range(40):
            fired += m.observe(Request(
                request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                model="claude-opus-5", tenant="acme",
                usage={"input_tokens": 100, "cache_read_input_tokens": 0,
                       "cache_creation_input_tokens": 6000},
                segments=[Segment(id=f"s{i}", role="system", tokens=6000, index=0,
                                  cache_marked=True, ttl="5m")], session="s", target_id="anthropic/direct"))
        for a in fired:
            with self.subTest(code=a.code):
                self.assertIsInstance(str(a), str)


class TestARegistryLookupItCannotMakeIsSaidAloud(unittest.TestCase):
    """A check that abstains because the registry could not answer must say so.

    RT-NOSURFACE existed for this, and probed exactly one key --
    `min_cacheable_tokens`. Measured before the fix, over 30 requests on a
    surface where one lookup at a time was made to fail:

        min_cacheable_tokens unavailable      -> RT-NOSURFACE, 1 alert
        capability('max_breakpoints') missing -> 0 alerts
        capability('lookback_blocks') missing -> 0 alerts

    A surface can answer the minimum and not the budget, so the one probe did
    not imply the other two. `lookback_blocks` was in no review; it was found
    by asking the code which keys it reads. So the tests below discover the
    dependency set the same way -- by watching the reads -- rather than naming
    the three that are known today.
    """

    def _spy(self, on_capability, on_minimum):
        """Swap both registry entry points, restoring them in a `finally`.

        A leaked monkeypatch corrupts every test that runs afterwards and the
        failure surfaces somewhere unrelated, so nothing here patches without
        this wrapper.
        """
        import contextlib

        @contextlib.contextmanager
        def patched():
            real_cap = registry.capability
            real_min = registry.min_cacheable_tokens
            registry.capability = on_capability(real_cap)
            registry.min_cacheable_tokens = on_minimum(real_min)
            try:
                yield
            finally:
                registry.capability = real_cap
                registry.min_cacheable_tokens = real_min
        return patched()

    def _stream(self, target="anthropic/direct", n=30, marked=True, ttl="5m"):
        m, fired = Monitor(), []
        for i in range(n):
            fired += m.observe(req(i, [sg(0, "system", 8000, "sys",
                                          "instructions", marked=marked,
                                          ttl=ttl if marked else None),
                                       sg(1, "user", 100, f"t{i}", "user_turn")],
                                   target=target, ttl=ttl, session="s"))
        return m, fired

    def _notices(self, **kw):
        return [a for a in self._stream(**kw)[1] if a.code == "RT-NOSURFACE"]

    def _dependencies(self):
        """Which registry keys the checks actually read, discovered by
        watching them read, so a key added later is covered here without
        anyone editing this file."""
        asked = set()

        def cap(real):
            def f(target_id, name, allow_contested=False):
                asked.add(("capability", name))
                return real(target_id, name, allow_contested)
            return f

        def mn(real):
            def f(target_id, model):
                asked.add(("min_cacheable_tokens", None))
                return real(target_id, model)
            return f

        with self._spy(cap, mn):
            self._stream()
        return sorted(asked, key=lambda d: (d[0], d[1] or ""))

    def _with_unavailable(self, kind, name, *, as_null=False):
        """One stream where exactly one lookup cannot be answered.

        `as_null` picks the second route into the same silence: the capability
        *is* recorded, as JSON null. `amazon-bedrock/converse` and
        `openai/direct` both ship that way, so this is not a hypothetical.
        """
        def cap(real):
            def f(target_id, cname, allow_contested=False):
                if kind == "capability" and cname == name:
                    if as_null:
                        return None
                    raise registry.RegistryError(f"test: {cname} unavailable")
                return real(target_id, cname, allow_contested)
            return f

        def mn(real):
            def f(target_id, model):
                if kind == "min_cacheable_tokens":
                    if as_null:
                        return None
                    raise registry.RegistryError("test: minimum unavailable")
                return real(target_id, model)
            return f

        with self._spy(cap, mn):
            return self._stream()[1]

    def test_there_are_dependencies_to_check(self):
        """Guard the guard: if the discovery finds nothing, every assertion
        below is vacuously true."""
        self.assertTrue(self._dependencies())

    def test_an_unanswerable_lookup_is_announced(self):
        silent = []
        for kind, name in self._dependencies():
            with self.subTest(dependency=f"{kind}:{name}"):
                fired = self._with_unavailable(kind, name)
                if "RT-NOSURFACE" not in {a.code for a in fired}:
                    silent.append(f"{kind}({name})")
        self.assertEqual([], silent,
                         "these disable or unbound a check with no alert: "
                         + ", ".join(silent))

    def test_a_lookup_answered_with_null_is_announced_too(self):
        """The second route to the same silence. `capability()` returning None
        never raised, so the `except RegistryError` guard never saw it and
        `if not budget: return` swallowed it."""
        silent = []
        for kind, name in self._dependencies():
            with self.subTest(dependency=f"{kind}:{name}"):
                fired = self._with_unavailable(kind, name, as_null=True)
                if "RT-NOSURFACE" not in {a.code for a in fired}:
                    silent.append(f"{kind}({name})")
        self.assertEqual([], silent,
                         "these are recorded as null and disable a check with "
                         "no alert: " + ", ".join(silent))

    def test_the_same_missing_key_is_not_deduped_across_different_checks(self):
        r = req(0, STABLE)
        blocked = monitor._RegistryReads()
        blocked.unanswered = {
            "min_cacheable_tokens": (
                {monitor._NOT_RECORDED}, {"RT-BLOCKED (inactive)"})}
        below_min = monitor._RegistryReads()
        below_min.unanswered = {
            "min_cacheable_tokens": (
                {monitor._NOT_RECORDED}, {"RT-MIN (inactive)"})}
        self.assertNotEqual(
            Monitor._nosurface(r, ("scope",), blocked).subject,
            Monitor._nosurface(r, ("scope",), below_min).subject,
            "a later check using the same missing registry key would be "
            "suppressed as a duplicate")

    def test_the_alert_names_the_key_and_what_it_costs(self):
        """"Some checks are inactive" is not actionable. The reader needs the
        key to record and the check they are currently not getting."""
        fired = self._with_unavailable("capability", "max_breakpoints")
        a = next(x for x in fired if x.code == "RT-NOSURFACE")
        self.assertIn("max_breakpoints", a.detail)
        self.assertIn("RT-BUDGET", a.detail)
        self.assertIn("unmeasured, not healthy", a.detail)
        self.assertNotIn("lookback_blocks", a.detail,
                         "only the lookup that failed may be named")

    def test_a_fully_recorded_surface_is_not_nagged(self):
        self.assertNotIn("RT-NOSURFACE",
                         {a.code for a in self._stream()[1]})

    def test_it_is_said_once_per_scope_not_once_per_request(self):
        fired = self._with_unavailable("capability", "max_breakpoints")
        self.assertEqual(1, [a.code for a in fired].count("RT-NOSURFACE"))

    def test_fixing_one_lookup_still_reports_the_others(self):
        """The dedup is keyed on the *set* of unanswered lookups, so a surface
        that gains one recorded key and still lacks another is reported again
        rather than swallowed as a repeat of an alert about a different gap."""
        state = {"deny": {"max_breakpoints", "lookback_blocks"}}

        def cap(real):
            def f(target_id, name, allow_contested=False):
                if name in state["deny"]:
                    raise registry.RegistryError(f"test: {name} unavailable")
                return real(target_id, name, allow_contested)
            return f

        fired = []
        with self._spy(cap, lambda real: real):
            m = Monitor()
            for i in range(20):
                if i == 10:
                    state["deny"] = {"lookback_blocks"}
                fired += m.observe(req(i, STABLE, session="s"))
        subjects = [a.subject for a in fired if a.code == "RT-NOSURFACE"]
        self.assertEqual(2, len(subjects), subjects)
        self.assertNotEqual(subjects[0], subjects[1])

    def test_the_suppression_table_stays_bounded(self):
        """RT-NOSURFACE's subject now varies, and the two hand-rolled dedup
        sites it and RT-BLIND used wrote to `firing` without the eviction the
        check loop does. Same table, same cap, one code path."""
        m, _ = self._stream()
        st = m._scopes[(None, "anthropic/direct", "claude-opus-5")]
        for i in range(monitor.MAX_FIRING * 4):
            m._fire(st, monitor.Alert("RT-SYNTHETIC", "low", ("x",), "s", "d",
                                      subject=f"n{i}"), [])
        self.assertLessEqual(len(st.firing), monitor.MAX_FIRING)

    def test_a_recorded_zero_is_an_answer_not_a_gap(self):
        """`deepseek/direct` records `max_breakpoints: 0` -- it allows no
        explicit breakpoints. That is a fact about the surface, so RT-BUDGET
        is correctly quiet and there is nothing to announce. Testing the value
        with `not budget` rather than `is None` would have reported it as
        missing registry data and sent somebody to fix a file that is right."""
        fired = self._stream(target="deepseek/direct")[1]
        a = next(x for x in fired if x.code == "RT-NOSURFACE")
        self.assertNotIn("max_breakpoints", a.detail)

    def test_a_shipped_surface_with_a_null_capability_says_so(self):
        """No monkeypatching at all: these are rows in the registry as it
        ships, and both reach the silent path this class is about."""
        for target, key in (("amazon-bedrock/converse", "lookback_blocks"),
                            ("openai/direct", "max_breakpoints")):
            with self.subTest(target=target):
                fired = self._stream(target=target)[1]
                a = next((x for x in fired if x.code == "RT-NOSURFACE"), None)
                self.assertIsNotNone(a, f"{target} abstains in silence")
                self.assertIn(key, a.detail)

    def test_the_notice_carries_no_dollar_figure(self):
        for target in ("deepseek/direct", "openai/direct"):
            for a in self._stream(target=target)[1]:
                with self.subTest(target=target, code=a.code):
                    self.assertNotIn("$", a.summary + a.detail + a.fix)


class TestTheNoticeNamesTheRealCause(unittest.TestCase):
    """An alert that fires on ordinary traffic gets switched off, and then it
    protects nothing -- the same end state as the silence it was added to fix.

    Three ways the seam announced a lookup that was not actually the reason
    the operator was getting no answer. All three measured on the registry as
    it ships, no monkeypatching.
    """

    _stream = TestARegistryLookupItCannotMakeIsSaidAloud._stream
    _notices = TestARegistryLookupItCannotMakeIsSaidAloud._notices

    def test_unmarked_traffic_is_not_told_the_budget_check_is_off(self):
        """`openai/direct` records `max_breakpoints: null`. A request carrying
        no markers is answered without it -- zero cannot exhaust any
        non-negative budget -- so RT-BUDGET is concluding, not abstaining.

        Measured before the fix: 1 RT-NOSURFACE on 20 ordinary unmarked
        requests, claiming RT-BUDGET was inactive."""
        self.assertEqual([], self._notices(target="openai/direct",
                                           marked=False, ttl="30m"))

    def test_the_marked_request_that_follows_is_still_reported(self):
        """The sharper half. The subject is keyed on the set of unanswered
        lookups, so the false notice on unmarked traffic burned the very slot
        the genuine one needed and the real abstention was then suppressed as
        a repeat."""
        m, fired = Monitor(), []
        for i in range(10):
            fired += m.observe(req(i, [sg(0, "system", 8000, "sys")],
                                   target="openai/direct", ttl="30m",
                                   session="s"))
        self.assertEqual([], [a for a in fired if a.code == "RT-NOSURFACE"])
        for i in range(10, 20):
            fired += m.observe(req(i, [sg(0, "system", 8000, "sys",
                                          marked=True, ttl="30m"),
                                       sg(1, "user", 100, f"t{i}")],
                                   target="openai/direct", ttl="30m",
                                   session="s"))
        notices = [a for a in fired if a.code == "RT-NOSURFACE"]
        self.assertEqual(1, len(notices), "the genuine abstention was swallowed")
        self.assertIn("max_breakpoints", notices[0].detail)

    def test_a_contested_row_is_reported_as_contested(self):
        """`ContestedRow` subclasses `RegistryError`, so one `except` clause
        reported a disputed row as missing data and sent the reader to add a
        value already on file. This repo's standing rule is that a contested
        row is never fact, so the dispute has to be named and the settle
        remedy has to appear.

        This test asserted the opposite exclusion in its first form -- that a
        contested row says *nothing* about absence -- which encoded the very
        collapse the next review found: contested and absent are independent,
        and `openai/bedrock` is both. Naming the dispute is the requirement;
        suppressing the other half never was."""
        for target in ("openai/bedrock", "google/gemini-explicit"):
            with self.subTest(target=target):
                a = next(iter(self._notices(target=target)), None)
                self.assertIsNotNone(a, f"{target} abstains in silence")
                self.assertIn("contested", a.detail)
                self.assertIn("settle the contested row", a.fix.lower())

    def test_a_genuinely_absent_row_still_says_so(self):
        """The other direction, so the fix above cannot be "call everything
        contested"."""
        a = next(iter(self._notices(target=registry.UNATTRIBUTED)))
        self.assertIn("not recorded for this surface", a.detail)
        self.assertNotIn("contested", a.detail)
        self.assertIn("record the missing limits", a.fix.lower())

    def test_a_lifetime_rt_ttl_cannot_evaluate_does_not_blame_the_lookback(self):
        """`openai/direct` advertises 30m; RT-TTL evaluates 5m and 1h. It
        abstains on the lifetime before the lookback bound can matter, so
        filling that registry gap would change nothing and naming it names a
        cause that is not the real one."""
        self.assertNotIn("30m", monitor.TTL_SECONDS)
        for a in self._notices(target="openai/direct", ttl="30m"):
            self.assertNotIn("lookback_blocks", a.detail)
            self.assertNotIn("RT-TTL", a.detail)

    def test_a_lifetime_rt_ttl_can_evaluate_still_blames_the_lookback(self):
        """The gate must not become a blanket silence: on a lifetime RT-TTL
        does act on, the unbounded timeline is the real consequence and has to
        be said. `amazon-bedrock/converse` ships `lookback_blocks: null`.

        The lifetimes swept here come from the registry rather than being
        written out, because they are per model on this surface: 1h went GA on
        Bedrock for three models and `claude-opus-5` is not one of them, so
        hard-coding "5m and 1h" would assert the check speaks about a lifetime
        the provider would reject."""
        offered = registry.supported_ttls("amazon-bedrock/converse",
                                          "claude-opus-5")
        self.assertTrue(offered)
        for ttl in offered:
            with self.subTest(ttl=ttl):
                a = next(iter(self._notices(
                    target="amazon-bedrock/converse", ttl=ttl)), None)
                self.assertIsNotNone(a, f"{ttl} abstains in silence")
                self.assertIn("lookback_blocks", a.detail)
                self.assertIn("unbounded", a.detail)

    def test_a_lifetime_the_model_cannot_use_is_not_argued_about(self):
        """The per-model narrowing, end to end. `claude-opus-5` on Bedrock is
        offered 5m only, so a 1h request there must not produce cadence advice
        even though the surface as a whole lists 1h."""
        self.assertNotIn("1h", registry.supported_ttls(
            "amazon-bedrock/converse", "claude-opus-5"))
        self.assertIn("1h", registry.supported_ttls(
            "amazon-bedrock/converse", "claude-haiku-4-5"))
        m = Monitor()
        r = req(0, [sg(0, "system", 8000, "sys", marked=True, ttl="1h")],
                ttl="1h", target="amazon-bedrock/converse")
        self.assertIsNone(
            m._ttl_rt_ttl_can_read(r, monitor._RegistryReads())[0])

    def test_the_gate_and_the_check_share_one_definition(self):
        """The announcement is gated on whether RT-TTL can read the lifetime.
        Two copies of that condition would drift, and a drifted copy is how
        the alert starts describing a check that no longer behaves that way."""
        m = Monitor()
        for ttl, expected in (("5m", "5m"), ("1h", "1h"), ("30m", None),
                              (None, None)):
            with self.subTest(ttl=ttl):
                r = req(0, [sg(0, "system", 8000, "sys", marked=True, ttl=ttl)],
                        ttl=ttl)
                ttl, offered = m._ttl_rt_ttl_can_read(
                    r, monitor._RegistryReads())
                self.assertEqual(expected, ttl)

    def test_a_contested_row_that_also_lacks_the_key_says_both(self):
        """Contested and absent are flags, not alternatives. Shipped
        `openai/bedrock` is flagged contested AND carries
        `capabilities: {"_unknown": true}`, so the keys are missing too.

        Catching ContestedRow first fixed the label and introduced a second
        collapse: every key on a contested row was reported present-but-
        disputed without checking, and only the settle remedy was emitted. An
        operator was told to settle a row that also needs values recorded."""
        a = next(iter(self._notices(target="openai/bedrock")))
        self.assertIn("contested", a.detail)
        self.assertIn("not recorded for this surface", a.detail)
        self.assertIn("settle the contested row", a.fix.lower())
        self.assertIn("record the missing limits", a.fix.lower())

    def test_a_contested_row_that_does_record_the_key_says_only_that(self):
        """The other direction, inspected rather than assumed. Nothing may
        report a key as absent without having looked."""
        from cacheeconomics import registry as reg
        real = reg.capability
        real_min = reg.min_cacheable_tokens

        def cap(target_id, name, allow_contested=False):
            if not allow_contested:
                raise reg.ContestedRow("test: contested")
            # On file, and inspectable without publishing. Real shapes per
            # key, so the narrowing the registry does on the way back still
            # gets something it can work with.
            return real(target_id, name, allow_contested=True)

        def mn(target_id, model, allow_contested=False):
            if not allow_contested:
                raise reg.ContestedRow("test: contested")
            return real_min(target_id, model, allow_contested=True)

        reg.capability, reg.min_cacheable_tokens = cap, mn
        try:
            a = next(iter(self._notices(target="anthropic/direct")))
        finally:
            reg.capability, reg.min_cacheable_tokens = real, real_min
        self.assertIn("contested", a.detail)
        self.assertNotIn("not recorded for this surface", a.detail)
        self.assertNotIn("record the missing limits", a.fix.lower())

    def test_the_minimum_on_a_contested_row_is_now_inspected(self):
        """`min_cacheable_tokens` used to take no `allow_contested`, so its
        presence on a contested row was reported as "not inspected" -- true,
        but it left an operator able to settle the contest and find the same
        checks still off for an unreported second reason. `openai/bedrock`
        records no minimum and no `inherits_minimums_from`, and now says so.

        The registry grew the parameter rather than the monitor growing a copy
        of the inheritance walk."""
        a = next(iter(self._notices(target="openai/bedrock")))
        part = next(p for p in a.detail.split("; ")
                    if p.startswith("min_cacheable_tokens ("))
        self.assertIn("contested", part)
        self.assertIn("not recorded for this surface", part)
        self.assertNotIn("was not inspected", part)

    def test_a_lookup_with_no_inspecting_form_still_says_it_did_not_look(self):
        """Every read site now passes an inspecting form, so this is about the
        seam's default rather than a live surface: a site added later without
        one must report that it did not look, not guess in either direction.
        Guessing "present" understates a row that needs values recorded;
        guessing "absent" invents a gap."""
        reads = monitor._RegistryReads()

        def raises():
            raise registry.ContestedRow("test: contested")

        reads.get("some_future_key", raises, needed_for="RT-FUTURE")
        causes, _ = reads.unanswered["some_future_key"]
        self.assertIn(monitor._UNINSPECTED, causes)
        self.assertNotIn(monitor._NOT_RECORDED, causes)

    def test_two_lifetimes_in_one_request_are_still_ambiguous(self):
        """`_cadence_vs_ttl` abstains when a request carries a durable prefix
        under an advancing turn. The shared helper has to keep that, or the
        refactor quietly turned an abstention into a guess."""
        m = Monitor()
        r = req(0, [sg(0, "system", 8000, "sys", marked=True, ttl="1h"),
                    sg(1, "user", 4000, "turn", marked=True, ttl="5m")],
                ttl="5m")
        self.assertEqual(2, len(r.marker_lifetimes))
        self.assertIsNone(
            m._ttl_rt_ttl_can_read(r, monitor._RegistryReads())[0])


if __name__ == "__main__":
    unittest.main()
