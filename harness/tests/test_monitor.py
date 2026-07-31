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

from cacheeconomics import monitor  # noqa: E402
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
                                  cache_marked=True, ttl="5m")], session="s"))
        for a in fired:
            with self.subTest(code=a.code):
                self.assertIsInstance(str(a), str)


if __name__ == "__main__":
    unittest.main()
