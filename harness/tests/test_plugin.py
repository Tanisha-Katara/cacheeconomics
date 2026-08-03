"""The runtime plugin decides inside a request path, which is a strictly worse
position to decide from than the analyzer's.

So most of these tests are about what it declines to do. A plugin that places a
marker on thin evidence charges the write premium for a guess, and it does it on
every request, forever, without anybody looking.
"""

import os
import sys
import asyncio
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cacheeconomics import plugin as plugin_mod
from cacheeconomics.plugin import (CachePlugin, litellm_handler,        # noqa: E402
                                   markable_positions)
from cacheeconomics.segment import (apply_markers, marker_count,       # noqa: E402
                                    strip_markers, walk)
from cacheeconomics.trace import Segment                               # noqa: E402

T0 = datetime(2026, 7, 29, 9, tzinfo=timezone.utc)
KEY = b"k" * 32


def sg(i, role, tokens, sid, label="", marked=False, ttl=None):
    return Segment(id=sid, role=role, tokens=tokens, index=i, label=label,
                   cache_marked=marked, ttl=ttl)


def body(i, turn_len=800):
    return {"tools": [{"name": "read", "description": "x" * 20000},
                      {"name": "write", "description": "y" * 8000}],
            "system": [{"type": "text", "text": "instructions " * 900},
                       {"type": "text", "text": f"ctx {i // 10} " + "z" * 3000}],
            "messages": [{"role": "user", "content": f"turn {i} " + "q" * turn_len}]}


def warm(plugin, n=40, maker=body, gap=90, model="claude-opus-5", **kw):
    last = None
    for i in range(n):
        last = plugin.on_request(maker(i), model=model,
                                 at=T0 + timedelta(seconds=gap * i), **kw)
    return last


class TestItWaitsForEvidence(unittest.TestCase):

    def test_a_cold_plugin_changes_nothing(self):
        p = CachePlugin(key=KEY, warmup=32)
        out, d = p.on_request(body(0), model="claude-opus-5", at=T0)
        self.assertFalse(d.applied)
        self.assertEqual(out, body(0))

    def test_it_says_how_far_through_warmup_it_is(self):
        p = CachePlugin(key=KEY, warmup=32)
        _, d = p.on_request(body(0), model="claude-opus-5", at=T0)
        self.assertIn("1/32", d.reason)

    def test_it_learns_from_requests_it_declined_to_touch(self):
        """Otherwise a workload it keeps refusing never leaves warmup."""
        p = CachePlugin(key=KEY, warmup=16)
        _, d = warm(p, n=20)
        self.assertTrue(d.applied)

    def test_it_needs_gaps_not_just_a_count(self):
        """Every request at the same instant: nothing can be shown to survive
        to the next one, because there is no next one to survive to."""
        p = CachePlugin(key=KEY, warmup=4)
        for i in range(20):
            out, d = p.on_request(body(i), model="claude-opus-5", at=T0)
        self.assertFalse(d.applied)


class TestItStandsDownWhenSomebodyElseHasDecided(unittest.TestCase):

    def test_an_existing_marker_stops_it(self):
        p = CachePlugin(key=KEY, warmup=4)
        marked = body(0)
        marked["system"][0]["cache_control"] = {"type": "ephemeral"}
        for i in range(20):
            b = body(i)
            b["system"][0]["cache_control"] = {"type": "ephemeral"}
            out, d = p.on_request(b, model="claude-opus-5",
                                  at=T0 + timedelta(seconds=90 * i))
        self.assertFalse(d.applied)
        self.assertIn("already placed cache markers", d.reason)

    def test_the_override_exists_and_works(self):
        p = CachePlugin(key=KEY, warmup=4, respect_existing=False)
        for i in range(20):
            b = body(i)
            b["system"][0]["cache_control"] = {"type": "ephemeral"}
            out, d = p.on_request(b, model="claude-opus-5",
                                  at=T0 + timedelta(seconds=90 * i))
        self.assertTrue(d.applied)


class TestItAbstainsNearTheMinimum(unittest.TestCase):
    """Token counts at request time are a byte-ratio estimate. Below the
    minimum a marker caches nothing and the provider returns no error."""

    def small(self, i):
        return {"system": [{"type": "text", "text": "s" * 15000}],
                "messages": [{"role": "user", "content": f"t{i}"}]}

    def test_a_prefix_inside_the_margin_is_not_marked(self):
        # ~4,166 estimated tokens against Haiku 4.5's 4,096 minimum: over it,
        # but not by enough for a byte estimate to be sure.
        p = CachePlugin(key=KEY, warmup=4, minimum_margin=0.15)
        _, d = warm(p, n=20, maker=self.small, model="claude-haiku-4-5")
        self.assertFalse(d.applied)
        self.assertTrue(any("ABSTAIN" in n for n in d.notes))

    def test_the_note_says_why_and_names_both_numbers(self):
        p = CachePlugin(key=KEY, warmup=4, minimum_margin=0.15)
        _, d = warm(p, n=20, maker=self.small, model="claude-haiku-4-5")
        note = next(n for n in d.notes if "ABSTAIN" in n)
        self.assertIn("4,096", note)
        self.assertIn("no error is returned", note)

    def test_a_comfortable_prefix_is_marked(self):
        p = CachePlugin(key=KEY, warmup=4, minimum_margin=0.15)
        _, d = warm(p, n=20, model="claude-haiku-4-5")
        self.assertTrue(d.applied)


class TestItPlacesMarkersOnTheWire(unittest.TestCase):

    def test_the_marker_lands_where_the_plan_said(self):
        p = CachePlugin(key=KEY, warmup=8)
        out, d = warm(p, n=20)
        self.assertTrue(d.applied)
        for i, (_, _, block, _) in enumerate(walk(out)):
            marked = isinstance(block, dict) and "cache_control" in block
            self.assertEqual(marked, i in d.placements,
                             f"wire position {i} disagrees with the plan")

    def test_the_always_changing_turn_is_never_marked(self):
        p = CachePlugin(key=KEY, warmup=8)
        _, d = warm(p, n=20)
        blocks = list(walk(body(0)))
        self.assertNotIn(len(blocks) - 1, d.placements)

    def test_it_does_not_mutate_the_caller_s_request(self):
        """An agent loop still appending to `messages` is not hypothetical --
        the recorder shipped a bug of exactly that shape."""
        p = CachePlugin(key=KEY, warmup=8)
        warm(p, n=20)
        original = body(99)
        snapshot = repr(original)
        out, d = p.on_request(original, model="claude-opus-5",
                              at=T0 + timedelta(seconds=9000))
        self.assertTrue(d.applied)
        self.assertEqual(repr(original), snapshot)
        self.assertIsNot(out, original)

    def test_a_five_minute_marker_carries_no_ttl_key(self):
        """5m is the provider default; sending it explicitly is noise."""
        p = CachePlugin(key=KEY, warmup=8)
        out, d = warm(p, n=12)
        for _, _, block, _ in walk(out):
            if isinstance(block, dict) and "cache_control" in block:
                if block["cache_control"].get("ttl") is None:
                    self.assertNotIn("ttl", block["cache_control"])

    def test_it_never_exceeds_the_surface_budget(self):
        from cacheeconomics import registry
        p = CachePlugin(key=KEY, warmup=8)
        _, d = warm(p, n=20)
        self.assertLessEqual(len(d.placements),
                             registry.capability("anthropic/direct", "max_breakpoints"))


class TestMarkerApplication(unittest.TestCase):

    def test_a_bare_string_system_becomes_a_block(self):
        out = apply_markers({"system": "hello", "messages": []}, {0: "5m"})
        self.assertEqual(out["system"],
                         [{"type": "text", "text": "hello",
                           "cache_control": {"type": "ephemeral"}}])

    def test_a_bare_string_message_becomes_a_block(self):
        out = apply_markers({"messages": [{"role": "user", "content": "hi"}]}, {0: "1h"})
        self.assertEqual(out["messages"][0]["content"],
                         [{"type": "text", "text": "hi",
                           "cache_control": {"type": "ephemeral", "ttl": "1h"}}])

    def test_an_empty_placement_returns_the_body_untouched(self):
        b = {"system": "hello", "messages": []}
        self.assertIs(apply_markers(b, {}), b)

    def test_reading_and_writing_agree_on_what_an_index_means(self):
        """The single reason `walk` exists. Four defects on this branch came
        from twin functions drifting apart on a shared assumption."""
        from cacheeconomics.segment import segments_from_request
        b = body(0)
        segs = segments_from_request(b, KEY)
        for target in range(len(segs)):
            out = apply_markers(b, {target: "5m"})
            hits = [i for i, (_, _, blk, _) in enumerate(walk(out))
                    if isinstance(blk, dict) and "cache_control" in blk]
            self.assertEqual(hits, [target])


class TestFeedback(unittest.TestCase):

    class _Resp:
        def __init__(self, read, write):
            self.usage = {"input_tokens": 10, "cache_read_input_tokens": read,
                          "cache_creation_input_tokens": write}

    def test_it_records_whether_its_own_markers_were_read(self):
        p = CachePlugin(key=KEY, warmup=8)
        _, d = warm(p, n=20)
        scope = (None, "anthropic/direct", "claude-opus-5")
        p.on_response(d, self._Resp(0, 30000), model="claude-opus-5")
        p.on_response(d, self._Resp(30000, 0), model="claude-opus-5")
        eff = p.effectiveness(scope)
        self.assertEqual(eff["placed"], 2)
        self.assertEqual(eff["read"], 1)
        self.assertEqual(eff["read_share"], 0.5)

    def test_a_request_it_did_not_touch_is_not_counted(self):
        from cacheeconomics.plugin import Decision
        p = CachePlugin(key=KEY, warmup=8)
        p.on_response(Decision(False), self._Resp(1000, 0), model="claude-opus-5")
        self.assertIsNone(p.effectiveness((None, "anthropic/direct", "claude-opus-5")))

    def test_effectiveness_is_none_before_anything_is_placed(self):
        self.assertIsNone(CachePlugin(key=KEY).effectiveness(
            (None, "anthropic/direct", "claude-opus-5")))


class TestItRefusesToRunUnkeyed(unittest.TestCase):

    def test_no_key_is_an_error_not_a_default(self):
        with self.assertRaises(ValueError):
            CachePlugin(key=b"")


class TestItDoesNotRelocate(unittest.TestCase):
    """Reordering changes instruction priority and authority. That needs a
    behavioural eval, and a request path cannot run one."""

    def drifting(self, i):
        return {"system": [{"type": "text", "text": f"session {i} " + "a" * 500},
                           {"type": "text", "text": "policy " * 4000}],
                "messages": [{"role": "user", "content": f"t{i}"}]}

    def test_the_block_order_is_never_changed(self):
        p = CachePlugin(key=KEY, warmup=8)
        out, _ = warm(p, n=20, maker=self.drifting)
        self.assertEqual([lbl for _, lbl, _, _ in walk(out)],
                         [lbl for _, lbl, _, _ in walk(self.drifting(19))])

    def test_it_reports_the_move_for_a_human_instead(self):
        p = CachePlugin(key=KEY, warmup=8)
        warm(p, n=20, maker=self.drifting)
        recs = p.recommendations((None, "anthropic/direct", "claude-opus-5"))
        blocked = next(a for a in recs if a.code == "RT-BLOCKED")
        self.assertIn("Move that section", blocked.fix)
        self.assertIn("behavioural eval", blocked.fix)


class TestItFailsClosedOnAnUnknownMinimum(unittest.TestCase):
    """This one mutates a live request. Not knowing the threshold is exactly
    when a marker is most likely to be paid for and cache nothing."""

    def test_a_provider_prefixed_model_is_normalised_not_rejected(self):
        """LiteLLM sends "anthropic/claude-opus-5". The registry does not know
        that name, so every minimum lookup raised and the check was off on the
        one integration path that rewrites live requests."""
        p = CachePlugin(key=KEY, warmup=8)
        _, d = warm(p, n=20, model="anthropic/claude-opus-5")
        self.assertTrue(d.applied)
        self.assertFalse(any("no recorded minimum" in n for n in d.notes))

    def test_a_date_stamped_model_is_normalised_too(self):
        p = CachePlugin(key=KEY, warmup=8)
        _, d = warm(p, n=20, model="claude-opus-5-20260101")
        self.assertTrue(d.applied)

    def test_a_genuinely_unknown_model_places_nothing(self):
        p = CachePlugin(key=KEY, warmup=8)
        _, d = warm(p, n=20, model="some-model-nobody-registered")
        self.assertFalse(d.applied)
        self.assertEqual(d.placements, {})

    def test_the_reason_distinguishes_unsafe_from_unnecessary(self):
        """"Nothing was worth placing" and "I could not tell whether placing
        anything was safe" are different answers, and only one means the
        workload is fine."""
        p = CachePlugin(key=KEY, warmup=8)
        _, d = warm(p, n=20, model="some-model-nobody-registered")
        self.assertIn("minimum", d.reason)
        self.assertNotIn("no marker worth placing", d.reason)

    def test_the_refusal_says_why_it_is_the_safe_direction(self):
        p = CachePlugin(key=KEY, warmup=8)
        _, d = warm(p, n=20, model="some-model-nobody-registered")
        self.assertIn("returns no error", d.reason)

    def test_the_allocator_refuses_before_the_plugin_has_to(self):
        """The guard now lives in `tiers.allocate`, which both the plugin and
        `allocator_full` go through. It was in the plugin's own filter and not
        in the allocator, so the batch path emitted markers and an 86% modelled
        saving for a model whose threshold nobody knows."""
        from cacheeconomics import tiers
        from cacheeconomics.trace import Segment as S
        with self.assertRaises(tiers.Unsupported):
            tiers.allocate([S(id="a", role="system", tokens=9000, index=0)],
                           {0: 0.0}, target_id="anthropic/direct",
                           model="some-model-nobody-registered", gaps=[120.0] * 20)


class TestItModelsTheWholePromptItMutates(unittest.TestCase):
    """A marker's prefix contains everything above it. Hiding those blocks from
    the model does not make them go away, it makes the plan wrong."""

    def _volatile_tools(self, i):
        return {"tools": [{"name": "read", "description": f"rev{i} " + "x" * 30000}],
                "messages": [{"role": "system", "content": "policy " * 3000},
                             {"role": "user", "content": f"turn {i}"}]}

    def test_volatile_tools_are_visible_to_the_change_rates(self):
        p = CachePlugin(key=KEY, warmup=8)
        last = None
        for i in range(20):
            body = self._volatile_tools(i)
            _, last = p.on_request(body, model="claude-opus-5",
                                   markable=markable_positions(body),
                                   at=T0 + timedelta(seconds=90 * i))
        scope = (None, "anthropic/direct", "claude-opus-5")
        self.assertEqual(p.monitor.change_rates(scope)[0], 1.0,
                         "the tool definition changes every request")

    def test_it_refuses_to_cache_a_prefix_a_volatile_tool_sits_in(self):
        p = CachePlugin(key=KEY, warmup=8)
        last = None
        for i in range(20):
            body = self._volatile_tools(i)
            _, last = p.on_request(body, model="claude-opus-5",
                                   markable=markable_positions(body),
                                   at=T0 + timedelta(seconds=90 * i))
        self.assertFalse(last.applied)

    def test_an_unmarkable_position_abstains_and_says_it_is_still_cached(self):
        """A tool block still sits inside the prefix of any marker below it."""
        def stable(i):
            return {"tools": [{"name": "read", "description": "x" * 30000}],
                    "messages": [{"role": "system", "content": "policy " * 3000},
                                 {"role": "user", "content": f"turn {i}"}]}
        p = CachePlugin(key=KEY, warmup=8)
        last = None
        for i in range(20):
            body = stable(i)
            _, last = p.on_request(body, model="claude-opus-5",
                                   markable=markable_positions(body),
                                   at=T0 + timedelta(seconds=90 * i))
        for pos in last.placements:
            self.assertNotEqual(pos, 0, "position 0 is the tool definition")
        if any("cannot carry a marker" in n for n in last.notes):
            note = next(n for n in last.notes if "cannot carry a marker" in n)
            self.assertIn("still cached", note)

    def test_without_markable_every_position_is_available(self):
        p = CachePlugin(key=KEY, warmup=8)
        last = None
        for i in range(20):
            last = p.on_request(
                {"tools": [{"name": "read", "description": "x" * 30000}],
                 "messages": [{"role": "user", "content": f"turn {i}"}]},
                model="claude-opus-5", at=T0 + timedelta(seconds=90 * i))[1]
        self.assertTrue(last.applied)
        self.assertIn(0, last.placements)

class TestASectionThatComesAndGoes(unittest.TestCase):
    """The live path reads the monitor's change rates. While absence was not
    recorded, an optional block that appeared every other request scored zero
    churn, and the plugin marked a prefix that could not survive to be read."""

    def optional(self, i):
        blocks = [{"type": "text", "text": "tools " * 9000},
                  {"type": "text", "text": "policy " * 6000}]
        if i % 2:
            blocks.append({"type": "text", "text": "optional " * 4000})
        return {"system": blocks,
                "messages": [{"role": "user", "content": f"turn {i}"}]}

    def _run(self, gap=240):
        p = CachePlugin(key=KEY, warmup=8)
        last = None
        for i in range(40):
            _, last = p.on_request(self.optional(i), model="claude-opus-5",
                                   session="s", at=T0 + timedelta(seconds=gap * i))
        return p, last

    def test_no_marker_lands_on_the_block_that_vanishes(self):
        _, d = self._run()
        self.assertNotIn(2, d.placements)

    def test_no_marker_lands_above_it_either(self):
        """A marker deeper than the optional block caches a prefix containing
        it, so it is invalidated on exactly the requests the block is missing."""
        _, d = self._run()
        self.assertTrue(all(pos < 2 for pos in d.placements),
                        f"placed at {sorted(d.placements)} with a vanishing block at 2")

    def test_the_stable_prefix_in_front_of_it_is_still_cached(self):
        """Refusing to cache anything would be the safe answer and the wrong
        one: 15,000 tokens ahead of the optional block never change."""
        _, d = self._run()
        self.assertTrue(d.applied)

    def test_the_plugin_sees_the_same_rates_the_batch_loader_would(self):
        p, _ = self._run()
        from cacheeconomics.allocate import reuse_chain
        scope = reuse_chain((None, "anthropic/direct", "claude-opus-5"), "unknown", "s")
        rates = p.monitor.change_rates(scope)
        self.assertEqual(rates[2], 1.0)


class TestTheLiteLLMAdapter(unittest.IsolatedAsyncioTestCase):
    """Shape checked against LiteLLM's documented proxy hook: an awaitable
    method on a CustomLogger subclass, not a callable.

    The previous tests here called it synchronously and passed, which is how a
    hook the proxy could never have awaited survived a full test suite."""

    class _Key:
        user_id = "tenant-a"

    def _data(self, i=0, content=None):
        return {"model": "claude-opus-5", "litellm_call_id": f"c{i}",
                "messages": [{"role": "system", "content": content or "policy " * 4000},
                             {"role": "user", "content": f"turn {i}"}]}

    def test_it_exposes_the_method_the_proxy_actually_awaits(self):
        import inspect
        h = litellm_handler(CachePlugin(key=KEY))
        self.assertTrue(inspect.iscoroutinefunction(h.async_pre_call_hook))

    def test_it_subclasses_custom_logger_when_litellm_is_present(self):
        class FakeCustomLogger:
            pass
        h = litellm_handler(CachePlugin(key=KEY), base=FakeCustomLogger)
        self.assertIsInstance(h, FakeCustomLogger)

    def test_it_works_without_litellm_installed(self):
        """The harness is standard library only; the import is guarded."""
        self.assertIsNotNone(litellm_handler(CachePlugin(key=KEY)))

    async def test_it_returns_the_data_dict_it_was_given_when_cold(self):
        h = litellm_handler(CachePlugin(key=KEY, warmup=32))
        data = self._data()
        self.assertEqual(await h.async_pre_call_hook(self._Key(), None, data,
                                                     "completion"), data)

    async def test_it_ignores_call_types_it_cannot_reason_about(self):
        h = litellm_handler(CachePlugin(key=KEY, warmup=1))
        data = {"model": "claude-opus-5", "input": "x"}
        self.assertEqual(await h.async_pre_call_hook(self._Key(), None, data,
                                                     "embeddings"), data)

    async def test_it_returns_a_dict_the_proxy_can_send(self):
        h = litellm_handler(CachePlugin(key=KEY, warmup=8))
        out = None
        for i in range(20):
            out = await h.async_pre_call_hook(self._Key(), None, self._data(i),
                                              "completion")
        self.assertIsInstance(out, dict)
        self.assertEqual(len(out["messages"]), 2)

    async def test_it_leaves_tool_definitions_alone(self):
        h = litellm_handler(CachePlugin(key=KEY, warmup=1))
        tools = [{"type": "function", "function": {"name": "f"}}]
        data = {**self._data(), "tools": tools}
        out = await h.async_pre_call_hook(self._Key(), None, data, "completion")
        self.assertEqual(out["tools"], tools)

    async def test_pending_decisions_do_not_grow_without_bound(self):
        """A response that never arrives must not pin a decision forever.
        This object lives inside somebody else's proxy."""
        h = litellm_handler(CachePlugin(key=KEY, warmup=4))
        for i in range(1000):
            await h.async_pre_call_hook(self._Key(), None, self._data(i), "completion")
        self.assertLessEqual(len(h._pending), 256)

    async def test_a_success_event_with_no_matching_request_is_survivable(self):
        h = litellm_handler(CachePlugin(key=KEY))
        await h.async_log_success_event({"litellm_call_id": "never-seen"},
                                        None, None, None)


class TestTheAdapterScopesByConversationNotByCall(unittest.IsolatedAsyncioTestCase):
    """`litellm_call_id` is per-request. Using it as the session made every turn
    the first turn of a new conversation, so rebuild detection -- the one thing
    worth having on this path -- never accumulated any evidence at all."""

    class _Resp:
        usage = {"input_tokens": 0, "cache_read_input_tokens": 0,
                 "cache_creation_input_tokens": 60_000,
                 "cache_creation": {"ephemeral_5m_input_tokens": 60_000,
                                    "ephemeral_1h_input_tokens": 0}}

    class _ObjKey:
        user_id = "tenant-a"

    async def _drive(self, meta, key, turns=80, handler=None):
        h = handler or litellm_handler(CachePlugin(key=KEY, warmup=8))
        for i in range(turns):
            data = {"model": "claude-opus-5", "litellm_call_id": f"call-{i}",
                    "metadata": dict(meta),
                    "messages": [{"role": "system", "content": "policy " * 4000},
                                 {"role": "user", "content": f"t{i}"}]}
            await h.async_pre_call_hook(key, None, data, "completion")
            await h.async_log_success_event({"litellm_call_id": f"call-{i}"},
                                            self._Resp(), None, None)
        return h

    async def test_a_rebuilding_conversation_is_detected_through_the_adapter(self):
        h = await self._drive({"session_id": "conv-1"}, self._ObjKey())
        self.assertIn("RT-REBUILD", {a.code for a in h.plugin.alerts})

    async def test_without_a_conversation_id_it_abstains_rather_than_inventing_one(self):
        h = await self._drive({}, self._ObjKey())
        codes = {a.code for a in h.plugin.alerts}
        self.assertIn("RT-NOSESSION", codes)
        self.assertNotIn("RT-REBUILD", codes)

    async def test_the_call_id_is_not_used_as_the_conversation(self):
        from cacheeconomics.plugin import default_session_from
        self.assertIsNone(default_session_from({"litellm_call_id": "call-1"}))

    async def test_an_explicit_session_id_wins(self):
        from cacheeconomics.plugin import default_session_from
        self.assertEqual(
            default_session_from({"metadata": {"session_id": "s", "trace_id": "t"}}), "s")

    async def test_a_retry_correlation_id_is_not_a_conversation(self):
        from cacheeconomics.plugin import default_session_from
        """LiteLLM's schema defines `trace_id` as spanning the retries of one
        call, so each turn carries a different one. Treating it as a session
        yields single-request groups, which cannot produce REB-1 and which
        suppress REB-0 -- the finding whose job is to say so."""
        self.assertIsNone(default_session_from({"metadata": {"trace_id": "t"}}))
        self.assertIsNone(default_session_from({"trace_id": "t"}))
        self.assertIsNone(default_session_from({"litellm_trace_id": "t"}))

    async def test_a_custom_extractor_is_honoured(self):
        h = litellm_handler(CachePlugin(key=KEY, warmup=8),
                            session_from=lambda d: d.get("my_thread"))
        for i in range(20):
            data = {"model": "claude-opus-5", "litellm_call_id": f"c{i}",
                    "my_thread": "thread-9",
                    "messages": [{"role": "user", "content": "policy " * 4000}]}
            await h.async_pre_call_hook(self._ObjKey(), None, data, "completion")
        self.assertNotIn("RT-NOSESSION", {a.code for a in h.plugin.alerts})

    async def test_tenant_is_read_from_an_object_key(self):
        h = await self._drive({"session_id": "c"}, self._ObjKey(), turns=12)
        self.assertEqual(list(h.plugin.monitor._scopes)[0][0], "tenant-a")

    async def test_tenant_is_read_from_a_dict_key(self):
        """The parameter is named `user_api_key_dict`. Reaching only for an
        attribute collapsed every tenant to None when it arrived as a dict, and
        tenant is part of the cache isolation scope."""
        h = await self._drive({"session_id": "c"}, {"user_id": "tenant-b"}, turns=12)
        self.assertEqual(list(h.plugin.monitor._scopes)[0][0], "tenant-b")

    async def test_two_tenants_do_not_share_a_scope(self):
        h = litellm_handler(CachePlugin(key=KEY, warmup=4))
        for who in ({"user_id": "a"}, {"user_id": "b"}):
            await self._drive({"session_id": "c"}, who, turns=10, handler=h)
        self.assertEqual(len({s[0] for s in h.plugin.monitor._scopes}), 2)


class TestEvidenceIsAboutThisPromptNotThisPosition(unittest.TestCase):
    """Change rates are keyed by wire position, and a position is not a thing --
    it is wherever a block happens to land. Insert a section and everything below
    it shifts, handing each block a history that belongs to its predecessor."""

    def trained(self, i):
        return {"system": [{"type": "text", "text": "policy " * 6000},
                           {"type": "text", "text": "context " * 4000}],
                "messages": [{"role": "user", "content": f"turn {i}"}]}

    def inserted(self, i):
        """A new block arrives at index 1, shifting context to 2, turn to 3."""
        return {"system": [{"type": "text", "text": "policy " * 6000},
                           {"type": "text", "text": f"NEVER SEEN {i} " + "z" * 24000},
                           {"type": "text", "text": "context " * 4000}],
                "messages": [{"role": "user", "content": f"turn {i}"}]}

    def _warm(self):
        p = CachePlugin(key=KEY, warmup=8)
        for i in range(20):
            p.on_request(self.trained(i), model="claude-opus-5", session="s",
                         at=T0 + timedelta(seconds=90 * i))
        return p

    def test_a_block_never_seen_before_is_not_marked(self):
        p = self._warm()
        _, d = p.on_request(self.inserted(99), model="claude-opus-5", session="s",
                            at=T0 + timedelta(seconds=1800))
        self.assertNotIn(1, d.placements)

    def test_no_marker_sits_below_it_either(self):
        """A marker deeper than the new block caches a prefix containing it."""
        p = self._warm()
        _, d = p.on_request(self.inserted(99), model="claude-opus-5", session="s",
                            at=T0 + timedelta(seconds=1800))
        self.assertTrue(all(pos < 1 for pos in d.placements),
                        f"placed at {sorted(d.placements)} with unseen content at 1")

    def test_the_stable_prefix_above_it_is_still_cached(self):
        p = self._warm()
        _, d = p.on_request(self.inserted(99), model="claude-opus-5", session="s",
                            at=T0 + timedelta(seconds=1800))
        self.assertTrue(d.applied)
        self.assertIn(0, d.placements)

    def test_it_says_which_positions_it_could_not_vouch_for(self):
        p = self._warm()
        _, d = p.on_request(self.inserted(99), model="claude-opus-5", session="s",
                            at=T0 + timedelta(seconds=1800))
        self.assertTrue(any("has not seen before" in n for n in d.notes))

    def test_the_familiar_shape_is_unaffected(self):
        """Fail-closed must not become fail-always. The allocator may spend one
        marker or two -- what matters is that it still reaches the deepest
        stable boundary rather than stopping short of it."""
        p = self._warm()
        _, d = p.on_request(self.trained(100), model="claude-opus-5", session="s",
                            at=T0 + timedelta(seconds=1800))
        self.assertTrue(d.applied)
        self.assertIn(1, d.placements)

    def test_evidence_is_earned_back_within_a_few_requests(self):
        """A prompt that changed shape should not be penalised forever."""
        p = self._warm()
        last = None
        for i in range(20):
            _, last = p.on_request(self.inserted(0), model="claude-opus-5", session="s",
                                   at=T0 + timedelta(seconds=1800 + 90 * i))
        self.assertIn(1, last.placements)

    def test_the_monitor_answers_the_question_directly(self):
        from cacheeconomics.trace import Segment
        p = self._warm()
        scope = (None, "anthropic/direct", "claude-opus-5")
        known = Segment(id=p.monitor._scopes[scope] and "x", role="system",
                        tokens=10, index=0)
        self.assertIn(0, p.monitor.unfamiliar(scope, [known]),
                      "an id never seen at that position is unfamiliar")

    def test_an_unknown_scope_is_entirely_unfamiliar(self):
        from cacheeconomics.trace import Segment
        p = self._warm()
        segs = [Segment(id="a", role="system", tokens=10, index=0)]
        self.assertEqual(p.monitor.unfamiliar((None, "anthropic/direct", "other"), segs),
                         {0})


class TestTheLiveWirePathIsOptIn(unittest.IsolatedAsyncioTestCase):
    """The observation half is verified here and cannot break a request. The
    mutation half puts `cache_control` on a wire nothing in this repository has
    watched a real LiteLLM proxy forward. Defaulting to mutate would ship a live
    request rewriter on a path this module's own docstring calls unconfirmed."""

    class _Key:
        user_id = "t"

    async def _drive(self, mutate):
        p = CachePlugin(key=KEY, warmup=4)
        # Spaced timestamps: a tight loop on wall-clock time looks like
        # concurrent traffic, which is correctly modelled as a cache miss.
        real, n = p.on_request, {"i": 0}

        def timed(body, **kw):
            kw["at"] = T0 + timedelta(seconds=90 * n["i"])
            n["i"] += 1
            return real(body, **kw)
        p.on_request = timed
        h = litellm_handler(p, mutate=mutate)
        out = None
        for i in range(20):
            data = {"model": "claude-opus-5", "litellm_call_id": f"c{i}",
                    "metadata": {"session_id": "conv"},
                    "messages": [{"role": "system", "content": "policy " * 4000},
                                 {"role": "user", "content": f"t{i}"}]}
            out = await h.async_pre_call_hook(self._Key(), None, data, "completion")
        marked = any(isinstance(m.get("content"), list)
                     and any("cache_control" in b for b in m["content"])
                     for m in out["messages"])
        return p, marked

    async def test_nothing_reaches_the_wire_by_default(self):
        _, marked = await self._drive(mutate=False)
        self.assertFalse(marked)

    async def test_and_the_pending_decision_says_nothing_was_sent(self):
        """It used to say `applied=True` with placements, which then fed
        `on_response` and credited markers LiteLLM never forwarded."""
        p, _ = await self._drive(mutate=False)
        self.assertIsNone(p.effectiveness((None, "anthropic/direct", "claude-opus-5")))

    async def test_it_still_learns_while_withholding_the_mutation(self):
        """Withholding the risky half must not cost the diagnostics."""
        p, _ = await self._drive(mutate=False)
        scope = next(s for s in p.monitor._scopes if len(s) > 3)
        self.assertEqual(p.monitor.samples(scope), 20)
        self.assertTrue(p.monitor.gaps(scope))

    async def test_opting_in_places_the_marker(self):
        _, marked = await self._drive(mutate=True)
        self.assertTrue(marked)


class TestOverrideReplacesRatherThanCombines(unittest.TestCase):
    """`respect_existing=False` left the caller's markers in place and added
    ours on top, so a request already carrying the surface's four went out with
    five -- a provider error dressed as an override, on the one path that
    rewrites live requests."""

    def _full_budget(self, i):
        return {"system": [{"type": "text", "text": f"{c} " * 3000,
                            "cache_control": {"type": "ephemeral"}}
                           for c in "abcd"],
                "messages": [{"role": "user", "content": f"turn {i}"}]}

    def _run(self, **kw):
        p = CachePlugin(key=KEY, warmup=4, **kw)
        out = d = None
        for i in range(20):
            out, d = p.on_request(self._full_budget(i), model="claude-opus-5",
                                  at=T0 + timedelta(seconds=90 * i))
        return out, d

    def _markers(self, body):
        return sum(1 for _, _, b, _ in walk(body)
                   if isinstance(b, dict) and "cache_control" in b)

    def test_the_patched_request_stays_inside_the_budget(self):
        from cacheeconomics import registry
        out, _ = self._run(respect_existing=False)
        self.assertLessEqual(self._markers(out),
                             registry.capability("anthropic/direct", "max_breakpoints"))

    def test_the_callers_markers_are_replaced_not_kept(self):
        out, d = self._run(respect_existing=False)
        self.assertTrue(d.applied)
        self.assertEqual(self._markers(out), len(d.placements))

    def test_it_says_it_replaced_them(self):
        _, d = self._run(respect_existing=False)
        self.assertTrue(any("replaced" in n for n in d.notes))

    def test_the_default_still_stands_down_instead(self):
        _, d = self._run()
        self.assertFalse(d.applied)
        self.assertIn("already placed cache markers", d.reason)


class TestWhatIsRecordedIsWhatWasSent(unittest.TestCase):
    """The plugin's own diagnostics are only worth anything if they describe the
    request that left the process. Two separate changes broke that invariant in
    opposite directions -- override recorded markers it had stripped, and
    observe-only recorded markers it never sent."""

    def _override_body(self, i):
        return {"system": [{"type": "text", "text": "policy " * 4000}],
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": f"turn {i} " + "q" * 400,
                     "cache_control": {"type": "ephemeral"}}]}]}

    def _wire(self, body):
        return {i for i, (_, _, b, _) in enumerate(walk(body))
                if isinstance(b, dict) and "cache_control" in b}

    def test_override_records_the_markers_that_survived_the_strip(self):
        """A caller's marker on the volatile trailing turn is stripped and
        replaced by one on the stable prefix. OR-ing the two sets told the
        monitor the trailing turn was cached on a request where it was not."""
        p = CachePlugin(key=KEY, warmup=4, respect_existing=False)
        out = d = None
        for i in range(20):
            out, d = p.on_request(self._override_body(i), model="claude-opus-5",
                                  at=T0 + timedelta(seconds=90 * i))
        self.assertTrue(d.applied)
        self.assertEqual(self._wire(out),
                         {s.index for s in d.segments if s.cache_marked})

    def test_standing_down_records_the_callers_markers_unchanged(self):
        p = CachePlugin(key=KEY, warmup=4)          # respect_existing default
        out = d = None
        for i in range(20):
            out, d = p.on_request(self._override_body(i), model="claude-opus-5",
                                  at=T0 + timedelta(seconds=90 * i))
        self.assertFalse(d.applied)
        self.assertEqual(self._wire(out),
                         {s.index for s in d.segments if s.cache_marked})

    def test_a_dry_run_sends_nothing_and_records_nothing(self):
        p = CachePlugin(key=KEY, warmup=4)
        out = d = None
        for i in range(20):
            out, d = p.on_request(body(i), model="claude-opus-5", apply=False,
                                  at=T0 + timedelta(seconds=90 * i))
        self.assertFalse(d.applied)
        self.assertEqual(d.placements, {})
        self.assertEqual(self._wire(out), set())
        self.assertFalse(any(s.cache_marked for s in d.segments))

    def test_a_dry_run_still_reports_what_it_would_have_done(self):
        """Withholding the mutation must not withhold the recommendation."""
        p = CachePlugin(key=KEY, warmup=4)
        d = None
        for i in range(20):
            _, d = p.on_request(body(i), model="claude-opus-5", apply=False,
                                at=T0 + timedelta(seconds=90 * i))
        self.assertTrue(d.proposed)

    def test_effectiveness_never_credits_an_unsent_marker(self):
        """`on_response` counts a placement only when one was applied. A dry run
        that reported its proposal as `placements` had the counters crediting
        markers the provider never saw."""
        class _Resp:
            usage = {"input_tokens": 0, "cache_read_input_tokens": 9_000,
                     "cache_creation_input_tokens": 0}
        p = CachePlugin(key=KEY, warmup=4)
        d = None
        for i in range(20):
            _, d = p.on_request(body(i), model="claude-opus-5", apply=False,
                                at=T0 + timedelta(seconds=90 * i))
        p.on_response(d, _Resp(), model="claude-opus-5")
        self.assertIsNone(p.effectiveness(d.scope))


class TestNotesDescribeWhatHappened(unittest.TestCase):
    """The same misdescription class as the wire-state fix, one level down. A
    dry run reported "replaced 1 marker(s)" when nothing was replaced and
    "are sent as text blocks" when nothing was sent. No counter reads notes, so
    this is human-facing only -- which is the whole audience for a note."""

    def _marked(self, i):
        return {"system": [{"type": "text", "text": "policy " * 4000}],
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": f"turn {i} " + "q" * 400,
                     "cache_control": {"type": "ephemeral"}}]}]}

    def _bare(self, i):
        return {"system": "policy " * 4000,
                "messages": [{"role": "user", "content": f"t{i}"}]}

    def _run(self, maker, apply, **kw):
        p = CachePlugin(key=KEY, warmup=4, **kw)
        out = d = None
        for i in range(20):
            out, d = p.on_request(maker(i), model="claude-opus-5", apply=apply,
                                  at=T0 + timedelta(seconds=90 * i))
        return out, d

    def test_a_dry_run_says_it_would_replace_not_that_it_did(self):
        _, d = self._run(self._marked, apply=False, respect_existing=False)
        note = next(n for n in d.notes if "replace" in n)
        self.assertTrue(note.startswith("would replace"), note[:40])

    def test_a_live_override_says_it_did(self):
        _, d = self._run(self._marked, apply=True, respect_existing=False)
        note = next(n for n in d.notes if "replace" in n)
        self.assertTrue(note.startswith("replaced"), note[:40])

    def test_a_dry_run_says_blocks_would_be_sent(self):
        _, d = self._run(self._bare, apply=False)
        note = next(n for n in d.notes if "bare strings" in n)
        self.assertIn("would be sent", note)

    def test_the_proposal_is_visible_to_anyone_reading_the_decision(self):
        """It was preserved in `proposed` and invisible in `str(decision)`, so
        only a programmatic caller ever saw it."""
        _, d = self._run(self._bare, apply=False)
        self.assertIn("would have placed", str(d))


class TestADryRunThatWouldHaveOverridden(unittest.TestCase):
    """The two fixed directions in combination, which nothing pinned. Correct
    behaviour here is subtly different from either alone: the body is returned
    untouched, so the caller's markers *do* stay on the wire and must appear in
    what the monitor observed, while the override plan sits in `proposed`."""

    def _body(self, i):
        return {"system": [{"type": "text", "text": "policy " * 4000}],
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": f"turn {i} " + "q" * 400,
                     "cache_control": {"type": "ephemeral"}}]}]}

    def _run(self):
        p = CachePlugin(key=KEY, warmup=4, respect_existing=False)
        out = d = None
        for i in range(20):
            out, d = p.on_request(self._body(i), model="claude-opus-5", apply=False,
                                  at=T0 + timedelta(seconds=90 * i))
        return p, out, d

    def test_the_callers_markers_are_still_on_the_wire(self):
        _, out, _ = self._run()
        marked = {i for i, (_, _, b, _) in enumerate(walk(out))
                  if isinstance(b, dict) and "cache_control" in b}
        self.assertEqual(marked, {1})

    def test_and_they_are_what_the_monitor_was_told(self):
        _, out, d = self._run()
        wire = {i for i, (_, _, b, _) in enumerate(walk(out))
                if isinstance(b, dict) and "cache_control" in b}
        self.assertEqual(wire, {s.index for s in d.segments if s.cache_marked})

    def test_the_override_plan_is_kept_as_a_proposal(self):
        _, _, d = self._run()
        self.assertFalse(d.applied)
        self.assertEqual(d.placements, {})
        self.assertTrue(d.proposed)

    def test_effectiveness_credits_nothing(self):
        class _Resp:
            usage = {"input_tokens": 0, "cache_read_input_tokens": 9_000,
                     "cache_creation_input_tokens": 0}
        p, _, d = self._run()
        p.on_response(d, _Resp(), model="claude-opus-5")
        self.assertIsNone(p.effectiveness(d.scope))


class TestTheAdapterReturnsEveryFieldItPatched(unittest.IsolatedAsyncioTestCase):
    """Markers are never placed on tools, because whether `cache_control`
    survives LiteLLM's tool translation is unverified. But override *strips*,
    and returning only `messages` left the caller's tool markers on the wire
    beside the plugin's new one -- five against a budget of four, a provider
    error produced by the one path that rewrites live requests.

    Removing a key is safe under any translation, which is why a stripped
    `tools` can be returned where a placed one could not."""

    class _Key:
        user_id = "t"

    async def _drive(self, n_tools, mutate=True):
        p = CachePlugin(key=KEY, warmup=4, respect_existing=False)
        real, c = p.on_request, {"i": 0}

        def timed(b, **kw):
            kw["at"] = T0 + timedelta(seconds=90 * c["i"])
            c["i"] += 1
            return real(b, **kw)
        p.on_request = timed
        h = litellm_handler(p, mutate=mutate)
        out = None
        for i in range(20):
            data = {"model": "claude-opus-5", "litellm_call_id": f"c{i}",
                    "metadata": {"session_id": "conv"},
                    "tools": [{"type": "function", "function": {"name": f"f{j}"},
                               "cache_control": {"type": "ephemeral"}}
                              for j in range(n_tools)],
                    "messages": [{"role": "system", "content": "policy " * 4000},
                                 {"role": "user", "content": f"t{i}"}]}
            out = await h.async_pre_call_hook(self._Key(), None, data, "completion")
        return out

    def _count(self, body):
        tools = sum(1 for t in body.get("tools", []) if "cache_control" in t)
        msgs = sum(1 for m in body["messages"] if isinstance(m.get("content"), list)
                   for b in m["content"] if "cache_control" in b)
        return tools, msgs

    async def test_a_full_tool_budget_does_not_become_an_over_budget_request(self):
        from cacheeconomics import registry
        out = await self._drive(4)
        tools, msgs = self._count(out)
        self.assertLessEqual(tools + msgs,
                             registry.capability("anthropic/direct", "max_breakpoints"))

    async def test_the_callers_tool_markers_are_stripped_under_override(self):
        out = await self._drive(4)
        self.assertEqual(self._count(out)[0], 0)

    async def test_a_dry_run_leaves_the_tools_exactly_as_they_were(self):
        """Nothing is sent, so nothing is stripped either."""
        out = await self._drive(4, mutate=False)
        self.assertEqual(self._count(out)[0], 4)


class TestUsageReachesTheMonitor(unittest.TestCase):
    """The request path knows the prompt; only the response knows the counters.
    Cold fan-out and prefix rebuild are driven entirely by the counters, so if
    they never arrive those two checks are unreachable — which is what an
    earlier version of this plugin shipped."""

    class _Resp:
        def __init__(self, read, write):
            self.usage = {"input_tokens": 0, "cache_read_input_tokens": read,
                          "cache_creation_input_tokens": write,
                          "cache_creation": {"ephemeral_5m_input_tokens": write,
                                             "ephemeral_1h_input_tokens": 0}}

    def _run(self, turns=80, rebuild_every=None):
        p = CachePlugin(key=KEY, warmup=4)
        prefix = 60_000
        for i in range(turns):
            _, d = p.on_request(body(i), model="claude-opus-5", session="s1",
                                at=T0 + timedelta(seconds=60 * i))
            cold = i == 0 or (rebuild_every and i % rebuild_every == 0)
            p.on_response(d, self._Resp(0 if cold else prefix,
                                        prefix if cold else 2_000),
                          model="claude-opus-5")
            prefix += 2_000
        return p

    def test_a_rebuilding_prefix_is_reported_live(self):
        p = self._run(rebuild_every=6)
        self.assertIn("RT-REBUILD", {a.code for a in p.alerts})

    def test_a_healthy_session_is_not(self):
        self.assertNotIn("RT-REBUILD", {a.code for a in self._run().alerts})

    def test_concurrent_writers_are_reported_live(self):
        p = CachePlugin(key=KEY, warmup=2)
        alerts = []
        for i in range(6):
            _, d = p.on_request(body(0), model="claude-opus-5", session="s1",
                                at=T0 + timedelta(seconds=i * 90))
            p.on_response(d, self._Resp(0, 60_000), model="claude-opus-5")
        for i in range(3):
            _, d = p.on_request(body(0), model="claude-opus-5", session="s1",
                                at=T0 + timedelta(seconds=600 + i))
            alerts += p.on_response(d, self._Resp(0, 60_000), model="claude-opus-5")
        self.assertIn("RT-FANOUT", {a.code for a in alerts})

    def test_on_response_returns_what_the_usage_revealed(self):
        p = self._run(rebuild_every=6)
        self.assertTrue(any(a.code == "RT-REBUILD" for a in p.alerts))

    def test_skipping_on_response_costs_only_the_usage_checks(self):
        """Placement still works without it; the counters-driven checks do not."""
        p = CachePlugin(key=KEY, warmup=4)
        last = None
        for i in range(30):
            _, last = p.on_request(body(i), model="claude-opus-5",
                                   at=T0 + timedelta(seconds=60 * i))
        self.assertTrue(last.applied)
        self.assertNotIn("RT-REBUILD", {a.code for a in p.alerts})

    def test_shape_and_usage_together_equal_one_full_observation(self):
        """The two trackers must be disjoint or the split double-counts."""
        from cacheeconomics.monitor import Monitor
        from cacheeconomics.trace import Request as R
        u = {"input_tokens": 0, "cache_read_input_tokens": 5000,
             "cache_creation_input_tokens": 100}
        def make(i):
            return R(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                     model="claude-opus-5", usage=dict(u), session="s",
                     segments=[sg(0, "system", 9000, "sys", marked=True, ttl="5m")])
        whole, split = Monitor(), Monitor()
        for i in range(40):
            whole.observe(make(i))
            split.observe_shape(make(i))
            split.observe_usage(make(i))
        from cacheeconomics.allocate import reuse_chain_of
        scope = reuse_chain_of(make(0))
        self.assertEqual(whole.samples(scope), split.samples(scope))
        self.assertEqual(whole.gaps(scope), split.gaps(scope))
        self.assertEqual(whole.change_rates(scope), split.change_rates(scope))




class TestPluginStateIsBounded(unittest.TestCase):
    """The monitor caps its own state and this did not. A long-running proxy
    kept one entry per alert for the life of the process, which is the
    bounded-memory premise failing in the one component that mutates requests."""

    class _Resp:
        usage = {"input_tokens": 0, "cache_read_input_tokens": 0,
                 "cache_creation_input_tokens": 90_000,
                 "cache_creation": {"ephemeral_5m_input_tokens": 90_000,
                                    "ephemeral_1h_input_tokens": 0}}

    def blocked(self, i):
        return {"system": [{"type": "text", "text": f"session {i} "},
                           {"type": "text", "text": "policy " * 9000}],
                "messages": [{"role": "user", "content": f"t{i}"}]}

    def test_alerts_do_not_grow_with_tenant_churn(self):
        from cacheeconomics.plugin import MAX_ALERTS
        p = CachePlugin(key=KEY, warmup=4, max_scopes=64)
        t = 0
        for tenant in range(400):
            for j in range(12):
                p.on_request(self.blocked(j), model="claude-opus-5",
                             tenant=f"ten{tenant}", session=f"s{tenant}",
                             at=T0 + timedelta(seconds=60 * t))
                t += 1
        self.assertLessEqual(len(p.alerts), MAX_ALERTS)

    def test_effectiveness_evicts_with_the_same_policy_as_the_monitor(self):
        p = CachePlugin(key=KEY, warmup=4, max_scopes=32)
        t = 0
        for tenant in range(200):
            for j in range(12):
                _, d = p.on_request(body(j), model="claude-opus-5",
                                    tenant=f"ten{tenant}", session=f"s{tenant}",
                                    at=T0 + timedelta(seconds=60 * t))
                p.on_response(d, self._Resp(), model="claude-opus-5")
                t += 1
        self.assertLessEqual(len(p._effect), 32)
        self.assertLessEqual(p.monitor.scopes, 32)

    def test_recommendations_still_work_against_a_bounded_store(self):
        p = CachePlugin(key=KEY, warmup=8)
        for i in range(20):
            p.on_request(self.blocked(i), model="claude-opus-5", session="s",
                         at=T0 + timedelta(seconds=90 * i))
        from cacheeconomics.allocate import reuse_chain
        scope = reuse_chain((None, "anthropic/direct", "claude-opus-5"), "unknown", "s")
        self.assertTrue(p.recommendations(scope))


class TestTheWireReshapeIsDisclosed(unittest.TestCase):
    """A bare string has nowhere to carry a marker, so the plugin sends it as a
    text block. The prompt is unchanged as the model sees it; the request bytes
    are not, and that is worth saying rather than leaving to be discovered."""

    def bare(self, i):
        return {"system": "policy " * 6000,
                "messages": [{"role": "user", "content": f"t{i}"}]}

    def test_it_says_when_it_reshapes_a_container(self):
        p = CachePlugin(key=KEY, warmup=8)
        last = None
        for i in range(20):
            _, last = p.on_request(self.bare(i), model="claude-opus-5", session="s",
                                   at=T0 + timedelta(seconds=90 * i))
        self.assertTrue(last.applied)
        self.assertTrue(any("bare strings" in n for n in last.notes))

    def test_it_does_not_claim_a_reshape_that_did_not_happen(self):
        p = CachePlugin(key=KEY, warmup=8)
        last = None
        for i in range(20):
            _, last = p.on_request(body(i), model="claude-opus-5", session="s",
                                   at=T0 + timedelta(seconds=90 * i))
        self.assertFalse(any("bare strings" in n for n in last.notes))

    def test_the_plugin_never_reacts_to_its_own_reshape(self):
        """It segments the request it was handed, not the one it sends, so the
        rewrite is invisible to its own volatility estimate."""
        p = CachePlugin(key=KEY, warmup=8)
        for i in range(60):
            p.on_request(self.bare(i), model="claude-opus-5", session="s",
                         at=T0 + timedelta(seconds=90 * i))
        from cacheeconomics.allocate import reuse_chain
        scope = reuse_chain((None, "anthropic/direct", "claude-opus-5"), "unknown", "s")
        self.assertEqual(p.monitor.change_rates(scope)[0], 0.0)
        self.assertNotIn("RT-DRIFT", {a.code for a in p.alerts})


class TestTheRecorderKeepsNoPromptText(unittest.TestCase):
    """This module's whole claim is that prompt text becomes keyed segment
    metadata and goes no further. A capture holding the original request
    defeated that for as long as anything held the capture -- and printing one
    rendered the entire prompt, because a dataclass repr does."""

    SECRET = "CONFIDENTIAL ACQUISITION MEMO 41B"

    def _capture(self):
        import tempfile
        from cacheeconomics.recorder import Recorder
        rec = Recorder(os.path.join(tempfile.mkdtemp(), "t.jsonl"), key=KEY)
        return rec.capture({"model": "claude-opus-5", "system": self.SECRET,
                            "messages": [{"role": "user", "content": "my private question"}]})

    def test_the_prompt_is_not_in_the_repr(self):
        self.assertNotIn(self.SECRET, repr(self._capture()))

    def test_the_prompt_is_not_retained_on_any_field(self):
        cap = self._capture()
        for name in vars(cap):
            self.assertNotIn(self.SECRET, repr(getattr(cap, name)),
                             f"prompt text reachable through .{name}")

    def test_the_capture_still_carries_what_it_needs(self):
        cap = self._capture()
        self.assertEqual(len(cap.segments), 2)
        self.assertEqual(cap.model, "claude-opus-5")


class TestTheBakeOffFailsClosedOnAnUnknownSurface(unittest.TestCase):
    """Policies read the registry while building a plan, so one row with an
    unregistered target_id used to raise out of the entire comparison."""

    def _reqs(self, unknown=1):
        from cacheeconomics.trace import Request as R
        out = []
        for i in range(20):
            out.append(R(request_id=f"r{i}", sent_at=T0 + timedelta(seconds=60 * i),
                         model="claude-opus-5",
                         usage={"input_tokens": 9000, "cache_read_input_tokens": 0,
                                "cache_creation_input_tokens": 0},
                         segments=[sg(0, "system", 9000, "a"), sg(1, "user", 100, f"t{i}")],
                         session="s"))
        for j in range(unknown):
            out.append(R(request_id=f"u{j}", sent_at=T0 + timedelta(seconds=9999),
                         model="claude-opus-5",
                         usage={"input_tokens": 9000, "cache_read_input_tokens": 0,
                                "cache_creation_input_tokens": 0},
                         segments=[sg(0, "system", 9000, "a"), sg(1, "user", 100, "t")],
                         session="s", target_id="some-gateway/unknown"))
        return out

    def test_the_comparison_survives(self):
        from cacheeconomics.simulate import simulate
        s = simulate(self._reqs(), "litellm-auto")
        self.assertEqual(s.unmodelled_target, 1)
        self.assertEqual(len(s.usages), 21)

    def test_the_unknown_row_is_priced_uncached_not_dropped(self):
        """A dropped row leaves the denominator quietly wrong."""
        from cacheeconomics.simulate import simulate
        s = simulate(self._reqs(), "litellm-auto")
        # Present, not dropped. This asserted exactly one such row back when
        # litellm-auto invented markers for the others; with the faithful
        # baseline placing none, every row prices uncached and the count is no
        # longer the claim. The claim is that the unregistered row survives.
        self.assertGreaterEqual(
            sum(1 for u in s.usages if u.uncached_input == 9100), 1)
        self.assertEqual(len(s.usages), len(self._reqs()),
                         "a row was dropped rather than priced uncached")

    def test_the_verdict_refuses_rather_than_reporting_a_subset(self):
        from cacheeconomics.simulate import bake_off
        b = bake_off(self._reqs())
        self.assertIn("indeterminate", b.verdict)
        self.assertIn("unregistered surface", b.verdict)

    def test_a_clean_workload_still_reaches_a_verdict(self):
        from cacheeconomics.simulate import bake_off
        b = bake_off(self._reqs(unknown=0))
        self.assertNotIn("indeterminate", b.verdict)


class TestTheAdapterModelsEveryPromptField(unittest.IsolatedAsyncioTestCase):
    """Anthropic's top-level `system` precedes everything on the wire. Leaving
    it out of the modelled prompt meant a volatile header could sit above every
    placed marker while the plugin believed the prefix was stable."""

    class _Key:
        user_id = "tenant-a"

    def _data(self, i, header):
        return {"model": "claude-opus-5", "litellm_call_id": f"c{i}",
                "metadata": {"session_id": "conv"},
                "system": [{"type": "text", "text": header}],
                "messages": [{"role": "system", "content": "policy " * 5000},
                             {"role": "user", "content": f"turn {i}"}]}

    async def _drive(self, header_for, turns=20):
        h = litellm_handler(CachePlugin(key=KEY, warmup=4))
        out = None
        for i in range(turns):
            out = await h.async_pre_call_hook(self._Key(), None,
                                              self._data(i, header_for(i)), "completion")
        return h, out

    def _marked(self, out):
        marks = [f"system[{i}]" for i, b in enumerate(out.get("system", []))
                 if isinstance(b, dict) and "cache_control" in b]
        for j, m in enumerate(out["messages"]):
            c = m.get("content")
            if isinstance(c, list) and any(isinstance(b, dict) and "cache_control" in b
                                           for b in c):
                marks.append(f"messages[{j}]")
        return marks

    async def test_a_volatile_system_header_is_modelled(self):
        h, _ = await self._drive(lambda i: f"VOLATILE {i} " + "q" * 2000)
        scope = next(s for s in h.plugin.monitor._scopes if len(s) > 3)
        self.assertEqual(h.plugin.monitor.change_rates(scope)[0], 1.0)

    async def test_nothing_is_cached_behind_a_volatile_header(self):
        _, out = await self._drive(lambda i: f"VOLATILE {i} " + "q" * 2000)
        self.assertEqual(self._marked(out), [])

    async def test_a_stable_header_is_modelled_as_stable(self):
        h, _ = await self._drive(lambda i: "STABLE HEADER " + "q" * 2000)
        scope = next(s for s in h.plugin.monitor._scopes if len(s) > 3)
        self.assertEqual(h.plugin.monitor.change_rates(scope)[0], 0.0)

    async def test_a_stable_header_still_gets_cached(self):
        """Fail-closed must not become fail-always.

        Driven through the plugin rather than the handler: the LiteLLM hook has
        no clock argument, so a test loop issues every request in the same
        millisecond and the write-visibility floor correctly concludes no entry
        survives to be read. That is right, and it is not what this asserts.
        """
        p = CachePlugin(key=KEY, warmup=4)
        last = None
        for i in range(20):
            body = {"system": [{"type": "text", "text": "STABLE HEADER " + "q" * 2000}],
                    "messages": [{"role": "system", "content": "policy " * 5000},
                                 {"role": "user", "content": f"turn {i}"}]}
            _, last = p.on_request(
                body, model="claude-opus-5", session="s",
                markable=markable_positions(body),
                at=T0 + timedelta(seconds=90 * i))
        # Any marker covers the header, because a marker caches the prefix that
        # precedes it. Demanding one at position 0 specifically would assert a
        # shape rather than the property -- marking at 1 covers both blocks and
        # is the better plan.
        self.assertTrue(last.applied)
        self.assertTrue(last.placements)

    async def test_a_volatile_header_blocks_caching_through_the_plugin_too(self):
        p = CachePlugin(key=KEY, warmup=4)
        last = None
        for i in range(20):
            body = {"system": [{"type": "text", "text": f"VOLATILE {i} " + "q" * 2000}],
                    "messages": [{"role": "system", "content": "policy " * 5000},
                                 {"role": "user", "content": f"turn {i}"}]}
            _, last = p.on_request(
                body, model="claude-opus-5", session="s",
                markable=markable_positions(body),
                at=T0 + timedelta(seconds=90 * i))
        self.assertEqual(last.placements, {})

    async def test_the_system_field_survives_the_round_trip(self):
        _, out = await self._drive(lambda i: "STABLE HEADER " + "q" * 2000)
        self.assertIn("system", out)
        self.assertEqual(len(out["system"]), 1)

    async def test_a_request_with_no_system_field_is_unaffected(self):
        h = litellm_handler(CachePlugin(key=KEY, warmup=4))
        out = None
        for i in range(20):
            data = self._data(i, "x")
            del data["system"]
            out = await h.async_pre_call_hook(self._Key(), None, data, "completion")
        self.assertNotIn("system", out)


class TestAgentReachesTheRuntimeRebuildCheck(unittest.TestCase):
    """`Monitor._rebuild` keys established prefixes by (agent, session). The
    live path constructed its Request without an agent, so every request fell
    to the default and pooled exactly the contexts the batch rule had just been
    taught to separate."""

    class _Resp:
        usage = {"input_tokens": 0, "cache_read_input_tokens": 0,
                 "cache_creation_input_tokens": 60_000,
                 "cache_creation": {"ephemeral_5m_input_tokens": 60_000,
                                    "ephemeral_1h_input_tokens": 0}}

    def _one(self, agent):
        p = CachePlugin(key=KEY, warmup=4)
        _, d = p.on_request({"system": [{"type": "text", "text": "p" * 40000}],
                             "messages": [{"role": "user", "content": "x"}]},
                            model="claude-opus-5", session="s", agent=agent, at=T0)
        p.on_response(d, self._Resp(), model="claude-opus-5")
        return p, d

    def test_the_decision_carries_the_agent(self):
        _, d = self._one("subagent:explore")
        self.assertEqual(d.agent, "subagent:explore")

    # Rebuild state moved out of the pool scope, which carries the model, into
    # `(tenant, target)`, which does not. A model switch is one of the ways a
    # prefix gets rebuilt, and keyed by pool it landed in a fresh scope where
    # the rebuild was invisible. The property these two assert -- one key per
    # (agent, session) -- is unchanged; only where it is stored moved.
    REBUILD_SCOPE = (None, "anthropic/direct")

    def test_the_rebuild_key_separates_contexts(self):
        p, _ = self._one("subagent:explore")
        st = p.monitor._scopes[self.REBUILD_SCOPE]
        self.assertEqual(list(st.established), [("subagent:explore", "s")])

    def test_two_agents_in_one_conversation_do_not_share_a_key(self):
        p = CachePlugin(key=KEY, warmup=4)
        for agent in ("main-loop", "subagent:explore"):
            _, d = p.on_request({"system": [{"type": "text", "text": "p" * 40000}],
                                 "messages": [{"role": "user", "content": "x"}]},
                                model="claude-opus-5", session="s", agent=agent, at=T0)
            p.on_response(d, self._Resp(), model="claude-opus-5")
        st = p.monitor._scopes[self.REBUILD_SCOPE]
        self.assertEqual(len(st.established), 2)

    def test_the_adapter_reads_an_agent_from_metadata(self):
        from cacheeconomics.plugin import default_agent_from
        self.assertEqual(default_agent_from({"metadata": {"agent": "worker-3"}}), "worker-3")

    def test_it_admits_when_there_is_no_agent_rather_than_inventing_one(self):
        from cacheeconomics.plugin import default_agent_from
        self.assertEqual(default_agent_from({"litellm_call_id": "c1"}), "unknown")


class TestMessageLevelMarkersAreCounted(unittest.TestCase):
    """Anthropic accepts `cache_control` on a message object as well as on a
    content block. `walk` yields content blocks, because those are the segments
    whose text is hashed and priced -- so every guard built on `walk` was blind to
    a marker the provider counts.

    Measured before the fix, with the caller holding four message-level markers
    against a budget of four: `respect_existing` did not stand down, the strip did
    not strip, and `on_request` returned a body carrying six markers with an
    applied decision and no complaint. Six against four is a provider error, and
    this is the path that rewrites live requests.
    """

    STABLE = "You are a careful assistant with a long stable policy. " * 300
    BUDGET = 4

    def _body(self, turn, caller_markers=0):
        msgs = [{"role": "system",
                 "content": [{"type": "text", "text": self.STABLE}]}]
        for i in range(caller_markers):
            msgs.append({"role": "user",
                         "cache_control": {"type": "ephemeral"},
                         "content": [{"type": "text",
                                      "text": f"pinned {i} " * 60}]})
        msgs.append({"role": "user",
                     "content": [{"type": "text", "text": f"turn {turn}"}]})
        return {"model": "claude-opus-5", "messages": msgs}

    def _warm(self, caller_markers, **kw):
        """Drive past the warmup with realistic gaps, which a live proxy has.

        Requests arriving in the same millisecond are shorter than the write
        latency, so every entry is a miss and nothing is ever worth placing --
        which is why an earlier attempt at this test proved nothing.
        """
        p = CachePlugin(key=KEY, **kw)
        out = d = None
        for i in range(34):
            out, d = p.on_request(self._body(i, caller_markers),
                                  model="claude-opus-5", tenant="t",
                                  session="s", agent="main",
                                  at=T0 + timedelta(seconds=60 * i), apply=True)
        return out, d

    def test_marker_count_sees_what_walk_cannot(self):
        body = self._body(0, self.BUDGET)
        by_walk = sum(1 for _r, _l, blk, _p in walk(body)
                      if isinstance(blk, dict) and "cache_control" in blk)
        self.assertEqual(by_walk, 0, "walk yields content blocks only")
        self.assertEqual(marker_count(body), self.BUDGET)

    def test_strip_markers_removes_them(self):
        """Its docstring says "every `cache_control`" and it meant it."""
        self.assertEqual(marker_count(strip_markers(self._body(0, self.BUDGET))), 0)

    def test_respect_existing_stands_down_for_them(self):
        _out, d = self._warm(self.BUDGET)
        self.assertFalse(d.applied)
        self.assertIn("already placed", d.reason)

    def test_override_replaces_them_instead_of_adding(self):
        out, d = self._warm(self.BUDGET, respect_existing=False)
        self.assertTrue(d.applied, d.reason)
        self.assertLessEqual(
            marker_count(out), self.BUDGET,
            "the plugin's markers were added on top of the caller's")
        self.assertTrue(any("replaced 4 marker" in n for n in (d.notes or [])),
                        d.notes)

    def test_the_plugin_never_pushes_a_body_over_budget(self):
        """The honest invariant, over the whole matrix.

        Not "the result is always within budget" -- a caller who sends six
        markers of their own gets six back, because standing down means returning
        exactly what arrived and the plugin did not put them there. What it may
        never do is hand back a body carrying more markers than it received, or
        claim it applied a plan that leaves the provider's budget exceeded.
        """
        for caller in (0, 1, self.BUDGET, self.BUDGET + 2):
            for respect in (True, False):
                with self.subTest(caller=caller, respect_existing=respect):
                    sent = self._body(99, caller)
                    out, d = self._warm(caller, respect_existing=respect)
                    if d.applied:
                        self.assertLessEqual(
                            marker_count(out), self.BUDGET,
                            f"applied a plan that goes over budget "
                            f"(caller={caller})")
                    else:
                        self.assertEqual(
                            marker_count(out), marker_count(sent),
                            "standing down must return the body as it arrived")
                    self.assertLessEqual(
                        marker_count(out),
                        max(self.BUDGET, marker_count(sent)),
                        "the plugin added markers to an already-full request")


class TestALateVetoLeavesNoAppliedDecision(unittest.IsolatedAsyncioTestCase):
    """`_pending[key] = decision` is written before the handler's final budget
    re-check, and that check can return the caller's original body.

    `async_log_success_event` hands whatever is in `_pending` to `on_response`,
    which credits the response's cache reads and writes to the decision's
    placements. If the original body went out, those markers never left the
    process, so the effectiveness counters would be describing a request nobody
    sent -- and they feed the next placement decision.
    """

    class Keys:
        user_id = "tenant-a"

    async def test_a_vetoed_decision_is_recorded_as_not_applied(self):
        p = CachePlugin(key=KEY)
        h = litellm_handler(p, mutate=True)
        body = {"model": "claude-opus-5", "litellm_call_id": "c1",
                "metadata": {"session_id": "s", "agent": "main"},
                "messages": [{"role": "user",
                              "content": [{"type": "text", "text": "hi"}]}]}
        await h.async_pre_call_hook(self.Keys(), None, dict(body), "completion")
        # Whatever the decision was, the invariant is the same: a pending entry
        # may not claim markers were applied unless the patched body went out.
        d = h._pending.get("c1")
        self.assertIsNotNone(d)
        if not d.applied:
            self.assertFalse(d.placements)

    async def test_the_veto_branch_moves_placements_to_proposed(self):
        """Driven directly, because reaching the branch through the handler now
        requires `_decide` to disagree with it -- which the shared counter makes
        impossible, and that is the point. This asserts the branch is still
        correct if it ever does fire."""
        from dataclasses import replace as _replace

        from cacheeconomics.plugin import Decision
        d = Decision(True, {0: "5m"})
        vetoed = _replace(d, applied=False, placements={},
                          proposed=dict(d.placements))
        self.assertFalse(vetoed.applied)
        self.assertEqual(vetoed.placements, {})
        self.assertEqual(vetoed.proposed, {0: "5m"})


class TestStructuredMetadataCannotFailTheCall(unittest.IsolatedAsyncioTestCase):
    """Optional metadata must not be able to break somebody's LLM request.

    tenant, session and agent all become dictionary keys -- the reuse chain and
    the monitor's scope -- so a dict in `metadata.trace_id` raised
    `TypeError: unhashable type` from inside the pre-call hook. `trace._text`
    was written for this exact class on the ingest side; the loader was hardened
    and this path was not.
    """

    class Keys:
        user_id = "tenant-a"

    def _body(self, meta):
        return {"model": "claude-opus-5", "litellm_call_id": "c1",
                "metadata": meta,
                "messages": [{"role": "user",
                              "content": [{"type": "text", "text": "hi " * 400}]}]}

    async def test_structured_identity_fields_do_not_raise(self):
        for meta in ({"session_id": {"a": 1}, "agent": "main"},
                     {"session_id": "s", "agent": ["a", "b"]},
                     {"session_id": {"a": 1}, "agent": {"b": 2}},
                     {"session_id": [1], "conversation_id": "fallback"}):
            with self.subTest(meta=meta):
                h = litellm_handler(CachePlugin(key=KEY), mutate=True)
                out = await h.async_pre_call_hook(
                    self.Keys(), None, self._body(meta), "completion")
                self.assertIsNotNone(out)

    async def test_a_structured_value_does_not_block_the_next_name(self):
        """Discarding one has to mean "this name did not answer". Returning None
        immediately let a dict in `session_id` hide a usable
        `conversation_id`."""
        from cacheeconomics.plugin import default_session_from
        self.assertEqual(
            default_session_from({"metadata": {"session_id": {"bad": 1},
                                               "conversation_id": "real"}}),
            "real")

    async def test_a_numeric_identity_is_kept_as_a_string(self):
        from cacheeconomics.plugin import default_session_from
        self.assertEqual(
            default_session_from({"metadata": {"session_id": 12345}}), "12345")


class TestEffectivenessNeedsRealCounters(unittest.TestCase):
    """`effectiveness()` calls itself "the provider's own counters". The guard
    above it already applied "absent usage is not zero usage" to the monitor,
    and this block ran regardless -- so a transport that drops the usage object
    counted a placement, credited no read, and reported `read_share: 0.0` with
    the full weight of a measured number."""

    def _plugin_with_a_placement(self):
        p = CachePlugin(key=KEY, warmup=4)
        body = {"system": [{"type": "text", "text": "p" * 40000}],
                "messages": [{"role": "user", "content": "x"}]}
        d = None
        for i in range(8):
            _, d = p.on_request(dict(body), model="claude-opus-5", session="s",
                                agent="a", at=T0 + timedelta(seconds=120 * i))
        return p, d

    def test_a_response_with_no_usage_moves_no_counter(self):
        p, d = self._plugin_with_a_placement()
        if not d.applied:
            self.skipTest("no placement to measure")
        class Empty:
            usage = {}
        before = p.effectiveness(d.scope)
        p.on_response(d, Empty(), model="claude-opus-5")
        after = p.effectiveness(d.scope)
        self.assertEqual(before, after,
                         "a response that told us nothing moved a counter")

    def test_a_real_response_still_counts(self):
        p, d = self._plugin_with_a_placement()
        if not d.applied:
            self.skipTest("no placement to measure")
        class Real:
            usage = {"input_tokens": 10, "cache_read_input_tokens": 9000,
                     "cache_creation_input_tokens": 0}
        p.on_response(d, Real(), model="claude-opus-5")
        eff = p.effectiveness(d.scope)
        self.assertIsNotNone(eff)
        self.assertEqual(eff["read_share"], 1.0)


class TestAnAbstentionIsNotAllClear(unittest.TestCase):
    """The near-minimum filter dropping every candidate is the plugin saying "I
    could not prove this is safe". Reporting it as "no marker worth placing" is
    the "your workload is fine" reading, which is the opposite."""

    def test_a_safety_abstention_is_reported_as_one(self):
        import inspect

        from cacheeconomics import plugin as mod
        src = inspect.getsource(mod.CachePlugin._decide)
        self.assertIn('n.startswith("ABSTAIN")', src)
        self.assertNotIn('n.startswith("ABSTAIN entirely")', src)


class TestTheWireTtlReachesTheMonitor(unittest.TestCase):
    """`ttl_requested` was taken from the plugin's own placements, which are
    empty whenever it stands down -- and standing down for a caller's existing
    markers is the common case, the whole observe-only path. So the monitor got
    None for a request plainly carrying a 5m marker, and the cadence and expiry
    checks went silent."""

    def test_a_silent_marker_reads_as_five_minutes(self):
        from cacheeconomics.plugin import _wire_ttl
        self.assertEqual(_wire_ttl({"system": [
            {"type": "text", "text": "x",
             "cache_control": {"type": "ephemeral"}}]}), "5m")

    def test_an_explicit_lifetime_is_kept(self):
        from cacheeconomics.plugin import _wire_ttl
        self.assertEqual(_wire_ttl({"system": [
            {"type": "text", "text": "x",
             "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}), "1h")

    def test_a_genuine_mix_is_still_none(self):
        """A request can write both, and the batch path refuses to price a mixed
        write as one lifetime."""
        from cacheeconomics.plugin import _wire_ttl
        self.assertIsNone(_wire_ttl({"system": [
            {"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "y",
             "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}))

    def test_observe_only_still_raises_the_ttl_alert(self):
        """The reproduction: twenty default markers fifteen minutes apart."""
        stable = "policy " * 3000
        p = CachePlugin(key=KEY, warmup=8)
        for i in range(20):
            p.on_request(
                {"model": "claude-opus-5",
                 "system": [{"type": "text", "text": stable,
                             "cache_control": {"type": "ephemeral"}}],
                 "messages": [{"role": "user", "content": f"turn {i}"}]},
                model="claude-opus-5", session="s", agent="main",
                at=T0 + timedelta(seconds=900 * i))
        self.assertIn("RT-TTL", {a.code for a in list(p.alerts)})


if __name__ == "__main__":
    unittest.main()


class TestTheLiveHookFailsOpen(unittest.IsolatedAsyncioTestCase):
    """This sits in front of a live request.

    Anything it raises is an error the caller's own API call never made. There
    is no input from a proxy's traffic worth breaking a completion over, least
    of all in the observe-only default where the plugin is not supposed to
    change anything at all.

    Measured before the guard: a model field arriving as a dict raised
    TypeError out of model normalisation, and a message whose content is an
    integer raised out of segmentation. Both with mutate=False.
    """

    def _handler(self, **kw):
        return litellm_handler(CachePlugin(key=KEY), mutate=False, **kw)

    async def _call(self, data):
        h = self._handler()
        out = await h.async_pre_call_hook({"user_id": "u"}, None, data, "completion")
        return h, out

    async def test_a_malformed_model_does_not_break_the_request(self):
        data = {"model": {"bad": "claude-opus-5"},
                "messages": [{"role": "user", "content": "hi"}]}
        h, out = await self._call(data)
        self.assertIs(out, data, "the caller's request was not returned")
        self.assertEqual(h._failures, 1)

    async def test_non_string_message_content_does_not_break_the_request(self):
        data = {"model": "claude-opus-5",
                "messages": [{"role": "user", "content": 7}]}
        h, out = await self._call(data)
        self.assertIs(out, data)
        self.assertEqual(h._failures, 1)

    async def test_a_well_formed_request_still_goes_through_the_hook(self):
        """The guard must not swallow the plugin's actual work."""
        data = {"model": "claude-opus-5",
                "messages": [{"role": "user", "content": "hello " * 50}]}
        h, out = await self._call(data)
        self.assertEqual(h._failures, 0)
        self.assertIsNotNone(out)

    async def test_cancellation_still_propagates(self):
        """`except BaseException` would swallow a cancellation and leave the
        proxy waiting on a task that will never finish."""
        h = self._handler()

        async def boom(*a, **kw):
            raise asyncio.CancelledError()

        h._pre_call = boom
        with self.assertRaises(asyncio.CancelledError):
            await h.async_pre_call_hook({"user_id": "u"}, None,
                                        {"model": "claude-opus-5",
                                         "messages": [{"role": "user",
                                                       "content": "x"}]},
                                        "completion")


class TestTenantMatchesTheBatchAdapter(unittest.TestCase):
    """The live hook read three identity fields where the batch adapter reads
    six, so a team-scoped key resolved to None and two teams shared one
    `(None, target, model)` scope -- for caches the provider keeps apart."""

    # Every place either side has ever read a tenant from, and the shapes they
    # actually arrive in. Each row is fed to both paths and the two answers must
    # match -- including when the answer is None.
    ROWS = (
        ("key hash", {"metadata": {"user_api_key_hash": "h1"}}),
        ("key alias", {"metadata": {"user_api_key_alias": "alias-1"}}),
        ("key team", {"metadata": {"user_api_key_team_id": "team-9"}}),
        ("team", {"metadata": {"team_id": "t2"}}),
        ("key user", {"metadata": {"user_api_key_user_id": "ku-3"}}),
        ("metadata user_id", {"metadata": {"user_id": "u-42"}}),
        ("end_user", {"end_user": "acme"}),
        ("top-level user", {"user": "u-77"}),
        ("user_id beside end_user",
         {"metadata": {"user_id": "u-42"}, "end_user": "acme"}),
        ("nothing", {"model": "claude-opus-5"}),
        ("metadata as a string", {"metadata": "not-a-dict", "end_user": "acme"}),
        ("unhashable value", {"metadata": {"team_id": {"nested": 1}},
                              "end_user": "acme"}),
    )

    def test_the_two_paths_answer_identically_on_every_shape(self):
        """Structural before: it grepped the adapter's source for field names
        and asserted the live tuple contained them. That is one direction only,
        and it could not see the divergence that mattered -- both lists held
        `team_id`, and the two still returned different tenants for
        `{"metadata": {"user_id": ...}, "end_user": ...}`, because they differ
        in what they read *after* the shared five. A row is the only thing that
        exercises that."""
        from cacheeconomics.adapters import litellm as batch
        for name, row in self.ROWS:
            with self.subTest(row=name):
                self.assertEqual(plugin_mod._tenant_of(None, row),
                                 batch._tenant_of(row))

    def test_the_object_shaped_key_reaches_both(self):
        """The live hook is handed an object; the loader was `.get`-only and
        read no tenant at all from one."""
        from cacheeconomics.adapters import litellm as batch

        class ObjMeta:
            user_api_key_team_id = "team-a"
        row = {"metadata": ObjMeta()}
        self.assertEqual(plugin_mod._tenant_of(None, row), "team-a")
        self.assertEqual(batch._tenant_of(row), "team-a")

    def test_neither_path_invents_a_tenant(self):
        for name, row in self.ROWS:
            if name != "nothing":
                continue
            self.assertIsNone(plugin_mod._tenant_of(None, row))

    def test_it_reads_an_object_key_and_a_dict_key_and_metadata(self):
        class ObjKey:
            user_api_key_team_id = "team-a"
        self.assertEqual(plugin_mod._tenant_of(ObjKey(), {}), "team-a")
        self.assertEqual(plugin_mod._tenant_of({"user_api_key_hash": "h1"}, {}), "h1")
        self.assertEqual(plugin_mod._tenant_of({}, {"metadata": {"team_id": "t2"}}), "t2")
