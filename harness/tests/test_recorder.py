"""The recorder: does it capture enough to answer a counterfactual, and no more.

Two properties matter more than the rest. It must never hold prompt text, and a
streamed or failed call must be recorded rather than quietly dropped — a
recorder that skips the calls agentic workloads actually make reports a healthy
coverage number over an analysis that missed the traffic.
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cacheeconomics.recorder import (Recorder, segments_from_request,  # noqa: E402
                                     usage_from_response)
from cacheeconomics.simulate import bake_off  # noqa: E402
from cacheeconomics.trace import Tier, load_jsonl  # noqa: E402

KEY = b"test-key-not-a-real-one"

REQUEST = {
    "model": "claude-opus-5",
    "tools": [{"name": "read", "description": "d" * 100},
              {"name": "write", "description": "d" * 100}],
    "system": [{"type": "text", "text": "volatile header"},
               {"type": "text", "text": "s" * 4000,
                "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
    "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
}

RESPONSE = {"id": "msg_1", "usage": {
    "input_tokens": 300, "output_tokens": 40,
    "cache_read_input_tokens": 0, "cache_creation_input_tokens": 4700,
    "cache_creation": {"ephemeral_5m_input_tokens": 0,
                       "ephemeral_1h_input_tokens": 4700}}}


class TestSegmentation(unittest.TestCase):

    def test_wire_order_is_tools_then_system_then_messages(self):
        """The cache matches in wire order, so the index has to reflect it."""
        segs = segments_from_request(REQUEST, KEY)
        self.assertEqual([s["role"] for s in segs],
                         ["tools", "tools", "system", "system", "user"])
        self.assertEqual([s["index"] for s in segs], list(range(5)))

    def test_each_tool_is_its_own_segment(self):
        """Tool reordering is a real cause of misses and is invisible if the
        whole array is one hash. Labelled by position, not by name."""
        segs = segments_from_request(REQUEST, KEY)
        labels = [s["label"] for s in segs if s["role"] == "tools"]
        self.assertEqual(labels, ["tool[0]", "tool[1]"])

    def test_a_tool_name_is_prompt_content_and_does_not_survive(self):
        """`query_internal_billing_ledger` names an internal API. It is
        model-visible prompt content, and it was being copied verbatim into the
        segment label -- which the recorder persists and the report renders.
        Hashing the block and printing its name beside the hash gives away most
        of what the hash was protecting."""
        req = {"tools": [{"name": "query_internal_billing_ledger",
                          "description": "x" * 500}],
               "messages": [{"role": "user", "content": "hi"}]}
        blob = json.dumps(segments_from_request(req, KEY))
        self.assertNotIn("query_internal_billing_ledger", blob)
        self.assertNotIn("billing", blob)

    def test_no_prompt_text_is_retained(self):
        blob = json.dumps(segments_from_request(REQUEST, KEY))
        self.assertNotIn("volatile header", blob)
        self.assertNotIn("hello", blob)
        self.assertNotIn("ssss", blob)

    def test_ids_are_keyed_not_bare_digests(self):
        for s in segments_from_request(REQUEST, KEY):
            self.assertTrue(s["id"].startswith("hmac:"))

    def test_a_different_key_yields_different_ids(self):
        a = segments_from_request(REQUEST, KEY)
        b = segments_from_request(REQUEST, b"another-key")
        self.assertNotEqual([s["id"] for s in a], [s["id"] for s in b])

    def test_dict_key_order_does_not_change_a_segment_id(self):
        """Serialiser-order instability is an artefact, not a finding. Treating
        it as drift would manufacture volatility that costs nobody anything."""
        r1 = {"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]}
        r2 = {"messages": [{"role": "user", "content": [{"text": "x", "type": "text"}]}]}
        self.assertEqual(segments_from_request(r1, KEY)[0]["id"],
                         segments_from_request(r2, KEY)[0]["id"])

    def test_cache_control_and_ttl_are_captured(self):
        segs = segments_from_request(REQUEST, KEY)
        marked = [s for s in segs if s["cache_marked"]]
        self.assertEqual(len(marked), 1)
        self.assertEqual(marked[0]["ttl"], "1h")

    def test_a_recorder_refuses_to_run_without_a_key(self):
        with self.assertRaises(ValueError) as ctx:
            Recorder("/tmp/x.jsonl", key=b"")
        self.assertIn("HMAC key", str(ctx.exception))


class TestUsageCapture(unittest.TestCase):

    def test_the_lifetime_split_is_carried_through(self):
        """Without it the write lifetime is unprovable and nothing prices."""
        u = usage_from_response(RESPONSE)
        self.assertEqual(u["cache_creation"]["ephemeral_1h_input_tokens"], 4700)

    def test_it_reads_sdk_objects_as_well_as_dicts(self):
        class Obj:
            pass
        o, usage = Obj(), Obj()
        usage.input_tokens, usage.output_tokens = 5, 6
        usage.cache_read_input_tokens = 7
        usage.cache_creation_input_tokens = 0
        usage.cache_creation = {}
        o.usage = usage
        self.assertEqual(usage_from_response(o)["cache_read_input_tokens"], 7)


class TestRecordingRoundTrip(unittest.TestCase):

    def _record(self, n=6, fail_last=False, stream=False):
        path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        rec = Recorder(path, key=KEY, tenant="acme")
        for i in range(n):
            req = dict(REQUEST)
            req["messages"] = [{"role": "user",
                                "content": [{"type": "text", "text": f"step {i}"}]}]
            cap = rec.capture(req, agent="coder", session="s1")
            if stream:
                cap.first_token()
                cap.first_token()          # idempotent
            if fail_last and i == n - 1:
                cap.failed(status=429)
            else:
                cap.done(dict(RESPONSE, id=f"msg_{i}"))
        return path, rec

    def test_it_produces_an_instrumented_trace(self):
        path, rec = self._record()
        self.assertEqual(rec.recorded, 6)
        ts = load_jsonl(path)
        self.assertIs(ts.tier, Tier.INSTRUMENTED)
        self.assertEqual(ts.structural_coverage, 1.0)
        os.unlink(path)

    def test_segment_tokens_sum_to_the_billed_input(self):
        """Estimates are scaled to a measured total, so the bytes-per-token
        constant cancels out of the result."""
        path, _ = self._record(n=1)
        ts = load_jsonl(path)
        billed = 300 + 0 + 4700
        self.assertAlmostEqual(sum(s.tokens for s in ts.requests[0].segments),
                               billed, delta=len(ts.requests[0].segments))
        os.unlink(path)

    def test_estimation_is_declared_on_every_row(self):
        path, _ = self._record(n=2)
        with open(path) as f:
            for line in f:
                self.assertTrue(json.loads(line)["tokens_are_estimated"])
        os.unlink(path)

    def test_a_failed_call_is_recorded_not_dropped(self):
        """A change that makes an agent fail faster is not a saving."""
        path, rec = self._record(n=4, fail_last=True)
        self.assertEqual(rec.recorded, 4)
        with open(path) as f:
            rows = [json.loads(l) for l in f]
        self.assertEqual(rows[-1]["status"], 429)
        self.assertEqual(rows[-1]["usage"], {})
        # and the loader excludes it from analysis while counting it in coverage
        ts = load_jsonl(path)
        self.assertEqual(len(ts.analysable), 3)
        self.assertEqual(ts.coverage["total"], 4)
        os.unlink(path)

    def test_streaming_records_a_first_token_time(self):
        path, _ = self._record(n=2, stream=True)
        for r in load_jsonl(path).requests:
            self.assertIsNotNone(r.first_token_at)
        os.unlink(path)

    def test_mixed_lifetimes_record_no_single_requested_ttl(self):
        """Against one aggregate write total the split is unknowable, so the
        recorder must not pick whichever marker sorted first."""
        path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        rec = Recorder(path, key=KEY)
        req = dict(REQUEST)
        req["system"] = [
            {"type": "text", "text": "a" * 100, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "b" * 100,
             "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
        rec.capture(req, agent="a").done(RESPONSE)
        with open(path) as f:
            self.assertIsNone(json.loads(f.read())["ttl_requested"])
        os.unlink(path)

    def test_a_recorded_trace_can_run_a_bake_off(self):
        """The whole point: real structure, so a counterfactual is derivable."""
        path, _ = self._record(n=8)
        b = bake_off(load_jsonl(path).analysable, group="coder")
        self.assertEqual(b.n_requests, 8)
        self.assertEqual(b.unstructured, 0)
        self.assertIsNotNone(b.delta_pct)
        os.unlink(path)



class TestIdentityExcludesCacheMetadata(unittest.TestCase):
    """cache_control is an instruction to the cache, not content.

    Hashing it into the segment id made the same prompt text score as a
    different segment depending on whether it carried a marker -- so the tool
    would have read its own caching recommendations as prompt drift, and reacted
    to them.
    """

    BASE = {"type": "text", "text": "identical prompt text"}

    def _id(self, block):
        return segments_from_request({"system": [block]}, KEY)[0]["id"]

    def test_a_marker_does_not_change_the_segment_id(self):
        marked = dict(self.BASE, cache_control={"type": "ephemeral"})
        self.assertEqual(self._id(self.BASE), self._id(marked))

    def test_changing_the_lifetime_does_not_change_the_segment_id(self):
        a = dict(self.BASE, cache_control={"type": "ephemeral", "ttl": "5m"})
        b = dict(self.BASE, cache_control={"type": "ephemeral", "ttl": "1h"})
        self.assertEqual(self._id(a), self._id(b))

    def test_the_marker_is_still_recorded_separately(self):
        marked = dict(self.BASE, cache_control={"type": "ephemeral", "ttl": "1h"})
        seg = segments_from_request({"system": [marked]}, KEY)[0]
        self.assertTrue(seg["cache_marked"])
        self.assertEqual(seg["ttl"], "1h")

    def test_changing_the_actual_text_still_changes_the_id(self):
        other = dict(self.BASE, text="different prompt text")
        self.assertNotEqual(self._id(self.BASE), self._id(other))

    def test_marker_bytes_do_not_skew_token_attribution(self):
        """Otherwise a marked segment is credited tokens for its own metadata."""
        plain = segments_from_request({"system": [self.BASE]}, KEY)
        marked = segments_from_request(
            {"system": [dict(self.BASE, cache_control={"type": "ephemeral"})]}, KEY)
        from cacheeconomics.recorder import _scale_to_measured
        self.assertEqual(_scale_to_measured(plain, 1000)[0]["tokens"],
                         _scale_to_measured(marked, 1000)[0]["tokens"])


class TestAsyncCallsAreRecordedAfterTheyResolve(unittest.TestCase):
    """An earlier version recorded the coroutine as the response, so an async
    call was written out with empty usage and status 200 before the request had
    been sent, and a later failure was never captured."""

    def _rec(self):
        path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        return path, Recorder(path, key=KEY)

    def test_an_async_success_records_the_real_usage(self):
        import asyncio
        path, rec = self._rec()

        async def create(**kwargs):
            return dict(RESPONSE)

        async def go():
            return await rec.messages_create(create, **dict(REQUEST))

        asyncio.run(go())
        with open(path) as f:
            row = json.loads(f.read())
        self.assertEqual(row["status"], 200)
        self.assertEqual(row["usage"]["cache_creation_input_tokens"], 4700)
        os.unlink(path)

    def test_an_async_failure_is_recorded_as_a_failure(self):
        import asyncio
        path, rec = self._rec()

        async def create(**kwargs):
            raise RuntimeError("upstream 529")

        async def go():
            with self.assertRaises(RuntimeError):
                await rec.messages_create(create, **dict(REQUEST))

        asyncio.run(go())
        with open(path) as f:
            row = json.loads(f.read())
        self.assertEqual(row["status"], 0)
        self.assertEqual(row["usage"], {})
        os.unlink(path)

    def test_a_sync_call_is_unaffected(self):
        path, rec = self._rec()
        rec.messages_create(lambda **kw: dict(RESPONSE), **dict(REQUEST))
        with open(path) as f:
            self.assertEqual(json.loads(f.read())["usage"]["input_tokens"], 300)
        os.unlink(path)


class TestIdentityCoversTheContainer(unittest.TestCase):
    """A cached prefix is a tuple of segment ids, so if identity ignores the
    container, moving text between roles reads downstream as a cache hit --
    the exact authority confusion the relocation rules exist to prevent."""

    def _id(self, request):
        return segments_from_request(request, KEY)[0]["id"]

    def test_the_same_text_under_different_roles_is_not_the_same_segment(self):
        sys_id = self._id({"system": "same text"})
        usr_id = self._id({"messages": [{"role": "user", "content": "same text"}]})
        self.assertNotEqual(sys_id, usr_id)

    def test_assistant_and_user_turns_are_distinguished(self):
        a = self._id({"messages": [{"role": "user", "content": "same text"}]})
        b = self._id({"messages": [{"role": "assistant", "content": "same text"}]})
        self.assertNotEqual(a, b)

    def test_block_kind_is_part_of_identity(self):
        """A text block and a tool_result carrying the same string are not
        interchangeable on the wire."""
        a = self._id({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "x"}]}]})
        b = self._id({"messages": [{"role": "user", "content": [
            {"type": "tool_result", "text": "x"}]}]})
        self.assertNotEqual(a, b)

    def test_position_alone_does_not_change_identity(self):
        """Identical text at two positions genuinely is the same content, and
        position is already carried by `index`."""
        segs = segments_from_request(
            {"system": [{"type": "text", "text": "repeated"},
                        {"type": "text", "text": "repeated"}]}, KEY)
        self.assertEqual(segs[0]["id"], segs[1]["id"])
        self.assertEqual([s["index"] for s in segs], [0, 1])


class TestTokenApportionmentIsExact(unittest.TestCase):
    """The parts must sum to the measured whole.

    Rounding each segment independently and clamping to one token let the total
    drift from what the provider billed. That drift feeds bake-off spend and the
    minimum-cacheable check, where enough small segments can push a request past
    a provider threshold on tokens nobody was charged for.
    """

    def _segs(self, sizes):
        return [{"bytes": b} for b in sizes]

    def test_the_total_is_preserved_exactly(self):
        from cacheeconomics.recorder import _scale_to_measured
        import random
        random.seed(11)
        for _ in range(50):
            n, total = random.randint(1, 60), random.randint(1, 80000)
            segs = self._segs([random.randint(1, 9000) for _ in range(n)])
            got = sum(s["tokens"] for s in _scale_to_measured(segs, total))
            self.assertEqual(got, total, f"{n} segments, {total} billed")

    def test_more_segments_than_tokens_does_not_inflate_the_total(self):
        """A request with more segments than billed tokens is a real shape, and
        inventing a token per segment is how the total ran away."""
        from cacheeconomics.recorder import _scale_to_measured
        segs = _scale_to_measured(self._segs([10] * 40), 12)
        self.assertEqual(sum(s["tokens"] for s in segs), 12)
        self.assertTrue(any(s["tokens"] == 0 for s in segs))

    def test_larger_segments_still_receive_more_tokens(self):
        from cacheeconomics.recorder import _scale_to_measured
        segs = _scale_to_measured(self._segs([100, 900]), 1000)
        self.assertLess(segs[0]["tokens"], segs[1]["tokens"])

    def test_without_a_measured_total_it_falls_back_and_still_flags_estimated(self):
        from cacheeconomics.recorder import _scale_to_measured
        segs = _scale_to_measured(self._segs([3600, 3600]), None)
        self.assertTrue(all(s["tokens"] >= 1 for s in segs))



class TestCaptureIsCompleteOnlyOnceWritten(unittest.TestCase):
    """Marking the capture done before the write meant a failure to open,
    flush or fsync left it permanently complete with nothing appended -- losing
    a successful, billed call at exactly the durability boundary."""

    def _rec(self):
        path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        return path, Recorder(path, key=KEY)

    def test_a_failed_write_can_be_retried_and_records_once(self):
        path, rec = self._rec()
        cap = rec.capture(REQUEST, agent="a")
        real, calls = rec._write, {"n": 0}

        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("disk full")
            return real(*a, **k)

        rec._write = flaky
        with self.assertRaises(OSError):
            cap.done(RESPONSE)
        cap.done(RESPONSE)
        with open(path) as f:
            self.assertEqual(sum(1 for _ in f), 1)
        os.unlink(path)

    def test_the_failure_is_not_swallowed(self):
        """A recorder that hides a write failure is worse than one that stops."""
        path, rec = self._rec()
        rec._write = lambda *a, **k: (_ for _ in ()).throw(OSError("nope"))
        with self.assertRaises(OSError):
            rec.capture(REQUEST, agent="a").done(RESPONSE)
        os.unlink(path)

    def test_a_successful_capture_is_still_idempotent(self):
        path, rec = self._rec()
        cap = rec.capture(REQUEST, agent="a")
        cap.done(RESPONSE)
        cap.done(RESPONSE)
        cap.failed(429)
        with open(path) as f:
            self.assertEqual(sum(1 for _ in f), 1)
        os.unlink(path)


class TestStreamingCoversAsyncAndSetupFailure(unittest.TestCase):
    """The proxy implemented only the sync protocol, so an `async with` stream
    could not be wrapped at all and async streamed traffic bypassed the
    recorder; and a failure inside `__enter__` never reached `__exit__`, so a
    stream that died during setup was never recorded. Those are the
    high-volume and the degraded paths, which are the two this exists to keep."""

    class _Sync:
        def __init__(self, boom=False):
            self.boom = boom
        def __enter__(self):
            if self.boom:
                raise RuntimeError("stream setup failed")
            return self
        def __iter__(self):
            yield "delta"
        def get_final_message(self):
            return RESPONSE
        def __exit__(self, *a):
            return False

    class _Async:
        async def __aenter__(self):
            return self
        async def __aiter__(self):
            yield "delta"
        async def get_final_message(self):
            return RESPONSE
        async def __aexit__(self, *a):
            return False

    def _rec(self):
        path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        return path, Recorder(path, key=KEY)

    def _rows(self, path):
        with open(path) as f:
            return [json.loads(l) for l in f]

    def test_a_sync_stream_records_usage_and_first_token(self):
        path, rec = self._rec()
        cap = rec.capture(REQUEST, agent="a")
        with rec.stream(self._Sync(), cap) as st:
            for _ in st:
                pass
            st.get_final_message()
        rows = self._rows(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["usage"]["cache_creation_input_tokens"], 4700)
        self.assertIsNotNone(rows[0]["first_token_at"])
        os.unlink(path)

    def test_a_stream_that_fails_to_open_is_recorded_as_failed(self):
        path, rec = self._rec()
        cap = rec.capture(REQUEST, agent="a")
        with self.assertRaises(RuntimeError):
            with rec.stream(self._Sync(boom=True), cap):
                pass
        rows = self._rows(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], 0)
        os.unlink(path)

    def test_an_async_stream_records_the_resolved_usage(self):
        import asyncio
        path, rec = self._rec()

        async def go():
            cap = rec.capture(REQUEST, agent="a")
            async with rec.stream(self._Async(), cap) as st:
                async for _ in st:
                    pass
                await st.get_final_message()

        asyncio.run(go())
        rows = self._rows(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["usage"]["input_tokens"], 300)
        self.assertIsNotNone(rows[0]["first_token_at"])
        os.unlink(path)


class TestStreamTeardownFailureIsNotASuccess(unittest.TestCase):
    """Writing the success row before delegating to the stream's exit meant an
    SDK raising while finalising had already been recorded as a status-200 call
    with empty usage -- a degraded request filed as a successful one that
    happened to bill nothing."""

    class _BadClose:
        def __enter__(self):
            return self
        def __iter__(self):
            yield "delta"
        def __exit__(self, *a):
            raise RuntimeError("close failed")

    class _BadCloseAsync:
        async def __aenter__(self):
            return self
        async def __aiter__(self):
            yield "delta"
        async def __aexit__(self, *a):
            raise RuntimeError("close failed")

    def _rec(self):
        path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        return path, Recorder(path, key=KEY)

    def test_a_sync_teardown_failure_records_a_failure(self):
        path, rec = self._rec()
        cap = rec.capture(REQUEST, agent="a")
        with self.assertRaises(RuntimeError):
            with rec.stream(self._BadClose(), cap) as st:
                for _ in st:
                    pass
        with open(path) as f:
            rows = [json.loads(l) for l in f]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], 0)
        os.unlink(path)

    def test_an_async_teardown_failure_records_a_failure(self):
        import asyncio
        path, rec = self._rec()

        async def go():
            cap = rec.capture(REQUEST, agent="a")
            with self.assertRaises(RuntimeError):
                async with rec.stream(self._BadCloseAsync(), cap) as st:
                    async for _ in st:
                        pass

        asyncio.run(go())
        with open(path) as f:
            rows = [json.loads(l) for l in f]
        self.assertEqual(rows[0]["status"], 0)
        os.unlink(path)

    def test_a_clean_teardown_still_records_success(self):
        path, rec = self._rec()
        cap = rec.capture(REQUEST, agent="a")
        with rec.stream(self._Ok(), cap) as st:
            for _ in st:
                pass
            st.get_final_message()
        with open(path) as f:
            rows = [json.loads(l) for l in f]
        self.assertEqual(rows[0]["status"], 200)
        os.unlink(path)

    class _Ok:
        def __enter__(self):
            return self
        def __iter__(self):
            yield "delta"
        def get_final_message(self):
            return RESPONSE
        def __exit__(self, *a):
            return False


class TestFinalMessageWaitsForTeardown(unittest.TestCase):
    """A stream can hand back a complete message and still raise while closing.
    Writing the success row at get_final_message() set _done, so the later
    failure was ignored and a degraded call read as a clean 200 with usage."""

    class _FinalThenBoom:
        def __enter__(self):
            return self
        def __iter__(self):
            yield "delta"
        def get_final_message(self):
            return RESPONSE
        def __exit__(self, *a):
            raise RuntimeError("close failed")

    class _FinalThenBoomAsync:
        async def __aenter__(self):
            return self
        async def __aiter__(self):
            yield "delta"
        async def get_final_message(self):
            return RESPONSE
        async def __aexit__(self, *a):
            raise RuntimeError("close failed")

    class _Clean:
        def __enter__(self):
            return self
        def __iter__(self):
            yield "delta"
        def get_final_message(self):
            return RESPONSE
        def __exit__(self, *a):
            return False

    def _rec(self):
        path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        return path, Recorder(path, key=KEY)

    def _rows(self, path):
        with open(path) as f:
            return [json.loads(l) for l in f]

    def test_a_close_failure_after_a_final_message_records_a_failure(self):
        path, rec = self._rec()
        cap = rec.capture(REQUEST, agent="a")
        with self.assertRaises(RuntimeError):
            with rec.stream(self._FinalThenBoom(), cap) as st:
                for _ in st:
                    pass
                st.get_final_message()
        rows = self._rows(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], 0)
        os.unlink(path)

    def test_the_async_form_behaves_the_same(self):
        import asyncio
        path, rec = self._rec()

        async def go():
            cap = rec.capture(REQUEST, agent="a")
            with self.assertRaises(RuntimeError):
                async with rec.stream(self._FinalThenBoomAsync(), cap) as st:
                    async for _ in st:
                        pass
                    await st.get_final_message()

        asyncio.run(go())
        self.assertEqual(self._rows(path)[0]["status"], 0)
        os.unlink(path)

    def test_a_clean_stream_still_records_the_usage(self):
        path, rec = self._rec()
        cap = rec.capture(REQUEST, agent="a")
        with rec.stream(self._Clean(), cap) as st:
            for _ in st:
                pass
            st.get_final_message()
        rows = self._rows(path)
        self.assertEqual(rows[0]["status"], 200)
        self.assertEqual(rows[0]["usage"]["cache_creation_input_tokens"], 4700)
        os.unlink(path)


class TestPromptIsFrozenAtCapture(unittest.TestCase):
    """Agent loops append to a shared messages list between a call going out and
    coming back. Holding the request by reference and segmenting it after the
    response meant the trace hashed a prompt the provider never saw, paired with
    the original usage counters."""

    def _rec(self):
        path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        return path, Recorder(path, key=KEY)

    def _row(self, path):
        with open(path) as f:
            return json.loads(f.read())

    def test_appending_after_capture_does_not_change_the_trace(self):
        path, rec = self._rec()
        msgs = [{"role": "user", "content": "first turn"}]
        cap = rec.capture({"model": "claude-opus-5", "messages": msgs}, agent="a")
        msgs.append({"role": "assistant", "content": "reply"})
        msgs.append({"role": "user", "content": "second turn"})
        cap.done(RESPONSE)
        self.assertEqual(len(self._row(path)["segments"]), 1)
        os.unlink(path)

    def test_mutating_a_tool_after_capture_does_not_change_the_ids(self):
        path, rec = self._rec()
        tools = [{"name": "read", "description": "d" * 50}]
        cap = rec.capture({"model": "claude-opus-5", "tools": tools,
                           "messages": [{"role": "user", "content": "x"}]}, agent="a")
        before = [s["id"] for s in cap.segments]
        tools[0]["description"] = "totally different"
        cap.done(RESPONSE)
        self.assertEqual([s["id"] for s in self._row(path)["segments"]], before)
        os.unlink(path)

    def test_the_model_is_snapshotted_too(self):
        path, rec = self._rec()
        req = {"model": "claude-opus-5", "messages": [{"role": "user", "content": "x"}]}
        cap = rec.capture(req, agent="a")
        req["model"] = "claude-haiku-4-5"
        cap.done(RESPONSE)
        self.assertEqual(self._row(path)["model"], "claude-opus-5")
        os.unlink(path)


class TestCancellationIsRecorded(unittest.TestCase):
    """asyncio.CancelledError inherits BaseException, so `except Exception`
    let a cancelled or timed-out in-flight request leave no row -- dropping
    exactly the degraded traffic this recorder exists to keep."""

    def test_a_cancelled_request_records_a_failure(self):
        import asyncio
        path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        rec = Recorder(path, key=KEY)

        async def go():
            async def slow(**kw):
                await asyncio.sleep(10)
                return RESPONSE
            task = asyncio.ensure_future(
                rec.messages_create(slow, **{"model": "claude-opus-5", "messages": []}))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(go())
        with open(path) as f:
            rows = [json.loads(l) for l in f]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], 0)
        os.unlink(path)

    def test_a_normal_async_success_is_unaffected(self):
        import asyncio
        path = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        rec = Recorder(path, key=KEY)

        async def go():
            async def ok(**kw):
                return RESPONSE
            await rec.messages_create(ok, **{"model": "claude-opus-5", "messages": []})

        asyncio.run(go())
        with open(path) as f:
            self.assertEqual(json.loads(f.read())["status"], 200)
        os.unlink(path)


class TestACancelledStreamIsStillRecorded(unittest.IsolatedAsyncioTestCase):
    """`asyncio.CancelledError` inherits from BaseException. A stream cancelled
    while opening, or while closing normally, exited without touching failed()
    or done() and vanished from the trace -- biasing coverage and spend toward
    calls that succeeded, on the path agent workloads use most.

    The non-streaming async path was fixed for exactly this. The streaming one
    was not."""

    class _Inner:
        def __init__(self, where):
            self.where = where

        async def __aenter__(self):
            if self.where == "enter":
                raise asyncio.CancelledError()
            return self

        async def __aexit__(self, *a):
            if self.where == "exit":
                raise asyncio.CancelledError()
            return False

    def _proxy(self, where):
        from cacheeconomics.recorder import _StreamProxy

        class _Cap:
            def __init__(self):
                self.failed_calls = 0
                self._done = False

            def failed(self, *a, **k):
                self.failed_calls += 1
                self._done = True

            def done(self, *a, **k):
                self._done = True

            def first_token(self):
                pass
        cap = _Cap()
        return _StreamProxy(self._Inner(where), cap), cap

    async def test_cancellation_while_opening_is_recorded(self):
        proxy, cap = self._proxy("enter")
        with self.assertRaises(asyncio.CancelledError):
            await proxy.__aenter__()
        self.assertEqual(cap.failed_calls, 1)

    async def test_cancellation_while_closing_is_recorded(self):
        proxy, cap = self._proxy("exit")
        await proxy.__aenter__()
        with self.assertRaises(asyncio.CancelledError):
            await proxy.__aexit__(None, None, None)
        self.assertEqual(cap.failed_calls, 1)


class TestTheFinalMessageIsGuarded(unittest.IsolatedAsyncioTestCase):
    """Outside a context manager, resolving the final message is the only place
    a stream is finalised. A failure there was the whole record of the call and
    it escaped without one, so a degraded streamed request was omitted rather
    than recorded -- biasing coverage and spend toward successful streams
    exactly under timeout and load."""

    class _Cap:
        def __init__(self):
            self.failed_calls = 0
            self._done = False

        def failed(self, *a, **k):
            self.failed_calls += 1
            self._done = True

        def done(self, *a, **k):
            self._done = True

        def first_token(self):
            pass

    def _proxy(self, stream):
        """`_stream` is set by __enter__, not the constructor. Leaving it None
        made an earlier check pass for the wrong reason: the AttributeError on
        None is itself a BaseException, so `failed()` fired without the guard
        under test ever running."""
        from cacheeconomics.recorder import _StreamProxy
        cap = self._Cap()
        proxy = _StreamProxy(stream, cap)
        proxy._stream = stream
        return proxy, cap

    def test_a_synchronous_failure_is_recorded(self):
        class Boom:
            def get_final_message(self):
                raise RuntimeError("SDK gave up")
        proxy, cap = self._proxy(Boom())
        with self.assertRaises(RuntimeError):
            proxy.get_final_message()
        self.assertEqual(cap.failed_calls, 1)

    def test_an_interrupt_is_recorded_too(self):
        """KeyboardInterrupt and SystemExit are not Exceptions either."""
        class Boom:
            def get_final_message(self):
                raise KeyboardInterrupt()
        proxy, cap = self._proxy(Boom())
        with self.assertRaises(KeyboardInterrupt):
            proxy.get_final_message()
        self.assertEqual(cap.failed_calls, 1)

    async def test_a_cancelled_await_is_recorded(self):
        class Cancelled:
            def get_final_message(self):
                async def go():
                    raise asyncio.CancelledError()
                return go()
        proxy, cap = self._proxy(Cancelled())
        with self.assertRaises(asyncio.CancelledError):
            await proxy.get_final_message()
        self.assertEqual(cap.failed_calls, 1)

    async def test_it_is_not_recorded_twice(self):
        class Cancelled:
            def get_final_message(self):
                async def go():
                    raise asyncio.CancelledError()
                return go()
        proxy, cap = self._proxy(Cancelled())
        cap.failed()                       # already finalised by teardown
        with self.assertRaises(asyncio.CancelledError):
            await proxy.get_final_message()
        self.assertEqual(cap.failed_calls, 1)


if __name__ == "__main__":
    unittest.main()
