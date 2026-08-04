"""Capture real requests as an instrumented trace, without capturing prompts.

This is the instrument for the HIGH-confidence tier. Everything analysed so far
has been either synthetic with the failures planted by hand, or usage-only with
no structure at all -- which means no counterfactual, which means no bake-off.
A recorder is what lets a real workload answer the question the tool exists to
ask.

Three things it must get right, because each has a failure mode that quietly
invalidates every number downstream.

**Prompt text never leaves.** Segments are hashed at source with an HMAC key the
customer holds. A bare SHA-256 is guessable by dictionary over short segments --
a status label, a product name, a policy line -- and the key removes that,
while keeping the only property needed: equal content yields equal ids within
one analysis.

Keying does not hide *equality*, and this docstring used to claim it did. One
shared key means identical content produces an identical id whoever sent it, so
on a multi-tenant recorder a reader could join ids and learn two tenants share a
policy block without ever holding the key. The tenant is therefore part of
segment identity: ids are scoped, not merely keyed.

**Streaming is not optional.** Agentic workloads stream almost everything. A
recorder that quietly skips streamed calls reports a coverage number that looks
fine and an analysis that missed the traffic that mattered. Sync, async and
streamed calls all record here, and a call that fails records too, because a
change that makes an agent fail faster is not a saving.

**Token counts are estimated, and say so.** There is no tokenizer here, and
adding one would break the pure-Python promise that lets the same code run in
the browser. Segment sizes are estimated, then scaled so they sum to the input
total the provider actually billed. That makes each segment a proportional share
of a measured quantity rather than a guess, and `tokens_are_estimated` is
recorded on every row so nothing downstream can mistake one for the other.
"""

from __future__ import annotations

import inspect
import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .segment import (_billed_input, _get, _requested_ttl, _scale_to_measured,
                      segments_from_request, usage_from_response)
# The one spelling of "no surface was stated", shared with the loaders so a row
# this recorder writes and a row the JSONL loader defaults agree by construction.
from .trace import UNATTRIBUTED

SCHEMA_VERSION = 1


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Capture:
    """One in-flight request. Records whatever happened, including failure."""
    recorder: "Recorder"
    agent: str
    session: str | None
    tenant: str | None
    target_id: str
    sent_at: datetime = field(default_factory=_now)
    first_token_at: datetime | None = None
    # Snapshotted at capture time so later mutation of the caller's dict cannot
    # reach them.
    segments: list = field(default_factory=list)
    model: str = "unknown"
    ttl_requested: str | None = None
    _done: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def first_token(self) -> None:
        """Idempotent: call on every stream event, the first one wins.

        first_token_at is the tightest lower bound most traces carry on when a
        cache entry became readable, which the simulator needs to avoid letting
        concurrent requests hit an entry that did not exist yet.
        """
        if self.first_token_at is None:
            self.first_token_at = _now()

    def done(self, response, status: int = 200) -> None:
        """Marked complete only once the row is durably on disk.

        Setting the flag first meant a failure to open, write, flush or fsync
        left the capture permanently "done" with nothing appended -- and a
        retry then returned silently, so a successful, billed call vanished at
        exactly the durability boundary the fsync exists to defend. Worse, a
        caller retrying the API after a recorder exception would double the
        provider spend while the trace showed only the retry.
        """
        with self._lock:
            if self._done:
                return
            self.recorder._write(self, response, status)
            self._done = True

    def failed(self, status: int = 0) -> None:
        """A call that never returned still happened and still cost latency.

        Dropping it would quietly improve every ratio, and a change that makes
        an agent fail faster would look like a saving.
        """
        self.done(None, status=status)


class Recorder:
    """Appends instrumented trace rows to a JSONL file.

    Thread-safe because agent workloads fan out. The file is opened per write
    rather than held, so a crashed process leaves a valid partial trace instead
    of an empty buffer.
    """

    def __init__(self, path: str, key: bytes, *, tenant: str | None = None,
                 target_id: str = UNATTRIBUTED):
        """`target_id` is the surface these requests are actually sent to.

        It is stamped on every row and decides, months later and in another
        process, which rate table the trace is priced against. It defaulted to
        `anthropic/direct`, so a recorder wired into a proxy fronting Bedrock
        wrote first-party attribution onto traffic AWS invoices -- and nothing
        downstream could tell that surface from one somebody had chosen.

        `UNATTRIBUTED` is registered unpriceable, so an unnamed surface produces
        a report that refuses to publish dollars and says why, rather than one
        that publishes the wrong ones. That is the same value the JSONL loader
        already uses for a row that states no provider (`load_jsonl`'s
        `default_target`), which is what a recorder writing no surface produces.
        """
        if not key:
            raise ValueError(
                "a recorder needs an HMAC key. Segment ids are keyed so that a short, "
                "low-entropy segment cannot be recovered by dictionary attack. Generate "
                "one per customer with os.urandom(32) and keep it with them. Set "
                "`tenant` too on a shared recorder: the key does not scope ids across "
                "tenants, the tenant does.")
        self.path = path
        self.key = key
        self.tenant = tenant
        self.target_id = target_id
        self._lock = threading.Lock()
        self._n = 0
        # A previous process killed between `write` and `flush` leaves the last
        # row without its newline. Appending onto that joins this run's first
        # row to the broken one, so ingest drops a single unparseable line and
        # loses both. Each row is fsynced, so the window is small -- but this
        # appends across restarts, which is exactly when it is not zero.
        #
        # Terminated rather than refused: the half-row is already unrecoverable,
        # and a newline costs nothing while keeping every row of this run.
        try:
            if os.path.exists(path) and os.path.getsize(path):
                with open(path, "rb") as f:
                    f.seek(-1, 2)
                    if f.read(1) != b"\n":
                        with open(path, "a") as fix:
                            fix.write("\n")
        except OSError:
            pass

    def capture(self, request: dict, *, agent: str = "unknown",
                session: str | None = None) -> Capture:
        """Segment the prompt now, while it is still the prompt that was sent.

        The request dict was previously held by reference and segmented after
        the response arrived. Agent loops append to a shared `messages` list
        between the call going out and coming back, so the trace hashed a
        prompt the provider never saw and paired it with the original usage
        counters -- corrupting volatility, marker placement and the bake-off in
        a way no downstream check could detect. Measured: a request sent with
        one message recorded three.

        Only the byte sizes are frozen here; scaling them to the billed input
        total still waits for the response, which is the one part that legitimately
        arrives later.
        """
        # The request itself is never stored. Everything wanted from it is
        # extracted here and the dict is released: this module's whole claim is
        # that prompt text becomes keyed segment metadata and goes no further,
        # and a capture holding the original defeated that for as long as
        # anything held the capture -- including printing one, since a dataclass
        # repr would render the entire prompt.
        cap = Capture(self, agent, session, self.tenant, self.target_id)
        cap.segments = segments_from_request(request, self.key, self.tenant)
        cap.model = request.get("model", "unknown")
        cap.ttl_requested = _requested_ttl(request)
        return cap

    def _write(self, cap: Capture, response, status: int) -> None:
        usage = usage_from_response(response) if response is not None else {}
        segments = [dict(sg) for sg in cap.segments]
        _scale_to_measured(segments, _billed_input(usage) if usage else None)
        row = {
            "schema_version": SCHEMA_VERSION,
            "request_id": (_get(response, "id") or f"{cap.agent}-{cap.sent_at.isoformat()}"),
            "sent_at": cap.sent_at.isoformat(),
            "first_token_at": cap.first_token_at.isoformat() if cap.first_token_at else None,
            "model": cap.model,
            "agent": cap.agent,
            "session": cap.session,
            "tenant": cap.tenant,
            "target_id": cap.target_id,
            "status": status,
            "ttl_requested": cap.ttl_requested,
            "usage": usage,
            "segments": segments,
            # Stated on every row so nothing downstream can mistake a
            # proportional estimate for a counted quantity.
            "tokens_are_estimated": True,
        }
        line = json.dumps(row, separators=(",", ":")) + "\n"
        with self._lock:
            with open(self.path, "a") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            self._n += 1

    @property
    def recorded(self) -> int:
        return self._n

    # --- wrappers ---------------------------------------------------------

    def messages_create(self, create_fn, **request):
        """Wrap a non-streaming `messages.create`, sync or async.

        An earlier version recorded the coroutine itself as the response, so an
        async call was written out with empty usage and status 200 before the
        request had even been sent -- and a later failure was never captured.
        Async workloads would have appeared as a wall of successful requests
        that happened to report no tokens.
        """
        cap = self.capture(request, agent=request.pop("_agent", "unknown"),
                           session=request.pop("_session", None))
        try:
            resp = create_fn(**request)
        except BaseException:
            # The same reasoning as every other guard on this path: an
            # exception escaping means the call did not succeed, and the ones
            # that are not Exceptions -- KeyboardInterrupt on a long sync call,
            # SystemExit during shutdown -- are exactly when a request is most
            # likely to be abandoned mid-flight. Recording it costs nothing;
            # missing it drops degraded traffic and flatters every ratio.
            cap.failed()
            raise
        if inspect.isawaitable(resp):
            return self._await_and_record(resp, cap)
        cap.first_token()
        cap.done(resp)
        return resp

    async def _await_and_record(self, awaitable, cap: "Capture"):
        """`finally`, not `except Exception`.

        asyncio.CancelledError inherits BaseException, so a cancelled or
        timed-out in-flight request left no row at all -- dropping exactly the
        degraded traffic this recorder exists to keep, and biasing every ratio
        toward calls that succeeded. Measured: a cancelled request recorded
        nothing.
        """
        try:
            resp = await awaitable
        except BaseException:
            cap.failed()
            raise
        cap.first_token()
        cap.done(resp)
        return resp

    def stream(self, stream_ctx, cap: Capture):
        """Wrap a streaming call so first-token time and the final usage land.

        Usage arrives on the final message, not on the deltas, so the caller
        still has to hand back `get_final_message()`. What this guarantees is
        that a stream which raises partway is recorded rather than dropped.
        """
        return _StreamProxy(stream_ctx, cap)


class _StreamProxy:
    """Wraps a streaming call so first-token time and the final usage land.

    Both the sync and async forms, and both the setup and teardown failures.
    An earlier version implemented only `__enter__`/`__iter__`/`__exit__`, so an
    `async with` stream could not be wrapped at all and async streamed traffic
    bypassed the recorder entirely -- while a failure inside `__enter__` never
    reached `__exit__`, so a stream that died during setup was never recorded.
    Those are the high-volume and the degraded paths respectively, which are
    exactly the two this recorder exists to not lose.
    """

    def __init__(self, inner, cap: "Capture"):
        self._inner = inner
        self._cap = cap
        self._stream = None
        # Set while inside the context manager. The final message is buffered
        # rather than written there, because a stream can hand back a complete
        # message and still raise while closing -- and once the success row was
        # written, `_done` was set and the later failure was silently ignored.
        # A degraded streamed call then read as a clean 200 with usage.
        self._in_context = False
        self._final = None
        self._have_final = False

    # Every entry and exit below catches BaseException, not Exception.
    # `asyncio.CancelledError` inherits from BaseException, so a stream
    # cancelled while opening or while closing normally exited without touching
    # `failed()` or `done()` and vanished from the trace entirely -- biasing
    # coverage and spend toward calls that succeeded, on the streaming path
    # agent workloads use most. The non-streaming async path was fixed for this
    # and the streaming one was not.

    # --- sync ---------------------------------------------------------
    def __enter__(self):
        try:
            self._stream = self._inner.__enter__()
        except BaseException:
            self._cap.failed()
            raise
        self._in_context = True
        return self

    def __iter__(self):
        for event in self._stream:
            self._cap.first_token()
            yield event

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self._cap.failed()
            return self._inner.__exit__(exc_type, exc, tb)
        # Close first, then decide. Writing the success row before delegating
        # meant an SDK that raised while finalising the stream had already been
        # recorded as a status-200 call with empty usage -- a degraded request
        # filed as a successful one that happened to bill nothing, on the path
        # agent workloads use most.
        try:
            suppressed = self._inner.__exit__(exc_type, exc, tb)
        except BaseException:
            self._cap.failed()
            raise
        self._in_context = False
        if not self._cap._done:
            self._cap.done(self._final if self._have_final else None)
        return suppressed

    # --- async --------------------------------------------------------
    async def __aenter__(self):
        try:
            self._stream = await self._inner.__aenter__()
        except BaseException:
            self._cap.failed()
            raise
        self._in_context = True
        return self

    async def __aiter__(self):
        async for event in self._stream:
            self._cap.first_token()
            yield event

    async def __aexit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self._cap.failed()
            return await self._inner.__aexit__(exc_type, exc, tb)
        try:
            suppressed = await self._inner.__aexit__(exc_type, exc, tb)
        except BaseException:
            self._cap.failed()
            raise
        self._in_context = False
        if not self._cap._done:
            self._cap.done(self._final if self._have_final else None)
        return suppressed

    # --- shared -------------------------------------------------------
    def get_final_message(self):
        """Usage arrives on the final message, not on the deltas.

        Returns the awaitable untouched when the underlying stream is async, so
        the caller awaits it and the row is written from the resolved message
        rather than from a coroutine.
        """
        # Guarded, and with BaseException. Outside a context manager this is
        # the only place a stream is finalised, so a failure here -- a timeout,
        # a cancellation, an SDK raising while assembling the message -- was the
        # whole record of the call, and it escaped without one. The result is a
        # degraded streamed request omitted rather than recorded, biasing
        # coverage and spend toward successful streams precisely under load.
        try:
            msg = self._stream.get_final_message()
        except BaseException:
            if not self._cap._done:
                self._cap.failed()
            raise
        if inspect.isawaitable(msg):
            return self._await_final(msg)
        self._record_final(msg)
        return msg

    def _record_final(self, msg) -> None:
        """Hold the message until teardown proves the call actually finished.

        Writing here meant a stream that returned a complete message and then
        raised on close was already recorded as a success, and the failure could
        not correct it. Outside a context manager there is no teardown to wait
        for, so it is written immediately.
        """
        self._final, self._have_final = msg, True
        if not self._in_context:
            self._cap.done(msg)

    async def _await_final(self, awaitable):
        try:
            msg = await awaitable
        except BaseException:
            # asyncio.CancelledError is not an Exception. Two other guards on
            # this class were widened for exactly that and this one was missed.
            if not self._cap._done:
                self._cap.failed()
            raise
        self._record_final(msg)
        return msg
