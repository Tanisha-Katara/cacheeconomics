"""The scripts that leave the machine, and the bugs they have already had.

`tier-b/` is not part of the installed package, so none of the 1,026 tests
touched it. That is where every script that opens a socket lives, and in one
afternoon it produced five defects: a dry run that wrote a file the analyzer
read as authoritative, a dry run that under-reported its own egress, a proxy
that forwarded gzip as text, a proxy with no session identity that manufactured
a false finding, and two hard-coded hostnames that locked out the clients most
likely to care.

All five were fixed and none had a test, which is the same as not being fixed.

Network is never touched here. The counter takes an injected `count` callable
and the proxy is exercised against a stub upstream on loopback, so these run in
CI with no key and no egress.
"""

import contextlib
import hashlib
import http.client
import http.server
import io
import signal
import importlib.util
import json
import os
import socket
import time
import subprocess
import sys
import tempfile
import types
import threading
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TIER_B = os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tier-b"))


@contextlib.contextmanager
def _deadline(seconds):
    """Fail rather than hang. Two of the defects below are infinite loops, and a
    test that hangs the suite reads as CI being slow."""
    def bail(*a):
        raise TimeoutError(f"still running after {seconds}s")
    old = signal.signal(signal.SIGALRM, bail)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def load(name):
    """Import a tier-b script by path; it is not on any import path."""
    path = os.path.join(TIER_B, f"{name}.py")
    if not os.path.exists(path):
        raise unittest.SkipTest(f"{path} not present")
    spec = importlib.util.spec_from_file_location(f"_tierb_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BODY = {"model": "claude-haiku-4-5",
        "system": [{"type": "text", "text": "policy " * 200}],
        "messages": [{"role": "user", "content": "hello"}]}


def _row(i=0):
    return {"request_id": f"r{i}", "sent_at": "2026-08-01T09:00:00+00:00",
            "request": BODY,
            "response": {"usage": {"input_tokens": 1000,
                                   "cache_read_input_tokens": 0,
                                   "cache_creation_input_tokens": 0}}}


class TestADryRunLeavesNothingBehind(unittest.TestCase):
    """It wrote its output file with every count zero, and the loader read that
    as *counted*: `tokens_are_counted: True` on a file that had never spoken to
    a tokenizer, with a whole prompt's mass collapsed onto one segment.

    A file that looks authoritative and is not is the failure this toolchain
    exists to refuse, and this one was manufactured by the flag added to make
    the tool easier to trust.
    """

    def _run(self, *extra):
        src = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        src.write(json.dumps(_row()) + "\n")
        src.close()
        self.addCleanup(os.unlink, src.name)
        out = src.name.replace(".jsonl", "-out.jsonl")
        r = subprocess.run(
            [sys.executable, os.path.join(TIER_B, "count_tokens.py"),
             src.name, "-o", out, "--dry-run", *extra],
            capture_output=True, text=True,
            env=dict(os.environ, ANTHROPIC_API_KEY=""))
        return out, r

    def test_it_writes_no_output_file(self):
        out, r = self._run()
        self.addCleanup(lambda: os.path.exists(out) and os.unlink(out))
        self.assertFalse(os.path.exists(out),
                         "a dry run produced a file the analyzer would read")

    def test_it_needs_no_api_key(self):
        """It sends nothing, so demanding a key to find out what it would send
        makes the safe option the harder one."""
        _, r = self._run()
        self.assertEqual(r.returncode, 0, r.stderr[-300:])

    def test_it_names_the_host_it_would_send_to(self):
        _, r = self._run()
        self.assertIn("api.anthropic.com", r.stdout + r.stderr)

    def test_a_custom_endpoint_is_the_host_it_names(self):
        _, r = self._run("--endpoint", "https://gateway.internal/count")
        blob = r.stdout + r.stderr
        self.assertIn("gateway.internal", blob)


class TestTheCounterIsExactAndCached(unittest.TestCase):
    """`count_segments` takes differences of in-context prefix counts. The
    caching is the only reason it is affordable, so it is asserted rather than
    assumed."""

    def test_a_shared_prefix_is_counted_once(self):
        tok = load("../harness/cacheeconomics/tokenizer") if False else None
        from cacheeconomics.tokenizer import count_segments
        calls = []

        def fake(body):
            calls.append(1)
            return 10 * len(json.dumps(body))

        cache = {}
        count_segments(BODY, fake, cache)
        first = len(calls)
        count_segments(BODY, fake, cache)          # identical body
        self.assertEqual(len(calls), first,
                         "the second pass re-counted a prefix it had already seen")

    def test_counts_are_never_negative(self):
        """A boundary can measure under its predecessor when the tokenizer
        merges across it, and a negative token count is refused at pricing --
        which would drop the whole request rather than the one segment."""
        from cacheeconomics.tokenizer import count_segments
        shrinking = iter([100, 90, 80, 70, 60, 50, 40])
        got = count_segments(BODY, lambda b: next(shrinking, 0), {})
        self.assertTrue(all(n >= 0 for n in got), got)


class _Stub(http.server.BaseHTTPRequestHandler):
    # Counted so a test can prove the request actually arrived here.
    hits = 0

    """An upstream that answers like the provider, and gzips if asked."""

    def log_message(self, *a):
        pass

    def do_POST(self):
        import gzip
        n = int(self.headers.get("content-length") or 0)
        self.rfile.read(n)
        payload = json.dumps({"id": "msg_stub", "content": [],
                              "usage": {"input_tokens": 7,
                                        "cache_read_input_tokens": 0,
                                        "cache_creation_input_tokens": 0}}).encode()
        type(self).hits += 1
        wants_gzip = "gzip" in (self.headers.get("accept-encoding") or "")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        if wants_gzip:
            payload = gzip.compress(payload)
            self.send_header("content-encoding", "gzip")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class _ProxyCase(unittest.TestCase):
    """A live proxy in front of a loopback stub, for cases that need to send a
    real request through it. Shared because both cases below do."""

    BODY = {"model": "claude-opus-5", "max_tokens": 8,
            "system": [{"type": "text", "text": "you are precise",
                        "cache_control": {"type": "ephemeral"}}],
            "tools": [{"name": "t", "description": "d",
                       "input_schema": {"type": "object"}}],
            "messages": [{"role": "user", "content": "hi"}]}

    @staticmethod
    def _free_port():
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def setUp(self):
        _Stub.hits = 0
        self.stub = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
        self.port = self.stub.server_address[1]
        threading.Thread(target=self.stub.serve_forever, daemon=True).start()
        self.addCleanup(self.stub.shutdown)

        # Reserve the *name*, not the file. The proxy creates its output
        # exclusively now, so leaving an empty file here makes it refuse to
        # start -- which is exactly what happens to an operator who runs
        # `touch run.jsonl` first, and is worth the harness matching rather
        # than working around.
        self.out = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        self.out.close()
        os.unlink(self.out.name)
        self.addCleanup(lambda: os.path.exists(self.out.name) and os.unlink(self.out.name))

        # A known port. This passed `--port 0`, so the proxy chose one the test
        # could never discover -- which is exactly why nothing was ever sent
        # through it and every assertion in this class read the source instead.
        self.proxy_port = self._free_port()
        self.proxy = subprocess.Popen(
            [sys.executable, os.path.join(TIER_B, "capture_proxy.py"),
             "--out", self.out.name, "--port", str(self.proxy_port),
             "--upstream", f"http://127.0.0.1:{self.port}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            env=dict(os.environ, ANTHROPIC_API_KEY="test"))
        self.addCleanup(self.proxy.terminate)
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", self.proxy_port), 0.1):
                    break
            except OSError:
                time.sleep(0.05)
        else:
            # Not a skip. A proxy that crashes on import, exits before binding,
            # or binds the wrong interface used to skip six tests and leave CI
            # green -- the entire live capture path unguarded by the suite that
            # exists to guard it.
            rc = self.proxy.poll()
            err = ""
            if rc is not None:
                try:
                    err = (self.proxy.stderr.read() or b"").decode()[-600:]
                except Exception:                              # noqa: BLE001
                    err = "(stderr unavailable)"
            self.fail(f"capture_proxy did not accept a connection on "
                      f"127.0.0.1:{self.proxy_port}. exit={rc}\n{err}")

    def _post(self, accept_gzip=True):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.proxy_port}/v1/messages",
            data=json.dumps(self.BODY).encode(),
            headers={"content-type": "application/json",
                     "accept-encoding": "gzip" if accept_gzip else "identity"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.read()


class TestTheProxyForwardsSomethingReadable(_ProxyCase):
    """It stripped `content-encoding` and forwarded the still-compressed body,
    so the client decoded gzip as text. Measured as "'utf-8' codec can't decode
    byte 0x8b" -- the gzip magic number -- six steps into a real agent run.
    """

    def test_a_request_reaches_the_configured_upstream(self):
        """It hard-coded api.anthropic.com, which locks out every client behind
        a gateway -- the ones most likely to care about egress at all.

        Driven end to end. Asserting `--upstream` appears in the source passes
        on dead code that still forwards to the default host.
        """
        self._post()
        self.assertEqual(_Stub.hits, 1,
                         "the request never reached the configured upstream")

    def test_the_client_gets_a_body_it_can_decode(self):
        """The proxy stripped `content-encoding` and forwarded the still-gzipped
        bytes, so the client decoded gzip as text. Measured as "\'utf-8\' codec
        can\'t decode byte 0x8b" -- the gzip magic number -- six steps into a
        real agent run."""
        raw = self._post(accept_gzip=True)
        self.assertNotEqual(raw[:2], b"\x1f\x8b", "forwarded a compressed body")
        json.loads(raw.decode())          # must not raise


class TestEveryCapturedRowCarriesASession(_ProxyCase):
    """Without one, every request lands in one reuse chain. On a real capture
    that made three different *kinds* of request -- an agent loop beside a
    judgement call -- read as one conversation whose tools kept changing, and
    VOL-1 fired confidently on a prefix that had never drifted."""

    def test_the_key_is_derived_from_the_stable_prefix(self):
        """Read off a captured row, not off the source. Grepping for the string
        "session" passes on a constant, and a constant session is precisely the
        bug: every request lands in one reuse chain."""
        self._post()
        other = dict(self.BODY, tools=[{"name": "z", "description": "d",
                                        "input_schema": {"type": "object"}}])
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.proxy_port}/v1/messages",
            data=json.dumps(other).encode(),
            headers={"content-type": "application/json"})
        urllib.request.urlopen(req, timeout=5).read()
        rows = [json.loads(l) for l in open(self.out.name) if l.strip()]
        self.assertEqual(len(rows), 2, "the proxy did not capture both requests")
        keys = [r.get("session") for r in rows]
        self.assertTrue(all(keys), "a captured row carries no session")
        self.assertNotEqual(keys[0], keys[1],
                            "two different tool sets share a session key, so "
                            "every request lands in one reuse chain")



class _CountStub(http.server.BaseHTTPRequestHandler):
    """A token-count endpoint, so the default path can be exercised offline."""

    hits = 0

    def log_message(self, *a):
        pass

    def do_POST(self):
        type(self).hits += 1
        body = json.loads(self.rfile.read(
            int(self.headers.get("content-length") or 0)) or b"{}")
        # Proportional to content so the deltas count_segments takes are not
        # all zero -- a stub returning a constant would let a broken
        # differencing step pass.
        n = len(json.dumps(body, default=str)) // 4
        payload = json.dumps({"input_tokens": n}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class TestTheDefaultPathCounts(unittest.TestCase):
    """Counting was correct and optional, and optional meant skipped.

    The earlier version of this class asserted that `--estimate-only` appeared
    in `--help` and `--count-tokens` did not. That is help text, not behaviour:
    making estimate-only the default, so counting never runs at all, passed it.
    Verified by mutation before rewriting.

    These drive `run_diagnostic.py` against a local stub endpoint, so the
    default path actually executes and its output can be read.
    """

    def setUp(self):
        _CountStub.hits = 0
        self.stub = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _CountStub)
        self.port = self.stub.server_address[1]
        threading.Thread(target=self.stub.serve_forever, daemon=True).start()
        self.addCleanup(self.stub.shutdown)
        self.tmp = tempfile.mkdtemp()
        self.src = os.path.join(self.tmp, "bodies.jsonl")
        body = {"model": "claude-opus-5",
                "system": [{"type": "text", "text": "s" * 4000,
                            "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": "hello"}]}
        with open(self.src, "w") as f:
            for i in range(4):
                f.write(json.dumps({
                    "sent_at": f"2026-07-29T09:0{i}:00Z", "body": body,
                    "usage": {"input_tokens": 100,
                              "cache_read_input_tokens": 0,
                              "cache_creation_input_tokens": 1000,
                              "cache_creation": {
                                  "ephemeral_5m_input_tokens": 1000,
                                  "ephemeral_1h_input_tokens": 0}}}) + "\n")

    def _run(self, *extra):
        return subprocess.run(
            [sys.executable, os.path.join(TIER_B, "run_diagnostic.py"), self.src,
             "--endpoint", f"http://127.0.0.1:{self.port}/v1/messages/count_tokens",
             "--allow-unreconciled", *extra],
            capture_output=True, text=True,
            env=dict(os.environ, ANTHROPIC_API_KEY="test",
                     CACHEECONOMICS_HMAC_KEY="k" * 32))

    def test_counting_happens_without_being_asked_for(self):
        r = self._run()
        self.assertGreater(_CountStub.hits, 0,
                           f"the default path never counted.\n{r.stderr[-500:]}")

    def test_estimate_only_actually_skips_it(self):
        r = self._run("--estimate-only")
        self.assertEqual(_CountStub.hits, 0,
                         f"--estimate-only still counted.\n{r.stderr[-500:]}")
        self.assertIn("counting skipped", r.stderr)

    def test_the_counted_output_carries_segment_tokens(self):
        """The point of counting: sizes the analyzer can attach money to."""
        self._run()
        out = self.src.replace(".jsonl", "-counted.jsonl")
        self.assertTrue(os.path.exists(out), "no counted export was written")
        rows = [json.loads(l) for l in open(out) if l.strip()]
        self.assertTrue(rows)
        self.assertTrue(any(r.get("segment_tokens") for r in rows),
                        "counted export carries no segment_tokens")


if __name__ == "__main__":
    unittest.main()


class TestCountingCannotFakeAnExactResult(unittest.TestCase):
    """The counted path is what releases structural dollar figures.

    A file that looks counted and is not is the exact failure this toolchain
    exists to refuse, and there were two ways to produce one.
    """

    def _script(self):
        return os.path.join(TIER_B, "count_tokens.py")

    def _bodies(self, tmp, n):
        p = os.path.join(tmp, "b.jsonl")
        body = {"model": "claude-opus-5",
                "system": [{"type": "text", "text": "x" * 200}],
                "messages": [{"role": "user", "content": "hi"}]}
        with open(p, "w") as f:
            for i in range(n):
                b = json.loads(json.dumps(body))
                b["system"][0]["text"] = f"x{i}" * 200
                f.write(json.dumps({"sent_at": "2026-07-29T09:00:00Z",
                                    "body": b, "usage": {"input_tokens": 500}}) + "\n")
        return p

    def test_a_dry_run_writes_no_cache(self):
        """The checkpoint fired before the dry-run guard, so a dry run over 25+
        rows wrote real prefix keys mapped to zero counts. A later real run
        resumes from those and emits segment_tokens that never saw a tokenizer
        -- while the dry run printed "Nothing was sent and nothing was written."
        """
        with tempfile.TemporaryDirectory() as tmp:
            src = self._bodies(tmp, 60)
            out = os.path.join(tmp, "counted.jsonl")
            r = subprocess.run(
                [sys.executable, self._script(), src, "-o", out, "--dry-run"],
                capture_output=True, text=True,
                env=dict(os.environ, ANTHROPIC_API_KEY="test"))
            self.assertEqual(r.returncode, 0, r.stderr[-400:])
            self.assertFalse(os.path.exists(out), "a dry run wrote its output")
            self.assertFalse(os.path.exists(out + ".cache.json"),
                             "a dry run wrote a cache of zero counts")

    def test_the_flag_exists_and_is_documented(self):
        """`--allow-partial` is the opt-out for a mixed counted/estimated file.
        Without it a failed row now exits non-zero, because run_diagnostic.py
        reads only the exit code."""
        r = subprocess.run([sys.executable, self._script(), "--help"],
                           capture_output=True, text=True)
        self.assertIn("--allow-partial", r.stdout)
        self.assertIn("estimates them", r.stdout)


class _StreamStub(http.server.BaseHTTPRequestHandler):
    """An upstream that answers with an event stream, like a streaming model."""

    hits = 0
    seen_body = b""

    def log_message(self, *a):
        pass

    def do_POST(self):
        type(self).hits += 1
        if "chunked" in (self.headers.get("transfer-encoding") or "").lower():
            buf = []
            while True:
                line = self.rfile.readline().strip()
                if not line:
                    continue
                n = int(line.split(b";")[0], 16)
                if n == 0:
                    self.rfile.readline()
                    break
                buf.append(self.rfile.read(n))
                self.rfile.readline()
            type(self).seen_body = b"".join(buf)
        else:
            type(self).seen_body = self.rfile.read(
                int(self.headers.get("content-length") or 0))
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        for i in range(3):
            piece = b"data: {\"i\": %d}\n\n" % i
            self.wfile.write(b"%X\r\n%s\r\n" % (len(piece), piece))
            self.wfile.flush()
            time.sleep(0.4)               # so buffering is measurable
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


class TestTheProxyDoesNotReframeLiveTraffic(_ProxyCase):
    """It claimed to forward byte for byte and did not, on two ordinary shapes.

    A chunked upload has no `content-length`, so the body read as zero bytes and
    was forwarded empty -- the provider answered a request the caller never
    made, and the capture recorded that as what was sent. And every response was
    buffered whole, so an event stream was held until the model finished, or hit
    the 300s timeout and became a synthetic 502.
    """

    def setUp(self):
        _StreamStub.hits = 0
        _StreamStub.seen_body = b""
        self.stub = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _StreamStub)
        self.port = self.stub.server_address[1]
        threading.Thread(target=self.stub.serve_forever, daemon=True).start()
        self.addCleanup(self.stub.shutdown)
        # Reserve the *name*, not the file. The proxy creates its output
        # exclusively now, so leaving an empty file here makes it refuse to
        # start -- which is exactly what happens to an operator who runs
        # `touch run.jsonl` first, and is worth the harness matching rather
        # than working around.
        self.out = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        self.out.close()
        os.unlink(self.out.name)
        self.addCleanup(lambda: os.path.exists(self.out.name) and os.unlink(self.out.name))
        self.proxy_port = self._free_port()
        self.proxy = subprocess.Popen(
            [sys.executable, os.path.join(TIER_B, "capture_proxy.py"),
             "--out", self.out.name, "--port", str(self.proxy_port),
             "--upstream", f"http://127.0.0.1:{self.port}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            env=dict(os.environ, ANTHROPIC_API_KEY="test"))
        self.addCleanup(self.proxy.terminate)
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", self.proxy_port), 0.1):
                    break
            except OSError:
                time.sleep(0.05)
        else:
            # Not a skip. A proxy that crashes on import, exits before binding,
            # or binds the wrong interface used to skip six tests and leave CI
            # green -- the entire live capture path unguarded by the suite that
            # exists to guard it.
            rc = self.proxy.poll()
            err = ""
            if rc is not None:
                try:
                    err = (self.proxy.stderr.read() or b"").decode()[-600:]
                except Exception:                              # noqa: BLE001
                    err = "(stderr unavailable)"
            self.fail(f"capture_proxy did not accept a connection on "
                      f"127.0.0.1:{self.proxy_port}. exit={rc}\n{err}")

    def test_a_chunked_upload_arrives_intact(self):
        payload = json.dumps(self.BODY).encode()
        conn = http.client.HTTPConnection("127.0.0.1", self.proxy_port, timeout=5)
        conn.putrequest("POST", "/v1/messages")
        conn.putheader("content-type", "application/json")
        conn.putheader("transfer-encoding", "chunked")
        conn.endheaders()
        conn.send(b"%X\r\n%s\r\n0\r\n\r\n" % (len(payload), payload))
        conn.getresponse().read()
        conn.close()
        self.assertEqual(_StreamStub.seen_body, payload,
                         "the chunked body was dropped or truncated")

    def test_a_streaming_request_is_refused_not_reframed(self):
        """urllib buffers the upstream response, so an event stream reaches the
        client only once the model has finished -- or times out and becomes a
        synthetic 502.

        Relaying it chunk by chunk was tried and measured: first byte still
        arrived at 1.22s on a stream spanning 1.2s, so the buffering is above
        the relay. A capture that changes the behaviour it exists to observe is
        worse than no capture, so the proxy says so instead of pretending.
        """
        body = dict(self.BODY, stream=True)
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.proxy_port}/v1/messages",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(caught.exception.code, 501)
        detail = json.loads(caught.exception.read())["error"]["message"]
        self.assertIn("will not pretend", detail)
        self.assertEqual(_StreamStub.hits, 0,
                         "the request was forwarded before being refused")

    def test_a_non_streaming_request_is_still_captured(self):
        """The refusal must be narrow: ordinary traffic still goes through."""
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.proxy_port}/v1/messages",
            data=json.dumps(self.BODY).encode(),
            headers={"content-type": "application/json"})
        urllib.request.urlopen(req, timeout=5).read()
        for _ in range(40):
            rows = [l for l in open(self.out.name) if l.strip()]
            if rows:
                break
            time.sleep(0.05)
        self.assertTrue(rows, "a streamed request was not captured at all")




class TestSweepEvidenceCarriesItsGateState(unittest.TestCase):
    """`sweep_report.py` runs the CLI with --allow-unreconciled.

    That releases figures the analyzer stamps DRAFT and labels not for external
    use, and the script parses those released strings into a committed
    artifact -- so a file in tier-b/evidence can carry dollar projections the
    normal gate would have withheld.

    The first version of this test parsed `analyse()` for dict keys with `ast`.
    Returning `unreconciled=False` with an empty gate string passed it, which
    is the same source-inspection failure this file had just been cleaned of,
    reintroduced one turn later by the person doing the cleaning. It calls the
    real function against a real fixture now.
    """

    def _bodies(self, tmp):
        p = os.path.join(tmp, "run.jsonl")
        body = {"model": "claude-opus-5",
                "system": [{"type": "text", "text": "s" * 30_000,
                            "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": "hi"}]}
        with open(p, "w") as f:
            for i in range(20):
                f.write(json.dumps({
                    "sent_at": f"2026-07-29T09:{i:02d}:00Z", "body": body,
                    "usage": {"input_tokens": 200,
                              "cache_read_input_tokens": 30_000 if i else 0,
                              "cache_creation_input_tokens": 0 if i else 30_000,
                              "cache_creation": {
                                  "ephemeral_5m_input_tokens": 0 if i else 30_000,
                                  "ephemeral_1h_input_tokens": 0}}}) + "\n")
        return p

    def _analyse(self, tmp):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "sweep_report_under_test", os.path.join(TIER_B, "sweep_report.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.analyse(self._bodies(tmp))

    def test_the_result_declares_it_is_unreconciled(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = self._analyse(tmp)
        if "error" in got:
            self.fail(f"sweep analyse failed: {got['error']}")
        self.assertIs(got.get("unreconciled"), True)
        self.assertTrue((got.get("gate") or "").strip(),
                        "a dollar field shipped with an empty gate reason")

    def test_the_gate_reason_names_what_released_the_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = self._analyse(tmp)
        if "error" in got:
            self.fail(got["error"])
        self.assertIn("unreconciled", (got.get("gate") or "").lower())

    def test_a_dollar_field_never_travels_without_the_declaration(self):
        """The property rather than the field list: if a money key is present,
        the gate state must be present and true."""
        with tempfile.TemporaryDirectory() as tmp:
            got = self._analyse(tmp)
        if "error" in got:
            self.fail(got["error"])
        money = [k for k in got if "usd" in k]
        self.assertTrue(money, "fixture produced no dollar fields; vacuous")
        self.assertIs(got.get("unreconciled"), True)


class TestTheDechunkerTellsOneNothingFromAnother(unittest.TestCase):
    """`readline()` returns b"" at EOF and b"\\r\\n" on a blank line, and both
    are falsy after `.strip()`.

    The loop treated them alike and did `continue`, so a client that hung up
    mid-upload span that handler thread at 100% CPU for the life of the process
    and every aborted upload leaked another one. Driven directly rather than
    through a socket, so a regression fails in milliseconds instead of hanging
    the suite.
    """

    def setUp(self):
        self.proxy = load("capture_proxy")

    def _read(self, raw):
        """Every read is deadlined, not only the two that look like loops. A
        short chunk also spins under the old code, and finding that out by
        hanging the suite for two minutes is how this deadline earned its
        place."""
        class Fake:
            def __init__(self, b):
                self.rfile = io.BytesIO(b)
                self.headers = {"transfer-encoding": "chunked"}
        with _deadline(5):
            return self.proxy.Handler._read_request_body(Fake(raw))

    def test_a_well_formed_body_still_arrives(self):
        self.assertEqual(self._read(b"5\r\nhello\r\n0\r\n\r\n"), b"hello")

    def test_eof_mid_body_raises_instead_of_spinning(self):
        with self.assertRaises(self.proxy._BadChunkedBody):
            self._read(b"5\r\nhello\r\n")

    def test_an_empty_stream_raises_instead_of_spinning(self):
        with self.assertRaises(self.proxy._BadChunkedBody):
            self._read(b"")

    def test_malformed_framing_forwards_nothing(self):
        """It used to `break` and return the bytes read so far -- a plausible
        truncated request, harder to notice than the empty body this function
        was written to fix."""
        with self.assertRaises(self.proxy._BadChunkedBody):
            self._read(b"5\r\nhello\r\nZZZZ\r\nworld\r\n0\r\n\r\n")

    def test_a_short_chunk_is_not_accepted_as_whole(self):
        with self.assertRaises(self.proxy._BadChunkedBody):
            self._read(b"99\r\nhello\r\n0\r\n\r\n")

    def test_trailers_are_drained_so_keep_alive_survives(self):
        """Left unread, the next request on that socket starts parsing at
        `X-Checksum: abc` -- one capture corrupting every request behind it."""
        class Fake:
            def __init__(self, b):
                self.rfile = io.BytesIO(b)
                self.headers = {"transfer-encoding": "chunked"}
        f = Fake(b"5\r\nhello\r\n0\r\nX-Checksum: abc\r\n\r\n"
                 b"POST /v1/messages HTTP/1.1\r\n")
        with _deadline(5):
            self.assertEqual(self.proxy.Handler._read_request_body(f), b"hello")
        self.assertTrue(f.rfile.read().startswith(b"POST "))


class TestTheCountCacheIsNotAPromptStore(unittest.TestCase):
    """Keys were `json.dumps(cut)`, so `<out>.cache.json` was a verbatim copy of
    every prompt counted -- and it was not gitignored.

    That contradicts the README's own promise that hashes, structure and token
    counts are enough, in the one file this tool writes to a client's disk
    without being asked.
    """

    def test_no_prompt_text_survives_into_the_cache(self):
        from cacheeconomics.tokenizer import count_segments
        body = {"system": [{"type": "text", "text": "SECRET-POLICY-TEXT"}],
                "messages": [{"role": "user", "content": "CONFIDENTIAL-QUESTION"}]}
        cache = {}
        count_segments(body, lambda b: 100, cache)
        blob = json.dumps(cache)
        self.assertNotIn("SECRET-POLICY-TEXT", blob)
        self.assertNotIn("CONFIDENTIAL-QUESTION", blob)
        self.assertTrue(cache)

    def test_the_cache_still_hits(self):
        """A digest that changed per call would be private and useless."""
        from cacheeconomics.tokenizer import count_segments
        body = {"messages": [{"role": "user", "content": "hello"}]}
        calls = []
        cache = {}
        count_segments(body, lambda b: (calls.append(1), 10)[1], cache)
        first = len(calls)
        count_segments(body, lambda b: (calls.append(1), 10)[1], cache)
        self.assertEqual(len(calls), first, "the second run re-counted everything")

    def test_the_cache_file_is_gitignored(self):
        root = os.path.dirname(TIER_B)
        with open(os.path.join(root, ".gitignore")) as f:
            self.assertIn("*.cache.json", f.read())


class TestCountingLargeExportsStaysLinear(unittest.TestCase):
    """The suite had no benchmark anywhere, and this is the one tool whose input
    is a client's entire month of traffic.

    Not a wall-clock threshold, which would be flaky on shared CI. It asserts the
    shape: ten times the segments must not cost anywhere near a hundred times the
    work. `prefix_cuts` is inherently quadratic in total output size -- every cut
    is a full snapshot -- so this pins that nothing *further* quadratic creeps in
    on top, which is what per-cut key serialization was.
    """

    @staticmethod
    def _elapsed(n):
        from cacheeconomics.tokenizer import count_segments
        body = {"messages": [{"role": "user",
                              "content": [{"type": "text", "text": f"block {i} " * 20}
                                          for i in range(n)]}]}
        cache = {}
        start = time.perf_counter()
        count_segments(body, lambda b: 10, cache)
        return time.perf_counter() - start

    def test_ten_times_the_segments_is_not_a_hundred_times_the_cost(self):
        self._elapsed(20)                      # warm the interpreter
        small = self._elapsed(20)
        large = self._elapsed(200)
        # Quadratic in output size gives ~100x. Anything past 100x means a
        # second quadratic factor -- which is what hashing the full prefix into
        # a dict key on every cut was.
        self.assertLess(large, small * 100 + 0.5,
                        f"200 segments cost {large:.3f}s against {small:.3f}s "
                        f"for 20 -- worse than the shape of the search itself")


class TestAMalformedUploadDoesNotTakeTheNextRequestWithIt(_ProxyCase):
    """`_BadChunkedBody` is raised partway through a body, so an unknown number
    of bytes are still unread on the socket.

    `protocol_version = "HTTP/1.1"` means keep-alive, so the next read starts
    inside the abandoned body and parses it as a request line -- the same
    corruption draining the trailers was written to prevent, reintroduced on
    the error path. One malformed upload took the following good request with
    it and dropped it from the capture silently.

    Driven over a raw socket, because `http.client` will not send framing this
    broken and the point is what a broken client does.
    """

    def _sock(self):
        s = socket.create_connection(("127.0.0.1", self.proxy_port), timeout=10)
        self.addCleanup(s.close)
        return s

    def test_the_proxy_refuses_it_and_closes_rather_than_desyncing(self):
        s = self._sock()
        s.sendall(b"POST /v1/messages HTTP/1.1\r\n"
                  b"Host: 127.0.0.1\r\n"
                  b"Transfer-Encoding: chunked\r\n\r\n"
                  b"5\r\nhello\r\nZZZZ\r\n")
        # Read to EOF, which the server closing is what makes possible -- and is
        # itself half of what this test is checking.
        reply = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            reply += chunk
        self.assertIn(b"400", reply.split(b"\r\n")[0])
        self.assertIn(b"cacheeconomics_proxy_bad_request", reply)
        # Closed, so a client reusing the socket gets a clean failure rather
        # than having its next request eaten by the abandoned body.
        head = reply.split(b"\r\n\r\n")[0].lower()
        self.assertIn(b"connection: close", head)

    def test_a_good_request_on_a_fresh_connection_still_works(self):
        """The other direction: refusing must not take the proxy down."""
        self._sock().sendall(b"POST /v1/messages HTTP/1.1\r\n"
                             b"Host: 127.0.0.1\r\n"
                             b"Transfer-Encoding: chunked\r\n\r\n"
                             b"5\r\nhello\r\nZZZZ\r\n")
        conn = http.client.HTTPConnection("127.0.0.1", self.proxy_port, timeout=10)
        conn.request("POST", "/v1/messages", json.dumps(self.BODY),
                     {"content-type": "application/json"})
        self.assertEqual(conn.getresponse().status, 200)


class TestAnAbortedCountLeavesNothingOnDisk(unittest.TestCase):
    """The streamed writer opened `<out>.partial-write` and closed it only on
    the normal path.

    Any exception between the open and the rename left an enriched export on
    disk holding client prompt bodies, under a name `.gitignore` did not cover.
    Same defect as the plaintext count cache the same change removed, one file
    over.
    """

    def setUp(self):
        self.ct = load("count_tokens")
        self.dir = tempfile.mkdtemp()
        self.src = os.path.join(self.dir, "in.jsonl")
        self.out = os.path.join(self.dir, "out.jsonl")
        with open(self.src, "w") as f:
            for i in range(4):
                f.write(json.dumps({"request": {
                    "model": "claude-opus-5",
                    "system": [{"type": "text", "text": f"SECRET-POLICY-{i}"}],
                    "messages": [{"role": "user",
                                  "content": f"CONFIDENTIAL-{i}"}]}}) + "\n")

    def _run(self, after):
        """Run the counter, raising `after` once one row has been emitted."""
        real = self.ct.count_segments
        seen = {"n": 0}

        def wrapped(body, count, cache, counter_id=""):
            seen["n"] += 1
            if seen["n"] > 1:
                raise after
            return real(body, count, cache, counter_id)

        self.ct.count_segments = wrapped
        self.addCleanup(setattr, self.ct, "count_segments", real)
        argv = sys.argv[:]
        sys.argv = ["count_tokens", self.src, "--out", self.out,
                    "--endpoint", "http://127.0.0.1:1/unused"]
        try:
            self.ct.main()
        except BaseException:                                  # noqa: BLE001
            pass
        finally:
            sys.argv = argv

    def test_an_interrupt_leaves_no_prompt_bearing_file(self):
        self._run(KeyboardInterrupt("ctrl-c"))
        left = [f for f in os.listdir(self.dir) if f != "in.jsonl"]
        for name in left:
            blob = open(os.path.join(self.dir, name)).read()
            self.assertNotIn("SECRET-POLICY", blob, f"{name} holds prompt text")
            self.assertNotIn("CONFIDENTIAL", blob, f"{name} holds prompt text")
        self.assertNotIn("out.jsonl.partial-write", left)

    def test_the_temp_name_is_gitignored_as_a_second_line_of_defence(self):
        root = os.path.dirname(TIER_B)
        with open(os.path.join(root, ".gitignore")) as f:
            self.assertIn("*.partial-write", f.read())


class TestACaptureCannotQuietlyMixTwoRuns(_ProxyCase):
    """The evidence file opened in append mode.

    Pointing two runs at one path interleaved full request and response
    bodies -- from two different clients, if an operator reused a path between
    engagements -- while the request counter restarted at one, so nothing
    downstream could tell the runs apart. A capture is evidence, and evidence
    that quietly merges with other evidence is not usable.
    """

    def _run(self, out, *extra):
        return subprocess.run(
            [sys.executable, "-B", os.path.join(TIER_B, "capture_proxy.py"),
             "--out", out, "--port", str(self._free_port()), *extra],
            capture_output=True, text=True, timeout=30,
            env=dict(os.environ, ANTHROPIC_API_KEY="test"))

    def test_an_existing_path_is_refused_rather_than_appended(self):
        existing = self.out.name          # created by setUp
        r = self._run(existing)
        self.assertEqual(r.returncode, 2)
        self.assertIn("already exists", r.stderr)
        self.assertIn("--append", r.stderr, "refused without naming the way out")

    def test_appending_is_available_but_has_to_be_typed(self):
        """Refusing must not become impossible-to-resume. The flag exists and
        says what it is doing."""
        proc = subprocess.Popen(
            [sys.executable, "-B", os.path.join(TIER_B, "capture_proxy.py"),
             "--out", self.out.name, "--append", "--port", str(self._free_port())],
            stderr=subprocess.PIPE, text=True,
            env=dict(os.environ, ANTHROPIC_API_KEY="test"))
        self.addCleanup(proc.terminate)
        deadline = time.time() + 15
        seen = ""
        while time.time() < deadline and "--append" not in seen:
            seen += proc.stderr.readline()
        self.assertIn("adding to an existing", seen)

    def test_every_captured_row_names_its_run(self):
        """Two runs can still end up in one file by hand. A row that cannot say
        which capture it came from cannot be separated out again."""
        conn = http.client.HTTPConnection("127.0.0.1", self.proxy_port, timeout=10)
        conn.request("POST", "/v1/messages", json.dumps(self.BODY),
                     {"content-type": "application/json"})
        self.assertEqual(conn.getresponse().status, 200)
        for _ in range(50):
            rows = [json.loads(l) for l in open(self.out.name) if l.strip()]
            if rows:
                break
            time.sleep(0.1)
        self.assertTrue(rows, "nothing was captured")
        self.assertTrue(all(r.get("capture_run") for r in rows))
        self.assertEqual(len({r["capture_run"] for r in rows}), 1,
                         "one process, one run id")


class TestOneDigestServesBothSidesOfTheProvenanceGate(unittest.TestCase):
    """The writer digests a body; the loader re-digests it to decide whether the
    counts may be trusted. If the two canonicalisations differ by one flag,
    every digest mismatches, every counted row silently falls back to byte-share
    estimation, and the counting feature regresses completely -- in the
    direction that looks fine. Nothing else in the suite would notice: the
    numbers would still be produced, still be reconciled, still be published,
    and they would be the estimates counting exists to replace.

    So there is one function, in the package, and both sides call it. This is
    written before either side depends on it, and it is the test that catches a
    drift: it computes a digest on both sides of the boundary and compares.
    """

    BODIES = [
        {"system": [{"type": "text", "text": "policy"}],
         "messages": [{"role": "user", "content": "hi"}]},
        # Key order reversed: canonicalisation must make these identical.
        {"messages": [{"role": "user", "content": "hi"}],
         "system": [{"type": "text", "text": "policy"}]},
        {"model": "claude-opus-5", "tools": [{"name": "t", "input_schema": {}}],
         "messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]},
        {"messages": []},
    ]

    def test_the_package_owns_the_digest(self):
        """Guards the guard: if the writer grows a private copy, the comparison
        below is comparing a function with itself."""
        from cacheeconomics import tokenizer
        self.assertTrue(hasattr(tokenizer, "body_sha256"))
        self.assertTrue(hasattr(tokenizer, "row_sha256"))
        src = open(os.path.join(TIER_B, "count_tokens.py")).read()
        self.assertNotIn("def body_sha256", src,
                         "count_tokens.py defines its own body digest; the "
                         "loader will disagree with it the first time either "
                         "canonicalisation is touched")
        self.assertNotIn("def row_sha256", src)

    def test_a_record_the_package_builds_passes_the_packages_own_gate(self):
        """The contract, as one property.

        An earlier split had `counts_provenance()` emitting a "vouching record"
        that `_counts_are_vouched` then rejected, because the two fields the
        loader cannot recompute lived only in the writer's layer. Twelve fixtures
        found it; a caller would have found it in production, as counts that were
        written, paid for, and quietly estimated.

        So: whatever the package says a vouching record is, the package's own
        gate must accept it. Anything less is a contract that holds only by
        agreement between two files.
        """
        from cacheeconomics.adapters.bodies import _counts_are_vouched
        from cacheeconomics.tokenizer import (COUNTS_PROVENANCE_KEY,
                                              counts_provenance)
        for body in self.BODIES:
            row = {"body": body}
            row[COUNTS_PROVENANCE_KEY] = counts_provenance(
                body, row, None, "https://endpoint", "tokenizer-1")
            with self.subTest(body=body):
                self.assertTrue(
                    _counts_are_vouched(row, body, None),
                    "the package built a record its own gate refuses")

    def test_the_gate_still_refuses_a_record_missing_either_half(self):
        """The other direction, so the property above cannot be satisfied by a
        gate that accepts everything."""
        from cacheeconomics.adapters.bodies import _counts_are_vouched
        from cacheeconomics.tokenizer import (COUNTS_PROVENANCE_KEY,
                                              counts_provenance,
                                              recomputable_provenance)
        body = self.BODIES[0]
        row = {"body": body}
        row[COUNTS_PROVENANCE_KEY] = recomputable_provenance(body, row, None)
        self.assertFalse(_counts_are_vouched(row, body, None),
                         "a record with no endpoint or tokenizer identity was "
                         "accepted")
        row[COUNTS_PROVENANCE_KEY] = counts_provenance(
            body, row, None, "https://endpoint", "tokenizer-1")
        row[COUNTS_PROVENANCE_KEY]["body_sha256"] = "tampered"
        self.assertFalse(_counts_are_vouched(row, body, None))

    def test_the_writer_stamps_exactly_what_the_loader_checks(self):
        """The comparison that matters, now that the writer holds no digest of
        its own: every field the loader requires must appear in the record the
        writer actually emits, with the same value."""
        from cacheeconomics.tokenizer import recomputable_provenance
        writer = load("count_tokens")
        for body in self.BODIES:
            stamped = writer.provenance({"body": body}, body,
                                        "https://e", None, None)
            with self.subTest(body=body):
                for k, v in recomputable_provenance(
                        body, {"body": body}, None).items():
                    self.assertEqual(stamped.get(k), v,
                                     f"the writer's {k} is not what the loader "
                                     f"will compute for the same body")

    def test_key_order_does_not_change_the_digest(self):
        """The property that makes the digest usable at all: a JSON round trip
        through a different exporter must not invalidate every count."""
        from cacheeconomics.tokenizer import body_sha256
        self.assertEqual(body_sha256(self.BODIES[0]),
                         body_sha256(self.BODIES[1]))

    def test_content_changes_do_change_the_digest(self):
        from cacheeconomics.tokenizer import body_sha256
        a = {"messages": [{"role": "user", "content": "hi"}]}
        b = {"messages": [{"role": "user", "content": "hi "}]}
        self.assertNotEqual(body_sha256(a), body_sha256(b))

    def test_it_follows_the_convention_the_module_already_had(self):
        """`_cache_key` established `json.dumps(..., sort_keys=True,
        default=str)` over sha256 in this module. A second convention beside it
        is a second thing to keep in step."""
        from cacheeconomics import tokenizer
        expected = hashlib.sha256(
            json.dumps(self.BODIES[0], sort_keys=True,
                       default=str).encode()).hexdigest()
        self.assertEqual(tokenizer.body_sha256(self.BODIES[0]), expected)

    def test_a_body_that_is_not_json_serialisable_still_digests(self):
        """`default=str` is part of the convention: an exporter that put a
        datetime in a body must not crash the loader's freshness check."""
        from cacheeconomics.tokenizer import body_sha256
        import datetime as dt
        body = {"messages": [{"role": "user", "content": dt.date(2026, 8, 1)}]}
        self.assertEqual(len(body_sha256(body)), 64)

    def test_the_digest_carries_no_prompt_text(self):
        from cacheeconomics.tokenizer import body_sha256, row_sha256
        body = {"system": [{"type": "text", "text": "SECRET-POLICY"}],
                "messages": [{"role": "user", "content": "CONFIDENTIAL"}]}
        for digest in (body_sha256(body), row_sha256({"body": body})):
            self.assertNotIn("SECRET", digest)
            self.assertNotIn("CONFIDENTIAL", digest)


class TestTheLoaderOnlyTrustsCountsItCanVouchFor(unittest.TestCase):
    """`load_bodies` accepted any correctly-shaped positive `segment_tokens`
    array as EXACT: right length, token-ish values, positive sum. Shape was the
    whole test.

    So a counted export from a different endpoint, a different tokenizer, an
    older counter, or a capture that has since been re-recorded loaded with
    `was_counted=True`, `tokens_counted` reached 1.0, and that is what releases
    structural dollars. tier-b could refuse to *reuse* such a file; nothing
    stopped one being handed straight to `analyze`.

    The counts must now come with a record showing they were taken from this
    body, cut this way, by a counter this loader knows. Unvouched rows are
    estimated by byte share and named in a note -- never dropped, because the
    row's billed total is still real and dropping it would understate spend.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.ct = load("count_tokens")

    BODY = {"model": "claude-opus-5",
            "system": [{"type": "text", "text": "policy " * 40}],
            "messages": [{"role": "user", "content": "hello there"}]}

    def _row(self, counts, provenance="valid", body=None):
        body = body or json.loads(json.dumps(self.BODY))
        row = {"sent_at": "2026-08-01T09:00:00Z", "body": body,
               "usage": {"input_tokens": 1000, "cache_read_input_tokens": 0,
                         "cache_creation_input_tokens": 0},
               "segment_tokens": counts}
        if provenance == "valid":
            row[self.ct.PROVENANCE_KEY] = self.ct.provenance(
                row, body, "https://anything", None, "stub-tokenizer-1")
        elif provenance is not None:
            row[self.ct.PROVENANCE_KEY] = provenance
        return row

    def _load(self, *rows):
        from cacheeconomics.adapters.bodies import load_bodies
        p = os.path.join(self.dir, f"t{len(os.listdir(self.dir))}.jsonl")
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return load_bodies(p, key=b"k" * 32)

    def _counts_for(self, body=None):
        """A `segment_tokens` array of the right length for this body."""
        from cacheeconomics.segment import segments_from_request
        segs = segments_from_request(body or self.BODY, b"k" * 32, None)
        return [10] * len(segs)

    def test_the_fixture_is_accepted_when_it_is_vouched_for(self):
        """Guards the guard. If the fixture never loaded as counted, every
        refusal below would pass for the wrong reason."""
        ts = self._load(self._row(self._counts_for()))
        self.assertEqual(ts.tokens_counted, 1.0,
                         "the fixture does not load as counted at all")

    def test_counts_with_no_record_are_estimated(self):
        ts = self._load(self._row(self._counts_for(), provenance=None))
        self.assertEqual(ts.tokens_counted, 0.0,
                         "counts with no provenance were accepted as exact")

    def test_counts_from_an_unknown_counter_version_are_estimated(self):
        from cacheeconomics.tokenizer import body_sha256, cuts_sha256
        row = self._row(self._counts_for())
        row[self.ct.PROVENANCE_KEY]["version"] = self.ct.COUNTER_VERSION + 1
        self.assertEqual(self._load(row).tokens_counted, 0.0)

    def test_counts_taken_from_a_different_body_are_estimated(self):
        row = self._row(self._counts_for())
        row["body"]["messages"][0]["content"] = "something else entirely"
        self.assertEqual(self._load(row).tokens_counted, 0.0,
                         "counts were applied to a body they never described")

    def test_a_resegmentation_that_keeps_the_segment_count_is_estimated(self):
        """Track B's case, and the reason the digest covers the prefix cuts and
        not only the body bytes.

        The length check and a body digest are both satisfied when a body is
        re-cut into the same *number* of segments — so the counts would be
        applied, in order, to segments they never corresponded to, and
        `tokens_counted` would clear the publish gate on stale proportions.
        """
        from cacheeconomics.tokenizer import body_sha256, cuts_sha256, prefix_cuts
        row = self._row(self._counts_for())
        original = json.loads(json.dumps(row["body"]))
        # Same segment count, different content in one of them: the cuts differ,
        # the segment count does not.
        row["body"]["system"][0]["text"] = "a different policy " * 40
        self.assertEqual(len(prefix_cuts(original)),
                         len(prefix_cuts(row["body"])),
                         "the fixture changed the segment count, so this would "
                         "have been caught by the length check and proves "
                         "nothing about the cuts digest")
        # Body digest deliberately left matching the *new* body, so only the
        # cuts digest can reject this.
        row[self.ct.PROVENANCE_KEY]["body_sha256"] = body_sha256(row["body"])
        self.assertNotEqual(row[self.ct.PROVENANCE_KEY]["cuts_sha256"],
                            cuts_sha256(row["body"]))
        self.assertEqual(self._load(row).tokens_counted, 0.0,
                         "counts survived a re-segmentation and would have been "
                         "applied to segments they never described")

    def test_an_unvouched_row_is_estimated_and_not_dropped(self):
        """Never rejected. The billed total is real and the structure is
        readable; only the claim that the sizes are exact is unsupported."""
        ts = self._load(self._row(self._counts_for(), provenance=None))
        self.assertEqual(len(ts.requests), 1, "the row was dropped")
        self.assertGreater(sum(s.tokens for s in ts.requests[0].segments), 0)

    def test_the_report_says_it_estimated_them(self):
        ts = self._load(self._row(self._counts_for(), provenance=None))
        said = " ".join(ts.notes)
        self.assertIn("could not vouch for", said)
        self.assertIn("estimated", said)

    def test_a_mixed_export_counts_only_the_vouched_rows(self):
        ts = self._load(self._row(self._counts_for()),
                        self._row(self._counts_for(), provenance=None))
        self.assertGreater(ts.tokens_counted, 0.0)
        self.assertLess(ts.tokens_counted, 1.0)

    def test_what_the_writer_emits_is_accepted_end_to_end(self):
        """The two halves against each other, through the real script rather
        than a hand-built record: whatever `count_tokens.py` writes must be what
        the loader vouches for. A drift in either direction shows up here as
        counting silently ceasing to work."""
        stub = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _CountStub)
        threading.Thread(target=stub.serve_forever, daemon=True).start()
        self.addCleanup(stub.shutdown)
        src = os.path.join(self.dir, "cap.jsonl")
        with open(src, "w") as f:
            f.write(json.dumps({
                "sent_at": "2026-08-01T09:00:00Z", "body": self.BODY,
                "usage": {"input_tokens": 1000, "cache_read_input_tokens": 0,
                          "cache_creation_input_tokens": 0}}) + "\n")
        out = os.path.join(self.dir, "counted.jsonl")
        r = subprocess.run(
            [sys.executable, "-B", os.path.join(TIER_B, "count_tokens.py"), src,
             "-o", out, "--tokenizer-id", "stub-1", "--endpoint",
             f"http://127.0.0.1:{stub.server_address[1]}/v1/messages/count_tokens"],
            capture_output=True, text=True, timeout=90,
            env=dict(os.environ, ANTHROPIC_API_KEY="test"))
        self.assertEqual(r.returncode, 0, r.stderr[-400:])
        from cacheeconomics.adapters.bodies import load_bodies
        ts = load_bodies(out, key=b"k" * 32)
        self.assertEqual(ts.tokens_counted, 1.0,
                         "the loader would not vouch for what the counter just "
                         "wrote; the two sides have drifted")

    def test_a_flat_row_survives_the_round_trip(self):
        """The aliasing defect, end to end, and the shape every other fixture
        here misses.

        For a flattened export `_find_body` returns the ROW ITSELF, so `body`
        and `row` are one object: storing `segment_tokens` mutated the thing
        whose digest was about to be taken, and adding the record mutated it
        again. On reload the loader hashed the *enriched* flat row, the digest
        missed, and the row was estimated -- after its prompt prefixes had
        already gone to the tokenizer. Flat exports paid full egress and got
        nothing.
        """
        stub = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _CountStub)
        threading.Thread(target=stub.serve_forever, daemon=True).start()
        self.addCleanup(stub.shutdown)
        src = os.path.join(self.dir, "flat.jsonl")
        # No `body`/`request` wrapper: the request fields sit on the row.
        flat = {"sent_at": "2026-08-01T09:00:00Z",
                "model": "claude-opus-5",
                "system": [{"type": "text", "text": "policy " * 40}],
                "messages": [{"role": "user", "content": "hello there"}],
                "usage": {"input_tokens": 1000, "cache_read_input_tokens": 0,
                          "cache_creation_input_tokens": 0}}
        with open(src, "w") as f:
            f.write(json.dumps(flat) + "\n")
        from cacheeconomics.adapters.bodies import _find_body
        self.assertIs(_find_body(json.loads(json.dumps(flat))) is None, False)
        out = os.path.join(self.dir, "flat-counted.jsonl")
        r = subprocess.run(
            [sys.executable, "-B", os.path.join(TIER_B, "count_tokens.py"), src,
             "-o", out, "--tokenizer-id", "stub-1", "--endpoint",
             f"http://127.0.0.1:{stub.server_address[1]}/v1/messages/count_tokens"],
            capture_output=True, text=True, timeout=90,
            env=dict(os.environ, ANTHROPIC_API_KEY="test"))
        self.assertEqual(r.returncode, 0, r.stderr[-400:])
        row = json.loads(open(out).read().strip())
        self.assertIn("segment_tokens", row, "the flat row was never counted")
        from cacheeconomics.adapters.bodies import load_bodies
        ts = load_bodies(out, key=b"k" * 32)
        self.assertEqual(ts.tokens_counted, 1.0,
                         "a counted flat row did not load as exact; its prompt "
                         "prefixes were sent and the counts thrown away")

    def test_a_digest_ignores_what_enrichment_adds(self):
        """Why the flat case works: the digest is over the request, not over
        whatever the row later accumulates."""
        from cacheeconomics.tokenizer import body_sha256, row_sha256
        flat = {"model": "claude-opus-5",
                "messages": [{"role": "user", "content": "hi"}]}
        enriched = dict(flat, segment_tokens=[1, 2],
                        segment_tokens_provenance={"version": 99})
        self.assertEqual(body_sha256(flat), body_sha256(enriched))
        self.assertEqual(row_sha256(flat), row_sha256(enriched))

    def test_the_loader_checks_everything_it_can_recompute(self):
        """The loader gated structural dollars on three fields while
        `sweep_report` checked eight, so the weaker gate was the one that
        mattered. Every field the loader can recompute must now match."""
        from cacheeconomics.tokenizer import recomputable_provenance
        for field in recomputable_provenance(self.BODY, {"body": self.BODY},
                                             None):
            with self.subTest(field=field):
                row = self._row(self._counts_for())
                row[self.ct.PROVENANCE_KEY][field] = "tampered"
                self.assertEqual(self._load(row).tokens_counted, 0.0,
                                 f"a counted row survived a changed {field}")

    def test_counts_with_no_asserted_tokenizer_are_estimated(self):
        """The two fields the loader cannot recompute are required to be
        present. A counted export produced without --tokenizer-id claims nothing
        about what answered, which is the unbacked claim the sweep already
        refuses to reuse -- the loader was accepting it."""
        for field in ("endpoint", "tokenizer_id"):
            with self.subTest(field=field):
                row = self._row(self._counts_for())
                row[self.ct.PROVENANCE_KEY][field] = None
                self.assertEqual(self._load(row).tokens_counted, 0.0)

    def test_the_two_rejection_reasons_are_counted_separately(self):
        """One counter fed both notes, so the shape-mismatch note reported a
        figure that included rows whose shape was fine. The two say different
        things: re-run the counter, versus these counts may be someone else's."""
        shape_bad = self._row([1])                    # wrong length
        vouch_bad = self._row(self._counts_for(), provenance=None)
        ts = self._load(shape_bad, vouch_bad)
        shape_note = next(n for n in ts.notes if "did not match their segments" in n)
        vouch_note = next(n for n in ts.notes if "could not vouch for" in n)
        self.assertIn("1 request(s)", shape_note)
        self.assertIn("1 request(s)", vouch_note)

    def test_the_accepted_version_tracks_the_counter(self):
        """The two constants are bumped in lockstep. If they part, the loader
        silently estimates everything the current counter produces."""
        from cacheeconomics.adapters import bodies
        self.assertEqual(bodies.ACCEPTED_COUNTER_VERSION,
                         self.ct.COUNTER_VERSION)
        self.assertEqual(bodies.PROVENANCE_KEY, self.ct.PROVENANCE_KEY)


class TestTheCountCacheIsScopedToItsCounter(unittest.TestCase):
    """The cache key was a digest of the prompt prefix and nothing else, while
    `count_tokens.py` exposes both `--model` and `--endpoint`.

    So a cache written by one tokenizer was reused by another without a single
    call being made — and those counts load as *exact*: `tokens_counted` reaches
    1.0 and structural money is released on per-segment sizes from a model that
    was never asked. Different Claude generations tokenize differently, and a
    gateway endpoint may not be Anthropic's tokenizer at all.
    """

    BODY = {"model": "claude-opus-5",
            "system": [{"type": "text", "text": "policy " * 200}],
            "messages": [{"role": "user", "content": "hi"}]}

    def _run(self, cache, counter_id, value):
        from cacheeconomics.tokenizer import count_segments
        calls = []
        count_segments(self.BODY, lambda b: (calls.append(1), value)[1],
                       cache, counter_id)
        return len(calls)

    def test_a_different_model_is_actually_asked(self):
        cache = {}
        self._run(cache, "claude-opus-5\x00https://a", 100)
        self.assertGreater(self._run(cache, "claude-haiku-4-5\x00https://a", 999), 0)

    def test_a_different_endpoint_is_actually_asked(self):
        cache = {}
        self._run(cache, "claude-opus-5\x00https://a", 100)
        self.assertGreater(self._run(cache, "claude-opus-5\x00https://b", 999), 0)

    def test_the_same_counter_still_resumes_for_free(self):
        """The other direction. A key that never hits has traded a wrong answer
        for a bill, which is its own kind of wrong."""
        cache = {}
        self._run(cache, "claude-opus-5\x00https://a", 100)
        self.assertEqual(self._run(cache, "claude-opus-5\x00https://a", 100), 0)

    def test_the_counts_do_not_bleed_between_counters(self):
        """Not just that it re-ran, but that it got its own answer."""
        from cacheeconomics.tokenizer import count_segments
        # Counters that scale with body size, so the *differences* differ.
        # A constant counter returns all-zero differences either way, which
        # would pass this test without proving anything.
        cache = {}
        a = count_segments(self.BODY, lambda b: len(json.dumps(b)),
                           cache, "opus\x00https://a")
        b = count_segments(self.BODY, lambda b: len(json.dumps(b)) * 10,
                           cache, "haiku\x00https://a")
        self.assertTrue(any(a), "fixture produced no counts at all")
        self.assertNotEqual(a, b)

    def test_scoping_did_not_put_prompt_text_back_in_the_cache(self):
        """The key is still a digest. This is the property the digest-key change
        established and it must survive a change to what is digested."""
        from cacheeconomics.tokenizer import count_segments
        cache = {}
        count_segments({"system": [{"type": "text", "text": "SECRET-POLICY"}],
                        "messages": [{"role": "user", "content": "CONFIDENTIAL"}]},
                       lambda b: 10, cache, "claude-opus-5\x00https://a")
        blob = json.dumps(cache)
        self.assertNotIn("SECRET-POLICY", blob)
        self.assertNotIn("CONFIDENTIAL", blob)
        self.assertNotIn("claude-opus-5", blob)


class _ModelStub(http.server.BaseHTTPRequestHandler):
    """A count endpoint that records which tokenizer each payload asked for.

    Answers proportionally to body size *and* scales by model, so the same text
    counted by two models gives two different answers. A stub answering the same
    for both would let a cache shared between them pass.
    """

    seen = []
    # The dated id is served alongside the bare one, which is the whole point of
    # the tokenizer/analysis split: a real endpoint accepts both and they are not
    # required to answer the same, so the scales differ here too.
    SCALE = {"claude-opus-5": 4, "claude-haiku-4-5": 1, "claude-sonnet-4-6": 2,
             "claude-opus-5-20260101": 3, "anthropic.claude-opus-5": 5}

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(
            int(self.headers.get("content-length") or 0)) or b"{}")
        model = body.get("model")
        type(self).seen.append(model)
        if model not in type(self).SCALE:
            # Like a gateway handed a model id it does not serve. The row then
            # fails to count rather than being counted by something else.
            self.send_response(400)
            self.send_header("content-length", "0")
            self.end_headers()
            return
        n = len(json.dumps(body, default=str)) // 4 * type(self).SCALE[model]
        payload = json.dumps({"input_tokens": n}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class TestEveryRowIsCountedByTheTokenizerItNames(unittest.TestCase):
    """`--model` was stamped over the model each row's own body carried
    (`payload["model"] = model`), and it was also the model half of the single
    `counter_id` every row was cached under.

    So an export holding both an opus planner and a haiku worker — the ordinary
    shape of a month of agent traffic — was counted end to end by one tokenizer
    and cached under one key. Measured before the fix on the two-row export
    below, whose rows differ only in the model they name: 5 of 5 outgoing
    payloads carried `claude-haiku-4-5`, including both prefixes of the row
    whose body said `claude-opus-5`, and the two rows came back with identical
    `segment_tokens`. Those counts then load as *exact* — `tokens_counted`
    reaches 1.0 and structural money is released on per-segment sizes the other
    model was never asked for.

    The scoping property `TestTheCountCacheIsScopedToItsCounter` establishes is
    only worth something if the model in the key is the one that answered, which
    is why these two classes belong next to each other.

    The first fix read the model out of the *extracted body* only, which closed
    one row shape and left the class open. An exporter that names the model at
    the top level of the row and nests the body under `body` is ordinary, and
    for those rows `body.get("model")` is None: a row declaring claude-opus-5
    resolved to the fallback claude-haiku-4-5, was cached under the fallback's
    counter_id, and still loaded as exact. So the shapes are enumerated here,
    and the precedence is the loader's own rather than a second opinion —
    `bodies.load_bodies` passes `model_override=(body or {}).get("model")` into
    `request_from_row`, which resolves `model_override or _first(row, "model")`.
    """

    def setUp(self):
        _ModelStub.seen = []
        self.stub = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ModelStub)
        self.port = self.stub.server_address[1]
        threading.Thread(target=self.stub.serve_forever, daemon=True).start()
        self.addCleanup(self.stub.shutdown)
        self.dir = tempfile.mkdtemp()
        self.out = os.path.join(self.dir, "counted.jsonl")

    def _body(self):
        return {"system": [{"type": "text", "text": "policy " * 200}],
                "messages": [{"role": "user", "content": "hi"}]}

    def _export(self, *models):
        """One row per model, named in the body. Bodies otherwise identical.

        Identical text on purpose: any difference in the counts that come back
        is then attributable to the tokenizer and nothing else.
        """
        p = os.path.join(self.dir, "mixed.jsonl")
        with open(p, "w") as f:
            for m in models:
                body = self._body()
                if m is not None:
                    body["model"] = m
                f.write(json.dumps({"sent_at": "2026-08-01T09:00:00Z",
                                    "body": body,
                                    "usage": {"input_tokens": 500}}) + "\n")
        return p

    def _top_level_export(self, *models):
        """The other ordinary shape: model on the row, body nested under it and
        carrying no model of its own. This is the shape the first fix missed."""
        p = os.path.join(self.dir, "toplevel.jsonl")
        with open(p, "w") as f:
            for m in models:
                row = {"sent_at": "2026-08-01T09:00:00Z", "body": self._body(),
                       "usage": {"input_tokens": 500}}
                if m is not None:
                    row["model"] = m
                assert "model" not in row["body"]
                f.write(json.dumps(row) + "\n")
        return p

    def _run(self, src, *extra):
        return subprocess.run(
            [sys.executable, "-B", os.path.join(TIER_B, "count_tokens.py"), src,
             "-o", self.out,
             "--endpoint", f"http://127.0.0.1:{self.port}/count", *extra],
            capture_output=True, text=True, timeout=90,
            env=dict(os.environ, ANTHROPIC_API_KEY="test"))

    def _rows(self):
        return [json.loads(l) for l in open(self.out) if l.strip()]

    def test_both_models_in_the_export_are_actually_asked(self):
        r = self._run(self._export("claude-opus-5", "claude-haiku-4-5"))
        self.assertEqual(r.returncode, 0, r.stderr[-400:])
        self.assertEqual(set(_ModelStub.seen),
                         {"claude-opus-5", "claude-haiku-4-5"},
                         "a mixed export was counted by one tokenizer")

    def test_no_payload_carries_a_model_no_row_asked_for(self):
        """The other direction: not merely that both appeared, but that nothing
        else did. A flag stamped over the rows shows up here."""
        self._run(self._export("claude-opus-5", "claude-haiku-4-5"),
                  "--model", "claude-sonnet-4-6")
        self.assertNotIn("claude-sonnet-4-6", set(_ModelStub.seen),
                         "the --model flag reached the wire for a row that "
                         "named its own model")

    def test_a_model_named_on_the_row_is_asked_when_the_body_omits_it(self):
        """The shape the first fix missed. `_find_body` returns the nested body,
        which carries no model, so reading the body alone resolved the fallback
        for a row that plainly declares one."""
        r = self._run(self._top_level_export("claude-opus-5"))
        self.assertEqual(r.returncode, 0, r.stderr[-400:])
        self.assertEqual(set(_ModelStub.seen), {"claude-opus-5"},
                         "a row naming its model at the top level was counted "
                         "by something else")

    def test_two_top_level_models_are_both_asked(self):
        """The mixed case for that shape, which is the one that costs money:
        different top-level models, neither named in the body."""
        r = self._run(self._top_level_export("claude-opus-5",
                                             "claude-haiku-4-5"))
        self.assertEqual(r.returncode, 0, r.stderr[-400:])
        self.assertEqual(set(_ModelStub.seen),
                         {"claude-opus-5", "claude-haiku-4-5"})
        rows = self._rows()
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]["segment_tokens"], rows[1]["segment_tokens"],
                            "identical bodies under two top-level models were "
                            "counted identically")

    def test_the_body_wins_over_the_row_as_the_loader_says(self):
        """Precedence, not merely presence — and the direction the analyzer
        uses, so the counts are attributed to the model the report will name.

        `request_from_row` takes `model_override or _first(row, "model")` with
        `model_override` set from the body, so the body is the more
        authoritative of the two. Counting has to agree, or the segment sizes
        belong to one model and the row they are attached to names another.
        """
        p = os.path.join(self.dir, "both.jsonl")
        body = self._body()
        body["model"] = "claude-opus-5"
        with open(p, "w") as f:
            f.write(json.dumps({"sent_at": "2026-08-01T09:00:00Z",
                                "model": "claude-haiku-4-5", "body": body,
                                "usage": {"input_tokens": 500}}) + "\n")
        self._run(p)
        self.assertEqual(set(_ModelStub.seen), {"claude-opus-5"})

    def test_two_rows_differing_only_in_model_get_different_counts(self):
        """The consequence, in the output rather than on the wire. Same text,
        two tokenizers, so the same `segment_tokens` for both means one
        tokenizer answered for both."""
        self._run(self._export("claude-opus-5", "claude-haiku-4-5"))
        rows = self._rows()
        self.assertEqual(len(rows), 2)
        opus, haiku = rows[0]["segment_tokens"], rows[1]["segment_tokens"]
        self.assertTrue(any(opus), "fixture produced no counts at all")
        self.assertNotEqual(opus, haiku,
                            "identical bodies under two models were counted "
                            "identically, so one tokenizer answered for both")

    def test_the_cache_is_keyed_by_the_model_that_answered(self):
        """A warm cache must resume for the models it was written for, and must
        not answer for one it was not.

        The third model arrives as a third row rather than a flag: the same
        prefix text, a model the cache has never held. If the key followed
        anything but the model that answered, those prefixes would come back
        free and exact from counts a different tokenizer produced.

        Run with `--tokenizer-id`, because resuming across runs is opt-in now —
        see `test_a_rerun_does_not_resume_without_an_asserted_tokenizer`.
        """
        tid = ("--tokenizer-id", "stub-1")
        self._run(self._export("claude-opus-5", "claude-haiku-4-5"), *tid)
        first = len(_ModelStub.seen)
        self.assertGreater(first, 0)

        self._run(self._export("claude-opus-5", "claude-haiku-4-5"), *tid)
        self.assertEqual(len(_ModelStub.seen), first,
                         "a warm cache re-counted rows it already held")

        self._run(self._export("claude-opus-5", "claude-haiku-4-5",
                               "claude-sonnet-4-6"), *tid)
        self.assertGreater(len(_ModelStub.seen), first,
                           "counts written for two models were handed back for "
                           "a third that was never asked")
        self.assertIn("claude-sonnet-4-6", set(_ModelStub.seen))

    def test_a_run_without_a_tokenizer_id_says_so_before_it_sends(self):
        """Correct behaviour discovered too late to act on is its own defect.

        Without `--tokenizer-id` the analyzer estimates these rows rather than
        trusting them, which is the right direction to fail -- but an operator
        who learns it from a downstream note has already spent the money and the
        egress. The warning is emitted before the first call, while the choice is
        still theirs, and the run is not refused: the counts may be wanted for
        something other than a report.
        """
        src = self._export("claude-opus-5")
        r = self._run(src)
        self.assertEqual(r.returncode, 0, r.stderr[-300:])
        self.assertIn("--tokenizer-id", r.stderr)
        self.assertIn("NOT load as exact", r.stderr)
        self.assertGreater(len(_ModelStub.seen), 0,
                           "the run was refused; it should warn and proceed")

    def test_the_notice_comes_before_the_row_work_not_after_it(self):
        """Ordering, asserted rather than assumed.

        "Before it sends" is the whole point, so it has to be observable. A row
        the counter refuses locally puts a `row 0:` line on the same stream after
        the notice, which makes the order checkable within one stream.
        """
        p = os.path.join(self.dir, "unknown.jsonl")
        with open(p, "w") as f:
            f.write(json.dumps({"body": {
                "model": "model-nobody-has-heard-of",
                "messages": [{"role": "user", "content": "hi"}]}}) + "\n")
        r = self._run(p, "--allow-partial")
        self.assertIn("NOT load as exact", r.stderr)
        self.assertIn("row 0:", r.stderr)
        self.assertLess(r.stderr.index("NOT load as exact"),
                        r.stderr.index("row 0:"),
                        "the notice was printed after the rows were processed, "
                        "which is after the decision it exists to inform")

    def test_a_run_with_a_tokenizer_id_does_not_nag(self):
        """The other direction: a notice that always fires is noise, and noise
        is how a real one gets missed."""
        r = self._run(self._export("claude-opus-5"),
                      "--tokenizer-id", "stub-1")
        self.assertNotIn("NOT load as exact", r.stderr)

    def test_the_dry_run_says_it_too(self):
        """The dry run exists to answer "what would this do" while the answer can
        still change the decision, so "you would pay for counts the report then
        estimates" belongs in it alongside the call count."""
        r = self._run(self._export("claude-opus-5"), "--dry-run")
        self.assertEqual(r.returncode, 0, r.stderr[-300:])
        said = r.stdout + r.stderr
        self.assertIn("--tokenizer-id", said)
        self.assertIn("before spending the egress", said)
        self.assertEqual(_ModelStub.seen, [], "a dry run sent something")

    def test_the_dry_run_does_not_nag_when_the_flag_is_given(self):
        r = self._run(self._export("claude-opus-5"), "--dry-run",
                      "--tokenizer-id", "stub-1")
        self.assertNotIn("before spending the egress", r.stdout + r.stderr)

    def test_a_rerun_does_not_resume_without_an_asserted_tokenizer(self):
        """Provenance is only as good as the cache it is stamped over.

        The cache key was `model\\0endpoint`, all of which can be identical while
        the deployment behind the endpoint has been replaced. A rerun then made
        zero calls, reused every prefix count, and wrote rows stamped with
        *current* provenance — which passed the freshness check downstream. So
        the cache is neither read nor written unless the operator asserts an
        identity for what is answering.
        """
        src = self._export("claude-opus-5")
        self._run(src)
        first = len(_ModelStub.seen)
        self.assertGreater(first, 0)
        self.assertFalse(os.path.exists(self.out + ".cache.json"),
                         "a cache that can never be safely resumed was written")
        self._run(src)
        self.assertEqual(len(_ModelStub.seen), first * 2,
                         "a rerun resumed from a cache nothing could vouch for")

    def test_a_cache_written_under_one_tokenizer_id_is_not_read_under_another(self):
        src = self._export("claude-opus-5")
        self._run(src, "--tokenizer-id", "gateway-41")
        first = len(_ModelStub.seen)
        self._run(src, "--tokenizer-id", "gateway-42")
        self.assertEqual(len(_ModelStub.seen), first * 2,
                         "counts from one tokenizer deployment were handed back "
                         "for another")

    def test_the_counter_version_scopes_the_cache_too(self):
        """A change to how a count is produced must not be resumable across."""
        m = load("count_tokens")
        a = m.counter_id(m.RowModels("claude-opus-5", "claude-opus-5"),
                         "https://e", "t")
        self.assertIn(f"v{m.COUNTER_VERSION}", a)
        self.assertNotEqual(
            a, m.counter_id(m.RowModels("claude-opus-5", "claude-opus-5"),
                            "https://e", "other"))
        self.assertNotEqual(
            a, m.counter_id(m.RowModels("claude-haiku-4-5", "claude-haiku-4-5"),
                            "https://e", "t"))

    def test_a_model_the_endpoint_rejects_degrades_to_an_estimate(self):
        """Why there is no flag to force one tokenizer over the export.

        A row whose model the endpoint will not accept fails to count. It keeps
        its place in the export, carries no `segment_tokens`, and the run exits
        non-zero so nothing downstream mistakes it for a counted file — the
        analyzer estimates that row by byte share and says so. Forcing a model
        would instead have produced a number that is wrong and indistinguishable
        from an exact one.
        """
        r = self._run(self._export("model-this-gateway-rejects"))
        self.assertNotEqual(r.returncode, 0,
                            "a run that counted nothing reported success")
        rows = [json.loads(l) for l in open(self.out + ".partial") if l.strip()]
        self.assertEqual(len(rows), 1, "the row was dropped rather than kept")
        self.assertNotIn("segment_tokens", rows[0],
                         "a row the tokenizer refused was marked as counted")

    def test_the_run_reports_which_tokenizers_answered(self):
        """These counts load as exact, so which model produced them belongs in
        the run's own output and not in an operator's assumption."""
        r = self._run(self._export("claude-opus-5", "claude-haiku-4-5"))
        self.assertIn("claude-opus-5", r.stdout)
        self.assertIn("claude-haiku-4-5", r.stdout)

    def test_there_is_no_way_to_name_the_tokenizer_from_the_cli(self):
        """Two flags for this existed and both were removed on review.

        An override produces counts that are wrong *and* load as exact. A
        fallback is the same defect one step quieter — the analyzer calls a row
        with no resolvable model "unknown", so counting it with haiku attaches
        haiku's sizes to a row the report names otherwise. Asserted on behaviour
        rather than on help text: whatever the CLI is asked, no tokenizer the
        rows did not name ever answers.
        """
        src = self._export("claude-opus-5")
        for extra in (("--model", "claude-haiku-4-5"),
                      ("--force-model", "claude-haiku-4-5"),
                      ("--model", "claude-haiku-4-5", "--allow-partial")):
            with self.subTest(extra=extra):
                _ModelStub.seen = []
                self._run(src, *extra)
                self.assertNotIn("claude-haiku-4-5", set(_ModelStub.seen),
                                 f"{extra} made another tokenizer answer for a "
                                 f"row that named claude-opus-5")

    def test_a_blank_model_puts_nothing_on_the_wire(self):
        """The egress half of the blank-model finding, driven through the real
        command. Measured before the fix: 3 calls, model `'   '`, and the
        system prompt text in the request bodies."""
        p = os.path.join(self.dir, "blank.jsonl")
        with open(p, "w") as f:
            f.write(json.dumps({"body": {
                "model": "   ",
                "system": [{"type": "text", "text": "SECRET-POLICY"}],
                "messages": [{"role": "user", "content": "hi"}]}}) + "\n")
        r = self._run(p, "--allow-partial")
        self.assertEqual(_ModelStub.seen, [],
                         "a prompt body was sent under a blank model id")
        self.assertIn("no model id", r.stderr)

    def test_a_surface_prefixed_id_puts_nothing_on_the_wire(self):
        """The same, for the routing prefix. Without a --target-id there is no
        way to tell routing from model id, and the call could not have returned
        a usable count."""
        p = os.path.join(self.dir, "bedrock.jsonl")
        with open(p, "w") as f:
            f.write(json.dumps({"body": {
                "model": "anthropic.claude-opus-5",
                "system": [{"type": "text", "text": "SECRET-POLICY"}],
                "messages": [{"role": "user", "content": "hi"}]}}) + "\n")
        r = self._run(p, "--allow-partial")
        self.assertEqual(_ModelStub.seen, [],
                         "a prompt body was sent under a surface routing prefix")
        self.assertIn("--target-id", r.stderr)

    def test_naming_the_surface_makes_that_row_countable(self):
        """The other direction: the refusal must be about the missing surface,
        not a blanket refusal of prefixed ids."""
        p = os.path.join(self.dir, "bedrock2.jsonl")
        with open(p, "w") as f:
            f.write(json.dumps({"body": {
                "model": "anthropic.claude-opus-5",
                "messages": [{"role": "user", "content": "hi"}]}}) + "\n")
        r = self._run(p, "--target-id", "amazon-bedrock/converse")
        self.assertEqual(r.returncode, 0, r.stderr[-400:])
        self.assertEqual(set(_ModelStub.seen), {"claude-opus-5"},
                         "the prefix was not stripped before the tokenizer was "
                         "asked")

    def test_an_unknown_model_puts_nothing_on_the_wire(self):
        """The egress rule as a rule rather than a list of known-bad shapes.

        Blank models, routing prefixes and Bedrock ids were each closed by a
        separate review; an unknown model, an OpenAI id against an Anthropic
        endpoint, and a composite routed id still sent a body and learned it was
        hopeless from the remote error. Nothing goes out unless the id that
        would be sent is one the registry recognises.
        """
        for model in ("gpt-4o", "model-nobody-has-heard-of",
                      "bedrock/anthropic.not-a-real-model",
                      "claude-opus-5-but-with-a-typo"):
            with self.subTest(model=model):
                _ModelStub.seen = []
                p = os.path.join(self.dir, "u.jsonl")
                with open(p, "w") as f:
                    f.write(json.dumps({"body": {
                        "model": model,
                        "system": [{"type": "text", "text": "SECRET-POLICY"}],
                        "messages": [{"role": "user", "content": "hi"}]}}) + "\n")
                r = self._run(p, "--allow-partial")
                self.assertEqual(_ModelStub.seen, [],
                                 f"a prompt body was sent for {model!r}, whose "
                                 f"call could not have returned a usable count")
                self.assertIn("not a model this registry knows", r.stderr)

    def test_the_operator_can_assert_a_gateway_serves_an_unknown_id(self):
        """The override. A gateway with its own model names is a real case, and
        the rule must be escapable explicitly rather than by accident."""
        p = os.path.join(self.dir, "gw.jsonl")
        with open(p, "w") as f:
            f.write(json.dumps({"body": {
                "model": "claude-opus-5",
                "messages": [{"role": "user", "content": "hi"}]}}) + "\n")
        r = self._run(p, "--assume-endpoint-serves")
        self.assertEqual(r.returncode, 0, r.stderr[-300:])
        self.assertEqual(set(_ModelStub.seen), {"claude-opus-5"})

    def test_a_known_model_is_still_countable(self):
        """The other direction: the rule must not refuse the ordinary case."""
        r = self._run(self._export("claude-opus-5"))
        self.assertEqual(r.returncode, 0, r.stderr[-300:])
        self.assertEqual(set(_ModelStub.seen), {"claude-opus-5"})

    def test_a_row_with_no_resolvable_model_is_not_counted_at_all(self):
        """The consequence of having no fallback, stated as behaviour.

        The analyzer calls such a row "unknown". Counting it with anything else
        attaches that tokenizer's sizes to a row the report names otherwise, so
        it is not counted: it keeps its place, carries no `segment_tokens`, and
        the analyzer estimates it by byte share.
        """
        r = self._run(self._export(None))
        self.assertNotEqual(r.returncode, 0,
                            "a run that counted nothing reported success")
        self.assertEqual(_ModelStub.seen, [],
                         "a request was sent for a row with no model id; the "
                         "only answer it can have is an error")
        rows = [json.loads(l) for l in open(self.out + ".partial") if l.strip()]
        self.assertEqual(len(rows), 1)
        self.assertNotIn("segment_tokens", rows[0])

    def test_a_dry_run_names_the_tokenizers_it_would_ask(self):
        """A mixed export costs one set of prefix calls per model. The dry run
        is where permission for the egress is asked for, so it says so."""
        r = self._run(self._export("claude-opus-5", "claude-haiku-4-5"),
                      "--dry-run")
        self.assertEqual(r.returncode, 0, r.stderr[-400:])
        self.assertEqual(_ModelStub.seen, [], "a dry run sent something")
        self.assertIn("claude-opus-5", r.stdout)
        self.assertIn("claude-haiku-4-5", r.stdout)


class TestTheCounterAsksTheLoaderWhichModelThisIs(unittest.TestCase):
    """Counted-model and analysed-model must not be able to differ.

    Three rounds of review found three different divergences here, each in a
    shape the previous fix had not modelled — the CLI flag stamped over the row,
    then the top-level model ignored, then the order of operations: the loader
    picks the raw value first and coerces once, `_text(raw_body or raw_row,
    "unknown")`, while the counter coerced each candidate before choosing. Six
    shapes disagreed, and in every one `segment_tokens` is written by one model
    and then accepted as exact under another.

    Matching the loader's behaviour is what failed twice, so the counter now
    calls it. These tests are the differential: for a generated cross-product of
    row and body shapes, whatever `request_from_row` returns is what must have
    been counted. They compare against the real function rather than against a
    table, so a change inside the loader moves both sides together — which is
    the property a re-derivation could never have.
    """

    VALUES = (None, "claude-opus-5", "claude-haiku-4-5",
              "claude-opus-5-20260101", "anthropic/claude-opus-5",
              "", "   ", 0, 5, 1.5, True, False,
              ["claude-opus-5"], {"id": "claude-opus-5"})

    def _cases(self):
        for row_v in self.VALUES:
            for body_v in self.VALUES:
                body = {"messages": [{"role": "user", "content": "hi"}]}
                if body_v is not None:
                    body["model"] = body_v
                row = {"body": body}
                if row_v is not None:
                    row["model"] = row_v
                yield row_v, body_v, row, body

    def test_the_resolved_model_is_always_the_loaders(self):
        from cacheeconomics.trace import request_from_row
        m = load("count_tokens")
        checked = 0
        for row_v, body_v, row, body in self._cases():
            loader = request_from_row(
                row, [], renamed={},
                model_override=(body or {}).get("model")).model
            with self.subTest(row=row_v, body=body_v):
                self.assertEqual(m.row_models(row, body).analysis, loader)
            checked += 1
        self.assertEqual(checked, len(self.VALUES) ** 2)

    def test_the_tokenizer_model_is_the_logged_id_minus_routing(self):
        """The other half, and the round-4 and round-5 findings together.

        `request_from_row` returns the normalised id — snapshot date stripped
        AND surface prefix stripped — which conflates two decisions. The date
        matters to a tokenizer (if the bare alias has moved, only the dated id
        still means what the log meant) and the routing prefix does not (nothing
        answers to `anthropic.claude-opus-5`).

        So the tokenizer id is the raw resolved value with routing removed and
        nothing else: whatever it is, it must be a *suffix* of the raw id, since
        only a leading prefix may be dropped.
        """
        from cacheeconomics.trace import _first, _text
        m = load("count_tokens")
        for _rv, _bv, row, body in self._cases():
            raw = _text((body or {}).get("model") or _first(row, "model"))
            raw = raw.strip() if isinstance(raw, str) else raw
            got = m.row_models(row, body).tokenizer
            with self.subTest(row=row.get("model"), body=body.get("model")):
                if got is None:
                    continue                 # unnamed, or routing we cannot resolve
                self.assertTrue(
                    raw.endswith(got),
                    f"{got!r} is not {raw!r} with only a leading prefix removed")

    def test_the_date_survives_and_the_prefix_does_not(self):
        """The two strippings, separated. `_normalised` does both; only one of
        them is right for a tokenizer."""
        m = load("count_tokens")
        dated = m.row_models({}, {"model": "claude-opus-5-20260101"}).tokenizer
        self.assertEqual(dated, "claude-opus-5-20260101")
        prefixed = m.row_models({}, {"model": "anthropic.claude-opus-5"},
                                "amazon-bedrock/converse").tokenizer
        self.assertEqual(prefixed, "claude-opus-5")

    def test_a_dated_model_is_counted_dated_and_analysed_bare(self):
        """The two answers, side by side, on the shape that produced the
        finding. Asserting they *differ* here is the point: a version of this
        that returns one value for both cannot pass."""
        m = load("count_tokens")
        body = {"model": "claude-opus-5-20260101",
                "messages": [{"role": "user", "content": "hi"}]}
        got = m.row_models({}, body)
        self.assertEqual(got.tokenizer, "claude-opus-5-20260101")
        self.assertEqual(got.analysis, "claude-opus-5")
        self.assertNotEqual(got.tokenizer, got.analysis)

    def test_the_dated_id_is_what_reaches_the_endpoint(self):
        """Through the real command, because the split is only worth anything if
        the raw id is what actually goes on the wire."""
        stub_seen = []

        class _Rec(_ModelStub):
            pass

        _Rec.seen = stub_seen
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Rec)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        d = tempfile.mkdtemp()
        src = os.path.join(d, "in.jsonl")
        with open(src, "w") as f:
            f.write(json.dumps({"body": {
                "model": "claude-opus-5-20260101",
                "messages": [{"role": "user", "content": "hi"}]}}) + "\n")
        subprocess.run(
            [sys.executable, "-B", os.path.join(TIER_B, "count_tokens.py"), src,
             "-o", os.path.join(d, "out.jsonl"), "--allow-partial",
             "--endpoint", f"http://127.0.0.1:{srv.server_address[1]}/count"],
            capture_output=True, text=True, timeout=60,
            env=dict(os.environ, ANTHROPIC_API_KEY="test"))
        self.assertEqual(set(stub_seen), {"claude-opus-5-20260101"},
                         "the normalised id was sent to the tokenizer")

    def test_the_cases_actually_disagree_with_a_naive_resolver(self):
        """Guards the guard. If every generated case resolved the same way under
        any reasonable rule, the differential above would pass without
        discriminating — so this pins that the cross-product really does contain
        the shapes that broke it, by scoring the resolver the last round used.
        """
        from cacheeconomics.trace import _first, _text, request_from_row
        disagreements = 0
        for _rv, _bv, row, body in self._cases():
            loader = request_from_row(
                row, [], renamed={},
                model_override=(body or {}).get("model")).model
            naive = (_text(_first(body, "model")) or _text(_first(row, "model"))
                     or "claude-haiku-4-5")
            if naive != loader:
                disagreements += 1
        self.assertGreater(disagreements, 10,
                           "the cross-product no longer contains the shapes "
                           "that broke this, so the differential proves little")

    def test_a_row_naming_nothing_has_no_tokenizer_to_ask(self):
        """No fallback. The analyzer calls such a row "unknown", and there is no
        id anyone could send, so it is not counted at all — not counted with a
        default, and not sent as the literal string "unknown" either."""
        m = load("count_tokens")
        got = m.row_models({}, {"messages": []})
        self.assertIsNone(got.tokenizer)
        self.assertEqual(got.analysis, "unknown")

    def test_a_row_that_is_not_a_dict_does_not_take_the_run_down(self):
        m = load("count_tokens")
        for bad_row in (None, [], "not-a-row", 7):
            with self.subTest(row=bad_row):
                got = m.row_models(bad_row, {"messages": []})
                self.assertIsNone(got.tokenizer)
                self.assertEqual(got.analysis, "unknown")

    def test_the_override_argument_is_still_the_one_the_adapter_passes(self):
        """The single line still transcribed from the caller.

        `row_models` reproduces `bodies.load_bodies`'s `model_override=`
        argument because that is the caller's choice rather than the resolver's
        logic. It is the last place a divergence can hide, so it is read back
        out of the adapter's own source.
        """
        import re
        src = open(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "cacheeconomics", "adapters", "bodies.py")).read()
        found = re.findall(r"model_override=([^,]+),", src)
        self.assertEqual(found, ['(body or {}).get("model")'],
                         "the adapter changed how it picks the override; "
                         "count_tokens.row_models must change with it")

    def test_the_target_id_reaches_the_resolution(self):
        """`--target-id` is an input to the resolved model: a surface's id
        prefix is stripped before the model is looked up. Counting and analysis
        have to be given the same one or they resolve different models from the
        same row."""
        m = load("count_tokens")
        body = {"model": "anthropic.claude-opus-5",
                "messages": [{"role": "user", "content": "hi"}]}
        bare = m.row_models({}, body).analysis
        with_surface = m.row_models({}, body, "amazon-bedrock/converse").analysis
        self.assertEqual(with_surface, "claude-opus-5")
        self.assertNotEqual(bare, with_surface,
                            "the surface made no difference, so this test no "
                            "longer covers the axis it was written for")

    def test_the_surface_prefix_is_stripped_from_the_tokenizer_id(self):
        """There are three model-shaped things, not two.

        Round 4 stopped before `_normalised` to keep the snapshot date, which
        was right, and thereby also kept the surface's *routing* prefix, which
        was wrong: `anthropic.` is how Bedrock addresses a model, not an id any
        tokenizer answers to. With --target-id amazon-bedrock/converse the whole
        prompt body was going to api.anthropic.com under
        `anthropic.claude-opus-5` — a call that cannot come back with a usable
        count, so egress for nothing.

        Date kept, prefix dropped, and the prefix recorded rather than lost.
        """
        m = load("count_tokens")
        for raw, target, tok, prefix in (
                ("anthropic.claude-opus-5", "amazon-bedrock/converse",
                 "claude-opus-5", "anthropic."),
                ("anthropic.claude-opus-5-20260101", "amazon-bedrock/converse",
                 "claude-opus-5-20260101", "anthropic."),
                ("bedrock/anthropic.claude-haiku-4-5", "amazon-bedrock/converse",
                 "claude-haiku-4-5", "bedrock/anthropic."),
                ("anthropic/claude-opus-5", None, "claude-opus-5", "anthropic/"),
                ("claude-opus-5-20260101", None, "claude-opus-5-20260101", None),
                ("claude-opus-5", None, "claude-opus-5", None)):
            with self.subTest(raw=raw, target=target):
                got = m.row_models({}, {"model": raw, "messages": []}, target)
                self.assertEqual(got.tokenizer, tok)
                self.assertEqual(got.prefix, prefix)

    def test_a_surface_prefixed_id_is_not_sent_when_no_surface_is_named(self):
        """`normalize_model` needs the target to know what the prefix is, so
        without one this cannot tell routing from model id — and guessing costs
        a prompt body on the wire. It refuses instead, and names the prefix so
        the message can say what to pass."""
        m = load("count_tokens")
        got = m.row_models({}, {"model": "anthropic.claude-opus-5",
                                "messages": []})
        self.assertIsNone(got.tokenizer)
        self.assertEqual(got.prefix, "anthropic.")

    def test_the_known_prefixes_come_from_the_registry(self):
        """Listing them here would go stale the first time a surface is added."""
        from cacheeconomics.tokenizer import _known_routing_prefixes
        self.assertIn("anthropic.", _known_routing_prefixes())

    def test_a_blank_model_is_unnamed_rather_than_a_model_named_blank(self):
        """`_text` returns strings unchanged, so `"   "` passed the truthiness
        test and a full prompt body went to the endpoint under a model name of
        three spaces. Trimming to decide whether a model exists is not the
        coercion round 2 rejected — `5` still resolves to "5"."""
        m = load("count_tokens")
        for blank in ("   ", "\t", " \n ", ""):
            with self.subTest(value=blank):
                self.assertIsNone(
                    m.row_models({}, {"model": blank, "messages": []}).tokenizer)
        self.assertEqual(
            m.row_models({}, {"model": 5, "messages": []}).tokenizer, "5",
            "a numeric model was coerced away; round 2 established it must not "
            "be")
        self.assertEqual(
            m.row_models({}, {"model": "  claude-opus-5 ",
                              "messages": []}).tokenizer, "claude-opus-5")


class TestEveryPathThisToolchainDerivesIsIgnored(unittest.TestCase):
    """The operator chooses the input path and `--out`. They do not choose the
    names this toolchain derives from either one, so keeping those out of a
    commit is our job.

    This class used to be `TestEverySuffixWeDeriveFromOutIsIgnored` and read the
    suffixes `count_tokens.py` appends to `--out` out of its source. That is the
    right pattern and it was half the surface. `run_diagnostic.py` derives the
    counted export from the *input* path instead (`_counted_path`: `run.jsonl`
    -> `run-counted.jsonl`), `sweep_report.py` does the same, and nothing
    ignored the result: `git check-ignore -v tier-b/trace-counted.jsonl` matched
    no rule, on a file that is the enriched export with the client's request
    bodies intact. Same defect class as the three suffixes before it, derived
    from the other end, and a test scoped to `--out` could not see it.

    So the surface is now both directions and their composition — an `--out`
    suffix appended to an input-derived name is itself a derived name — plus a
    last test that discovers members by *running* the toolchain and looking at
    what it left on disk. A name derived by some fourth route none of the above
    models fails there instead of in somebody's repository.
    """

    INPUT_SHAPES = ("run.jsonl", "capture", "run.jsonl.gz", "a.jsonl/b.jsonl",
                    "my.jsonl.backup.jsonl", ".jsonl", "no-ext-at-all",
                    "traces/2026-07.jsonl")

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.root = os.path.dirname(TIER_B)

    def _is_ignored(self, relpath):
        """Would git stage this file if it appeared inside a checkout?

        Probed at a path under `tier-b/`, because that is where an operator
        working from a checkout puts a capture and therefore where the derived
        names land — and because some of the existing rules are anchored there.
        """
        probe = os.path.join(TIER_B, relpath)
        r = subprocess.run(["git", "check-ignore", "-q", probe],
                           cwd=self.root, capture_output=True)
        return r.returncode == 0

    def _out_suffixes(self):
        """Every suffix `count_tokens.py` appends to the operator's `--out`."""
        import re
        src = open(os.path.join(TIER_B, "count_tokens.py")).read()
        return sorted(set(re.findall(r'args\.out \+ "([^"]+)"', src)))

    def _input_derived(self):
        """Every name the toolchain derives from an *input* path."""
        derive = load("count_tokens").counted_path
        return sorted({derive(p) for p in self.INPUT_SHAPES})

    def test_the_source_still_derives_what_we_think_it_does(self):
        """Guards the guard: if either derivation stops looking like this, the
        tests below silently check nothing."""
        found = self._out_suffixes()
        self.assertTrue(found, "no --out suffixes found; the pattern moved")
        self.assertIn(".partial", found)
        self.assertIn(".cache.json", found)
        derived = self._input_derived()
        self.assertTrue(derived, "no input-derived names found")
        self.assertTrue(all(d != p for d, p in zip(derived, self.INPUT_SHAPES)))

    def test_each_out_suffix_is_ignored(self):
        for suffix in self._out_suffixes():
            with self.subTest(suffix=suffix):
                self.assertTrue(
                    self._is_ignored("some-run.jsonl" + suffix),
                    f"'{suffix}' is written beside the operator's --out and "
                    f"git would stage it")

    def test_each_input_derived_name_is_ignored(self):
        for name in self._input_derived():
            with self.subTest(name=name):
                self.assertTrue(
                    self._is_ignored(name),
                    f"'{name}' is derived from the operator's input path, "
                    f"holds the enriched export, and git would stage it")

    def test_the_composition_of_the_two_is_ignored(self):
        """`run_diagnostic.py` hands its derived path to `count_tokens.py` as
        `--out`, so every suffix lands on every input-derived name. Covering
        each direction alone would leave the product unchecked."""
        for name in self._input_derived():
            for suffix in self._out_suffixes():
                with self.subTest(name=name + suffix):
                    self.assertTrue(self._is_ignored(name + suffix),
                                    f"'{name + suffix}' would be staged")

    def test_a_failed_count_writes_prompt_text_into_one_of_them(self):
        """The reason this matters: the file is not incidental, it carries the
        bodies. If a future change stops it doing so, this test should be
        revisited rather than the ignore rule quietly kept."""
        src = os.path.join(self.dir, "in.jsonl")
        with open(src, "w") as f:
            f.write(json.dumps({"request": {
                "model": "claude-opus-5",
                "system": [{"type": "text", "text": "SECRET-POLICY"}],
                "messages": [{"role": "user", "content": "hi"}]}}) + "\n")
        out = os.path.join(self.dir, "out.jsonl")
        subprocess.run(
            [sys.executable, "-B", os.path.join(TIER_B, "count_tokens.py"),
             src, "--out", out, "--endpoint", "http://127.0.0.1:1/x"],
            capture_output=True, text=True, timeout=60,
            env=dict(os.environ, ANTHROPIC_API_KEY="test"))
        partial = out + ".partial"
        self.assertTrue(os.path.exists(partial), "expected a partial export")
        self.assertIn("SECRET-POLICY", open(partial).read())

    def test_the_input_derived_export_carries_prompt_text_too(self):
        """The same check for the other end. `-counted.jsonl` is the enriched
        export, not a summary, and it is the file nothing ignored."""
        src = os.path.join(self.dir, "trace.jsonl")
        with open(src, "w") as f:
            f.write(json.dumps({"request": {
                "model": "claude-opus-5",
                "system": [{"type": "text", "text": "SECRET-POLICY"}],
                "messages": [{"role": "user", "content": "hi"}]}}) + "\n")
        stub = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _CountStub)
        threading.Thread(target=stub.serve_forever, daemon=True).start()
        self.addCleanup(stub.shutdown)
        subprocess.run(
            [sys.executable, "-B", os.path.join(TIER_B, "run_diagnostic.py"),
             src, "--endpoint",
             f"http://127.0.0.1:{stub.server_address[1]}/v1/messages/count_tokens",
             "--allow-unreconciled"],
            capture_output=True, text=True, timeout=120,
            env=dict(os.environ, ANTHROPIC_API_KEY="test",
                     CACHEECONOMICS_HMAC_KEY="k" * 32))
        counted = os.path.join(self.dir, "trace-counted.jsonl")
        self.assertTrue(os.path.exists(counted), "no counted export was written")
        self.assertIn("SECRET-POLICY", open(counted).read())

    def test_everything_a_real_run_leaves_on_disk_is_ignored(self):
        """Members discovered by running the thing, not by modelling it.

        Both paths, because they leave different files behind: a run whose
        counting succeeds and one whose endpoint is dead. Anything in the
        directory afterwards that the operator did not name is ours, and has to
        be unstageable.
        """
        src = os.path.join(self.dir, "capture.jsonl")
        with open(src, "w") as f:
            for i in range(3):
                f.write(json.dumps({
                    "sent_at": f"2026-07-29T09:0{i}:00Z",
                    "body": {"model": "claude-opus-5",
                             "system": [{"type": "text", "text": "s" * 800}],
                             "messages": [{"role": "user", "content": "hi"}]},
                    "usage": {"input_tokens": 100,
                              "cache_read_input_tokens": 0,
                              "cache_creation_input_tokens": 0}}) + "\n")
        stub = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _CountStub)
        threading.Thread(target=stub.serve_forever, daemon=True).start()
        self.addCleanup(stub.shutdown)
        live = f"http://127.0.0.1:{stub.server_address[1]}/v1/messages/count_tokens"
        for endpoint in ("http://127.0.0.1:1/x", live):
            subprocess.run(
                [sys.executable, "-B", os.path.join(TIER_B, "run_diagnostic.py"),
                 src, "--endpoint", endpoint, "--allow-unreconciled"],
                capture_output=True, text=True, timeout=120,
                env=dict(os.environ, ANTHROPIC_API_KEY="test",
                         CACHEECONOMICS_HMAC_KEY="k" * 32))

        left = set()
        for dirpath, _dirs, files in os.walk(self.dir):
            for name in files:
                left.add(os.path.relpath(os.path.join(dirpath, name), self.dir))
        self.assertTrue(left - {"capture.jsonl"},
                        "the run produced nothing; this test proved nothing")
        for name in sorted(left - {"capture.jsonl"}):
            with self.subTest(name=name):
                self.assertTrue(
                    self._is_ignored(name),
                    f"a real run left '{name}' beside the capture. The operator "
                    f"named 'capture.jsonl' and nothing else, and git would "
                    f"stage this one")


class TestCountedRowsSayWhatProducedThem(unittest.TestCase):
    """A counted row used to carry `segment_tokens` and nothing else.

    The loader accepts any correctly-shaped positive array as exact, so a
    counted export left over from a different endpoint, a different resolved
    model, an older version of the counter, or a capture that has since been
    re-recorded was indistinguishable from a fresh one — and
    `sweep_report.counted` reused it on the strength of the filename existing.
    Deleting the flag that produced some of those files closed the door and left
    the window open.

    So every counted row now records what produced it, and the record covers
    every input that changes a count.
    """

    def setUp(self):
        _ModelStub.seen = []
        self.stub = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ModelStub)
        self.port = self.stub.server_address[1]
        threading.Thread(target=self.stub.serve_forever, daemon=True).start()
        self.addCleanup(self.stub.shutdown)
        self.dir = tempfile.mkdtemp()
        self.endpoint = f"http://127.0.0.1:{self.port}/count"

    def _src(self, *models):
        p = os.path.join(self.dir, "cap.jsonl")
        with open(p, "w") as f:
            for mdl in models:
                f.write(json.dumps({
                    "sent_at": "2026-08-01T09:00:00Z",
                    "body": {"model": mdl,
                             "system": [{"type": "text", "text": "policy " * 50}],
                             "messages": [{"role": "user", "content": "hi"}]},
                    "usage": {"input_tokens": 500}}) + "\n")
        return p

    def _count(self, src, out, *extra):
        return subprocess.run(
            [sys.executable, "-B", os.path.join(TIER_B, "count_tokens.py"), src,
             "-o", out, "--endpoint", self.endpoint, *extra],
            capture_output=True, text=True, timeout=90,
            env=dict(os.environ, ANTHROPIC_API_KEY="test"))

    def test_a_counted_row_records_what_produced_it(self):
        m = load("count_tokens")
        src = self._src("claude-opus-5")
        out = os.path.join(self.dir, "out.jsonl")
        r = self._count(src, out)
        self.assertEqual(r.returncode, 0, r.stderr[-400:])
        row = json.loads(open(out).read().strip())
        p = row[m.PROVENANCE_KEY]
        self.assertEqual(p["tokenizer_model"], "claude-opus-5")
        self.assertEqual(p["analysis_model"], "claude-opus-5")
        self.assertEqual(p["endpoint"], self.endpoint)
        self.assertEqual(p["version"], m.COUNTER_VERSION)
        from cacheeconomics.tokenizer import body_sha256
        self.assertEqual(p["body_sha256"], body_sha256(row["body"]))
        self.assertIsNone(p["target_id"])
        self.assertIsNone(p["tokenizer_id"])

    def test_the_record_names_both_models_when_they_differ(self):
        """The round-4 split, recorded. One field could only ever answer one of
        the two questions, and a reader checking freshness may need either."""
        m = load("count_tokens")
        src = self._src("claude-opus-5-20260101")
        out = os.path.join(self.dir, "out.jsonl")
        self._count(src, out)
        p = json.loads(open(out).read().strip())[m.PROVENANCE_KEY]
        self.assertEqual(p["tokenizer_model"], "claude-opus-5-20260101")
        self.assertEqual(p["analysis_model"], "claude-opus-5")

    def test_the_record_covers_every_input_that_changes_a_count(self):
        """Named as a set rather than field by field, so a new input to counting
        that is not recorded fails here."""
        m = load("count_tokens")
        src = self._src("claude-opus-5")
        out = os.path.join(self.dir, "out.jsonl")
        self._count(src, out)
        row = json.loads(open(out).read().strip())
        self.assertEqual(
            set(row[m.PROVENANCE_KEY]),
            {"version", "tool", "row_sha256", "body_sha256", "cuts_sha256",
             "tokenizer_model", "analysis_model", "endpoint", "target_id",
             "tokenizer_id"})

    def test_the_row_digest_covers_what_the_body_digest_misses(self):
        """The body digest is blind to the top-level model, and to usage,
        timestamps, status and session. For the shape where the model sits on
        the row, changing it left the body digest identical."""
        m = load("count_tokens")
        base = {"sent_at": "2026-08-01T09:00:00Z", "model": "claude-opus-5",
                "body": {"messages": [{"role": "user", "content": "hi"}]},
                "usage": {"input_tokens": 500}}
        changed = json.loads(json.dumps(base))
        changed["model"] = "claude-haiku-4-5"
        from cacheeconomics.tokenizer import body_sha256, row_sha256
        self.assertEqual(body_sha256(base["body"]), body_sha256(changed["body"]),
                         "the fixture no longer isolates the row from the body")
        self.assertNotEqual(row_sha256(base), row_sha256(changed))
        for field, value in (("usage", {"input_tokens": 1}),
                             ("sent_at", "2026-08-02T09:00:00Z"),
                             ("status", 500), ("session", "other")):
            other = json.loads(json.dumps(base))
            other[field] = value
            with self.subTest(field=field):
                self.assertNotEqual(row_sha256(base), row_sha256(other))

    def test_it_carries_no_prompt_text(self):
        """The same rule the count cache was changed for: this lands on a
        client's disk, and structure plus digests are enough."""
        m = load("count_tokens")
        p = os.path.join(self.dir, "secret.jsonl")
        with open(p, "w") as f:
            f.write(json.dumps({"body": {
                "model": "claude-opus-5",
                "system": [{"type": "text", "text": "SECRET-POLICY"}],
                "messages": [{"role": "user", "content": "CONFIDENTIAL"}]}}) + "\n")
        out = os.path.join(self.dir, "out.jsonl")
        self._count(p, out)
        row = json.loads(open(out).read().strip())
        blob = json.dumps(row[m.PROVENANCE_KEY])
        self.assertNotIn("SECRET-POLICY", blob)
        self.assertNotIn("CONFIDENTIAL", blob)

    def test_an_uncounted_row_carries_no_record(self):
        """Counts and their provenance are written together or not at all. A row
        with a record and no counts, or counts and no record, is what a stale
        file looks like."""
        m = load("count_tokens")
        src = self._src("model-this-gateway-rejects")
        out = os.path.join(self.dir, "out.jsonl")
        self._count(src, out)
        row = json.loads(open(out + ".partial").read().strip())
        self.assertNotIn("segment_tokens", row)
        self.assertNotIn(m.PROVENANCE_KEY, row)

    def test_every_counted_row_in_a_mixed_export_records_its_own_model(self):
        m = load("count_tokens")
        src = self._src("claude-opus-5", "claude-haiku-4-5")
        out = os.path.join(self.dir, "out.jsonl")
        self._count(src, out)
        rows = [json.loads(l) for l in open(out) if l.strip()]
        self.assertEqual([r[m.PROVENANCE_KEY]["tokenizer_model"] for r in rows],
                         ["claude-opus-5", "claude-haiku-4-5"])

    def test_the_record_does_not_disturb_the_analyzer(self):
        """An extra key on the row must stay inert: `_find_body` must not mistake
        it for a body and the counts must still load as exact."""
        from cacheeconomics.adapters.bodies import _find_body
        m = load("count_tokens")
        src = self._src("claude-opus-5")
        out = os.path.join(self.dir, "out.jsonl")
        self._count(src, out)
        row = json.loads(open(out).read().strip())
        self.assertIs(_find_body(row), row["body"])
        self.assertNotIn("messages", row[m.PROVENANCE_KEY])


class TestASweepWillNotReuseACountedFileItCannotVouchFor(unittest.TestCase):
    """`sweep_report.counted` returned the sibling counted file because it
    existed. Nothing compared it to the capture beside it.

    So a stale `interval-*-counted.jsonl` — left by a removed `--force-model`
    path, by a different endpoint, or for a capture that has since been
    re-recorded — skipped `count_tokens.py` entirely and was analysed as exact.

    On a mismatch this refuses rather than recounting. Recounting would send
    that capture's prompt prefixes to the endpoint because a file on disk
    disagreed: new egress, chosen by the tool, on a path an operator approved
    for a different question. Refusing sends nothing, believes nothing, and says
    what it found.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.m = load("sweep_report")
        self.ct = load("count_tokens")
        self.src = os.path.join(self.dir, "interval-10m.jsonl")
        with open(self.src, "w") as f:
            f.write(json.dumps({"body": {
                "model": "claude-opus-5",
                "messages": [{"role": "user", "content": "hi"}]}}) + "\n")
        self.out = self.ct.counted_path(self.src)

    def _write_counted(self, extra_row=None, **overrides):
        """A counted export for `self.src`, with its provenance adjustable."""
        src_row = json.loads(open(self.src).read().strip())
        row = json.loads(json.dumps(src_row))
        row["segment_tokens"] = [7]
        prov = self.ct.provenance(src_row, src_row["body"],
                                  self.ct.DEFAULT_ENDPOINT, None, None)
        prov.update(overrides)
        row[self.ct.PROVENANCE_KEY] = prov
        if extra_row:
            row.update(extra_row)
        with open(self.out, "w") as f:
            f.write(json.dumps(row) + "\n")

    def _counted_never_shelling_out(self, **kw):
        """`counted()` with the subprocess barred, so a test can never both
        assert a refusal and quietly send the capture somewhere."""
        attempts = []
        real = self.m.subprocess.run

        class _NoRun:
            @staticmethod
            def run(cmd, *a, **k):
                attempts.append(cmd)
                raise AssertionError("counted() shelled out")

        self.m.subprocess = _NoRun
        self.addCleanup(setattr, self.m.subprocess, "run", real)
        return self.m.counted(self.src, **kw), attempts

    def test_a_matching_counted_file_is_reused(self):
        """The other direction. A rule that never reuses has traded a wrong
        answer for egress on every run, which is its own kind of wrong."""
        self._write_counted(tokenizer_id="stub-1")
        got, attempts = self._counted_never_shelling_out(tokenizer_id="stub-1")
        self.assertEqual(got, self.out)
        self.assertEqual(attempts, [])

    def test_a_counted_file_with_no_tokenizer_identity_is_not_reused(self):
        """A counted export IS a cache — of whole rows rather than prefixes —
        and it was the one of the two reuse paths the identity rule had not been
        applied to. `count_tokens.py` refuses to resume its prefix cache without
        `--tokenizer-id`; reusing a counted export written without one is the
        same claim with nothing behind it, and `None == None` passed.
        """
        self._write_counted()                       # written with no identity
        got, attempts = self._counted_never_shelling_out()   # and none asked
        self.assertEqual(got, self.src,
                         "a counted export that names no tokenizer deployment "
                         "was reused as exact")
        self.assertEqual(attempts, [], "it recounted instead of refusing")

    def test_a_stored_identity_with_none_requested_is_not_reused(self):
        self._write_counted(tokenizer_id="stub-1")
        got, _ = self._counted_never_shelling_out()
        self.assertEqual(got, self.src)

    def test_a_requested_identity_with_none_stored_is_not_reused(self):
        self._write_counted()
        got, _ = self._counted_never_shelling_out(tokenizer_id="stub-1")
        self.assertEqual(got, self.src)

    def test_a_changed_capture_is_not_reused(self):
        self._write_counted()
        with open(self.src, "w") as f:
            f.write(json.dumps({"body": {
                "model": "claude-opus-5",
                "messages": [{"role": "user", "content": "DIFFERENT"}]}}) + "\n")
        got, attempts = self._counted_never_shelling_out()
        self.assertEqual(got, self.src, "a stale counted export was reused")
        self.assertEqual(attempts, [], "it recounted instead of refusing")

    def test_a_different_endpoint_is_not_reused(self):
        self._write_counted(endpoint="https://someone-elses-gateway/count")
        got, _ = self._counted_never_shelling_out()
        self.assertEqual(got, self.src)

    def test_an_older_counter_version_is_not_reused(self):
        self._write_counted(version=self.ct.COUNTER_VERSION - 1)
        got, _ = self._counted_never_shelling_out()
        self.assertEqual(got, self.src)

    def test_a_different_surface_is_not_reused(self):
        self._write_counted()
        got, _ = self._counted_never_shelling_out(
            target_id="amazon-bedrock/converse")
        self.assertEqual(got, self.src)

    def test_counts_with_no_record_at_all_are_not_reused(self):
        """The files the old code produced. Every one of them is on disk
        somewhere already, and this is the case that matters most."""
        row = json.loads(open(self.src).read().strip())
        row["segment_tokens"] = [7]
        with open(self.out, "w") as f:
            f.write(json.dumps(row) + "\n")
        got, _ = self._counted_never_shelling_out()
        self.assertEqual(got, self.src,
                         "a counted export from before provenance existed was "
                         "reused as exact")

    def test_a_truncated_counted_file_is_not_reused(self):
        self._write_counted()
        with open(self.out, "w") as f:
            f.write("")
        got, _ = self._counted_never_shelling_out()
        self.assertEqual(got, self.src)

    def test_a_refusal_says_what_it_found_and_what_to_do(self):
        self._write_counted(tokenizer_id="stub-1",
                            endpoint="https://someone-elses-gateway/count")
        buf = io.StringIO()
        real, sys.stderr = sys.stderr, buf
        try:
            self.m.counted(self.src, tokenizer_id="stub-1")
        finally:
            sys.stderr = real
        said = buf.getvalue()
        self.assertIn("refusing to reuse", said)
        self.assertIn("someone-elses-gateway", said)
        self.assertIn("estimated", said)
        self.assertIn("delete", said)

    def test_a_row_that_was_never_counted_does_not_make_the_file_stale(self):
        """A partial export is a legitimate shape: rows the endpoint refused
        carry no counts and no record, and they are not evidence of staleness."""
        row = json.loads(open(self.src).read().strip())
        with open(self.out, "w") as f:
            f.write(json.dumps(row) + "\n")
        _merged, why = self.m.reusable_counts(self.src, self.out)
        self.assertIsNone(why)

    def test_a_changed_top_level_model_is_not_reused(self):
        """The gap the body digest could not see, and the shape round 3 was
        spent fixing: the model sits on the row, so changing it leaves the body
        byte-identical while the tokenizer that should answer changes."""
        self._write_counted()
        src_row = json.loads(open(self.src).read().strip())
        from cacheeconomics.tokenizer import body_sha256
        before = body_sha256(src_row["body"])
        src_row["model"] = "claude-haiku-4-5"
        del src_row["body"]["model"]
        with open(self.src, "w") as f:
            f.write(json.dumps(src_row) + "\n")
        self._write_counted()          # counted under the top-level haiku
        src_row["model"] = "claude-opus-5"
        with open(self.src, "w") as f:
            f.write(json.dumps(src_row) + "\n")
        self.assertEqual(before, body_sha256(src_row["body"]) if
                         src_row["body"].get("model") else before)
        got, attempts = self._counted_never_shelling_out()
        self.assertEqual(got, self.src,
                         "a file counted under the row's old model was reused")
        self.assertEqual(attempts, [])

    def test_changed_usage_on_the_row_is_not_reused(self):
        """The body digest is blind to it, and `_scale_to_measured` divides the
        billed total the usage reports."""
        self._write_counted()
        src_row = json.loads(open(self.src).read().strip())
        src_row["usage"] = {"input_tokens": 999}
        with open(self.src, "w") as f:
            f.write(json.dumps(src_row) + "\n")
        got, _ = self._counted_never_shelling_out()
        self.assertEqual(got, self.src)

    def test_a_different_tokenizer_id_is_not_reused(self):
        self._write_counted(tokenizer_id="gateway-build-41")
        got, _ = self._counted_never_shelling_out(tokenizer_id="gateway-build-42")
        self.assertEqual(got, self.src)

    def test_what_is_reused_is_the_counts_and_not_the_stored_row(self):
        """The counted file contributes `segment_tokens` and its record, never
        the rest of the row.

        Returning the stored file wholesale meant it supplied usage, timestamps,
        status and session too, and only the body digest was ever checked. Here
        the stored row carries a field that contradicts the capture; after reuse
        it must be gone, because the row came from the capture.
        """
        self._write_counted(tokenizer_id="stub-1",
                            extra_row={"usage": {"input_tokens": 123456},
                                       "agent": "from-the-stale-file"})
        got, _ = self._counted_never_shelling_out(tokenizer_id="stub-1")
        self.assertEqual(got, self.out)
        row = json.loads(open(self.out).read().strip())
        self.assertEqual(row["segment_tokens"], [7], "the counts were not kept")
        self.assertNotIn("agent", row,
                         "a field from the stored file survived into the "
                         "analysed rows")
        self.assertNotIn("usage", row)


class TestTheSweepCanBeToldWhereItsTrafficWent(unittest.TestCase):
    """`sweep_report.main()` exposed only `--dir`.

    `counted()` and `analyse()` both take a surface and an endpoint, and `main`
    called both with defaults — so a Bedrock, Vertex or gateway sweep could not
    strip the surface's model prefix, could not send its counting calls anywhere
    but the default host, and refused as stale any counted file correctly
    produced with `--target-id`, because this path passed None. Removing the
    `--model` flag left those exports no route through the sweep at all.

    Asserted by driving `main()` and recording what reaches the two callees,
    rather than by reading `--help`: the defect was that the values never
    arrived, and a flag that parses and goes nowhere passes a help-text check.
    """

    def setUp(self):
        self.m = load("sweep_report")
        self.dir = tempfile.mkdtemp()
        cap = os.path.join(self.dir, "interval-10m.jsonl")
        with open(cap, "w") as f:
            for i in range(3):
                f.write(json.dumps({
                    "sent_at": f"2026-08-01T09:{i * 10:02d}:00+00:00",
                    "session": "s1",
                    "body": {"model": "anthropic.claude-opus-5",
                             "messages": [{"role": "user", "content": "hi"}]},
                    "usage": {"input_tokens": 100}}) + "\n")
        self.cap = cap

    def _run_main(self, *argv):
        """`main()` with the real `counted()`, and `subprocess.run` recorded.

        The first version of this replaced `counted()` with a fake, so it proved
        only that `main` passed parsed values to an in-process function. Deleting
        the flag appends *inside* `counted()` -- the code that builds the
        `count_tokens.py` command line that actually sends prompt content -- left
        all of it green, measured at 18 passed. So the boundary recorded here is
        the subprocess, which is where the values have to arrive.

        `analyse` is still replaced: it shells out to the analyzer, which is a
        different subprocess with its own command line, captured separately.
        """
        seen = {"cmds": [], "analyse": None}

        def fake_analyse(path, *a, **k):
            seen["analyse"] = (path, a, k)
            return {"ttl1_raised": False, "recoverable_share": None,
                    "measured_usd": 0.0, "window_days": 1}

        class _Rec:
            @staticmethod
            def run(cmd, *a, **k):
                seen["cmds"].append(list(cmd))
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        real = (self.m.analyse, self.m.subprocess, sys.argv)
        self.m.analyse, self.m.subprocess = fake_analyse, _Rec
        sys.argv = ["sweep_report.py", "--dir", self.dir, *argv]
        try:
            rc = self.m.main()
        finally:
            self.m.analyse, self.m.subprocess, sys.argv = real
        return rc, seen

    def _count_cmd(self, seen):
        """The `count_tokens.py` command line, which is what sends prompts."""
        for cmd in seen["cmds"]:
            if any(str(c).endswith("count_tokens.py") for c in cmd):
                return cmd
        self.fail(f"counting never shelled out; commands were {seen['cmds']}")

    def _analyse_values(self, seen):
        _path, args, kwargs = seen["analyse"]
        return list(args) + list(kwargs.values())

    def test_the_surface_reaches_the_counting_subprocess_and_analysis(self):
        rc, seen = self._run_main("--target-id", "amazon-bedrock/converse")
        self.assertEqual(rc, 0)
        cmd = self._count_cmd(seen)
        self.assertIn("--target-id", cmd,
                      "--target-id never reached the counting subprocess, so "
                      "the surface prefix is not stripped before the tokenizer "
                      "is asked")
        self.assertEqual(cmd[cmd.index("--target-id") + 1],
                         "amazon-bedrock/converse")
        self.assertIn("amazon-bedrock/converse", self._analyse_values(seen),
                      "--target-id never reached analysis")

    def test_the_endpoint_reaches_the_counting_subprocess(self):
        rc, seen = self._run_main("--endpoint", "https://gateway.internal/count")
        self.assertEqual(rc, 0)
        cmd = self._count_cmd(seen)
        self.assertIn("--endpoint", cmd,
                      "--endpoint never reached the counting subprocess, so the "
                      "calls go to the default host whatever was asked for")
        self.assertEqual(cmd[cmd.index("--endpoint") + 1],
                         "https://gateway.internal/count")

    def test_the_tokenizer_id_reaches_the_counting_subprocess(self):
        rc, seen = self._run_main("--tokenizer-id", "gateway-build-7")
        self.assertEqual(rc, 0)
        cmd = self._count_cmd(seen)
        self.assertIn("--tokenizer-id", cmd)
        self.assertEqual(cmd[cmd.index("--tokenizer-id") + 1], "gateway-build-7")

    def test_all_three_arrive_together(self):
        """Each flag has its own append, and one missing is the whole defect."""
        rc, seen = self._run_main("--target-id", "amazon-bedrock/converse",
                                  "--endpoint", "https://gateway.internal/count",
                                  "--tokenizer-id", "gateway-build-7")
        self.assertEqual(rc, 0)
        cmd = self._count_cmd(seen)
        for flag, value in (("--target-id", "amazon-bedrock/converse"),
                            ("--endpoint", "https://gateway.internal/count"),
                            ("--tokenizer-id", "gateway-build-7")):
            with self.subTest(flag=flag):
                self.assertIn(flag, cmd)
                self.assertEqual(cmd[cmd.index(flag) + 1], value)

    def test_counting_and_analysis_are_given_the_same_surface(self):
        """The failure mode is not only that a value is missing, but that the
        two halves disagree: a file counted under one surface and analysed under
        another resolves two different models from one row."""
        _rc, seen = self._run_main("--target-id", "amazon-bedrock/converse")
        cmd = self._count_cmd(seen)
        self.assertEqual(cmd[cmd.index("--target-id") + 1],
                         "amazon-bedrock/converse")
        self.assertIn("amazon-bedrock/converse", self._analyse_values(seen),
                      "counting and analysis were not given the same surface")

    def test_the_artifact_records_the_settings_it_was_run_with(self):
        """They change what the points mean — the surface decides which rate
        table applies, the endpoint decides which tokenizer sized the segments —
        so an artifact recording neither cannot be told from one run against
        different settings."""
        self._run_main("--target-id", "amazon-bedrock/converse",
                       "--endpoint", "https://gateway.internal/count")
        art = json.load(open(os.path.join(self.dir, "sweep-report.json")))
        self.assertEqual(art.get("target_id"), "amazon-bedrock/converse")
        self.assertEqual(art.get("count_endpoint"),
                         "https://gateway.internal/count")


class TestOneCountedPathHelperForTheWholeToolchain(unittest.TestCase):
    """Two copies of the derivation is what produced the bug.

    `run_diagnostic.py` had `_counted_path` and `sweep_report.py` had its own
    `counted`, and only the first was fixed. Months later the second still read
    `path.replace(".jsonl", "-counted.jsonl")`, so a sweep directory named
    `a.jsonl` turned `sweep/a.jsonl/interval-10m.jsonl` into
    `sweep/a-counted.jsonl/interval-10m-counted.jsonl` — a directory that does
    not exist. Counting failed, the sweep fell back to the uncounted capture,
    and that point on the curve was estimated while its neighbours were counted.

    Fixing the second copy would have left the third to be written. So this
    asserts there is one implementation and that nothing carries a private
    version of it, by reading the sources rather than by trusting the imports.
    """

    def _sources(self):
        import glob
        for path in sorted(glob.glob(os.path.join(TIER_B, "*.py"))):
            yield os.path.basename(path), open(path).read()

    def test_no_script_derives_a_path_by_string_replacement(self):
        """The exact idiom that survived in the copy, in any file at all.

        `path.replace(".jsonl", ...)` is wrong for a directory component, for a
        name containing the extension twice, and for a name without it. There is
        no correct use of it on a path here, so its presence anywhere in tier-b
        is the finding.

        Read as code rather than as text: both files now quote the bad line in a
        comment explaining why it is gone, and a substring search calls that a
        violation.
        """
        import ast
        offenders = []
        for name, src in self._sources():
            for node in ast.walk(ast.parse(src)):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "replace"
                        and node.args
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)
                        and node.args[0].value.startswith(".")):
                    offenders.append(f"{name}:{node.lineno}")
        self.assertEqual(offenders, [],
                         "a script derives a path by string replacement instead "
                         "of using count_tokens.counted_path")

    def test_exactly_one_module_defines_the_derivation(self):
        offenders = [name for name, src in self._sources()
                     if "def counted_path(" in src]
        self.assertEqual(offenders, ["count_tokens.py"],
                         "the derivation is defined in more than one place, "
                         "which is what let the two copies drift apart")

    def test_every_script_that_needs_it_imports_the_one_helper(self):
        """Read as an import rather than as a substring: the import is a
        parenthesised multi-name one in `sweep_report.py`, and a literal search
        for a single-name form calls that a violation."""
        import ast
        for name, src in self._sources():
            if "-counted" not in src or name == "count_tokens.py":
                continue
            imported = {
                alias.name
                for node in ast.walk(ast.parse(src))
                if isinstance(node, ast.ImportFrom) and node.module == "count_tokens"
                for alias in node.names}
            with self.subTest(script=name):
                self.assertIn("counted_path", imported,
                              f"{name} names a counted export but does not use "
                              f"the shared derivation")

    def test_they_all_agree_on_every_shape(self):
        """Behaviour, not imports. If a script ever shadows the helper, the
        answers diverge here."""
        mods = {n: load(n) for n in
                ("count_tokens", "run_diagnostic", "sweep_report")}
        for p in ("run.jsonl", "capture", "run.jsonl.gz", "a.jsonl/b.jsonl",
                  "my.jsonl.backup.jsonl", ".jsonl", "no-ext-at-all",
                  "sweep/a.jsonl/interval-10m.jsonl"):
            answers = {n: m.counted_path(p) for n, m in mods.items()}
            with self.subTest(path=p):
                self.assertEqual(len(set(answers.values())), 1,
                                 f"scripts disagree on {p}: {answers}")

    def _sweep_dir(self):
        """A sweep whose directory is named `a.jsonl`, with one capture in it
        and its counted sibling already present."""
        d = tempfile.mkdtemp()
        sub = os.path.join(d, "a.jsonl")
        os.makedirs(sub)
        src = os.path.join(sub, "interval-10m.jsonl")
        row = json.dumps({"request": {
            "model": "claude-opus-5",
            "messages": [{"role": "user", "content": "hi"}]}})
        for p in (src, os.path.join(sub, "interval-10m-counted.jsonl")):
            with open(p, "w") as f:
                f.write(row + "\n")
        return sub, src

    def test_the_sweep_caller_reuses_the_counted_capture_beside_it(self):
        """Through `sweep_report.counted`, which is the function that had the
        bug — asserting on the helper alone tests the fix and not the caller.

        The counted sibling already exists, so a correct derivation finds it and
        returns it without counting anything. The broken one aimed at
        `a-counted.jsonl/interval-10m-counted.jsonl`, a directory that does not
        exist: it missed the existing file, tried to count, failed, and returned
        the *uncounted* capture, so that point on the curve was drawn from byte
        estimates while its neighbours were counted.
        """
        m = load("sweep_report")
        sub, src = self._sweep_dir()

        # Nothing may shell out here: in the broken state the miss falls through
        # to `count_tokens.py` with no `--endpoint`, whose default host is the
        # provider. Recorded rather than allowed, and the recording is asserted.
        attempts = []
        real_run = m.subprocess.run

        class _NoRun:
            @staticmethod
            def run(cmd, *a, **k):
                attempts.append(cmd)
                raise AssertionError("counted() shelled out")

        m.subprocess = _NoRun
        self.addCleanup(setattr, m.subprocess, "run", real_run)

        got = m.counted(src)
        self.assertEqual(attempts, [],
                         "counted() did not find the counted capture sitting "
                         "beside the input and tried to recount it")
        self.assertEqual(got, os.path.join(sub, "interval-10m-counted.jsonl"))

    def test_the_sweep_caller_never_aims_outside_the_capture_directory(self):
        """The same defect stated as the property, for every shape a sweep
        directory can have."""
        m = load("sweep_report")
        for d in ("sweep", "a.jsonl", "x.jsonl.bak", "2026.jsonl"):
            for base in ("interval-10m.jsonl", "interval-1h.jsonl", "capture"):
                p = os.path.join("/tmp", d, base)
                with self.subTest(path=p):
                    self.assertEqual(os.path.dirname(m.counted_path(p)),
                                     os.path.dirname(p))


class TestTheDiagnosticCannotOverwriteItsOwnInput(unittest.TestCase):
    """The toolchain derived the counted export with
    `path.replace(".jsonl", "-counted.jsonl")`, which is wrong in three ways an
    operator can hit:

      capture               -> capture            output *is* the input
      a.jsonl/b.jsonl       -> a-counted.jsonl/…  a directory rewritten
      my.jsonl.backup.jsonl -> both occurrences replaced

    The first destroys the capture: `count_tokens.py` opens the derived path
    for writing while reading the source. This is the one command that produces
    client evidence, and losing the evidence it was handed is the worst thing
    in its reach.

    It was fixed in `run_diagnostic.py` and not in `sweep_report.py`, which had
    its own copy of the same line — so every one of these cases was still live
    one file over. The derivation is now `count_tokens.counted_path` and there
    is one of it; `TestOneCountedPathHelperForTheWholeToolchain` holds that
    line, and these cases are asserted against every script that derives a
    counted path rather than against the one that was fixed first.
    """

    SCRIPTS = ("count_tokens", "run_diagnostic", "sweep_report")

    def _derivers(self):
        return [(n, load(n).counted_path) for n in self.SCRIPTS]

    def test_an_extensionless_input_gets_a_distinct_output(self):
        for name, d in self._derivers():
            with self.subTest(script=name):
                self.assertNotEqual(d("capture"), "capture")
                self.assertTrue(d("capture").endswith(".jsonl"))

    def test_a_directory_named_like_the_file_is_left_alone(self):
        for name, d in self._derivers():
            with self.subTest(script=name):
                self.assertEqual(d("a.jsonl/b.jsonl"),
                                 os.path.join("a.jsonl", "b-counted.jsonl"))

    def test_only_the_extension_is_replaced_not_every_occurrence(self):
        for name, d in self._derivers():
            with self.subTest(script=name):
                self.assertEqual(d("my.jsonl.backup.jsonl"),
                                 "my.jsonl.backup-counted.jsonl")

    def test_the_ordinary_case_is_unchanged(self):
        for name, d in self._derivers():
            with self.subTest(script=name):
                self.assertEqual(d("run.jsonl"), "run-counted.jsonl")
                self.assertEqual(d("/tmp/x/run.jsonl"), "/tmp/x/run-counted.jsonl")

    def test_no_input_shape_produces_a_colliding_output(self):
        """The property, not the four cases: whatever the shape, the counted
        path is never the path being read."""
        for name, d in self._derivers():
            for p in ("run.jsonl", "capture", "run.jsonl.gz", "a.jsonl/b.jsonl",
                      "my.jsonl.backup.jsonl", "/tmp/x/run.jsonl", ".jsonl",
                      "x/y.jsonl", "no-ext-at-all"):
                with self.subTest(script=name, path=p):
                    self.assertNotEqual(os.path.abspath(d(p)),
                                        os.path.abspath(p))

    def test_the_output_never_leaves_the_inputs_directory(self):
        """The sweep case. `a.jsonl/b.jsonl` -> `a-counted.jsonl/b-counted.jsonl`
        does not merely rename the file, it names a directory that does not
        exist — so counting fails, `sweep_report` falls back to the uncounted
        capture, and that point on the curve is drawn from byte estimates while
        its neighbours are counted."""
        for name, d in self._derivers():
            for p in ("a.jsonl/b.jsonl", "sweep/a.jsonl/interval-10m.jsonl",
                      "/tmp/x.jsonl/y.jsonl", "d.jsonl/capture"):
                with self.subTest(script=name, path=p):
                    self.assertEqual(os.path.dirname(d(p)), os.path.dirname(p))

    def test_end_to_end_the_source_survives(self):
        """The consequence, driven through the real command."""
        d = tempfile.mkdtemp()
        src = os.path.join(d, "capture")            # no extension
        row = json.dumps({"request": {
            "model": "claude-opus-5",
            "system": [{"type": "text", "text": "ORIGINAL-CAPTURE"}],
            "messages": [{"role": "user", "content": "hi"}]}})
        with open(src, "w") as f:
            f.write(row + "\n")
        subprocess.run(
            [sys.executable, "-B", os.path.join(TIER_B, "run_diagnostic.py"),
             src, "--endpoint", "http://127.0.0.1:1/x"],
            capture_output=True, text=True, timeout=90,
            env=dict(os.environ, ANTHROPIC_API_KEY="test"))
        self.assertIn("ORIGINAL-CAPTURE", open(src).read(),
                      "the diagnostic overwrote the capture it was given")


class TestAppendingOntoATruncatedTailLosesNothingExtra(unittest.TestCase):
    """A capture killed mid-write leaves its last row without a newline.

    Appending straight onto that joins the new run's first row to the broken
    one, so ingest drops a single unparseable line and *both* rows are lost --
    and `capture_run` cannot help, because the row carrying it never parses.
    Measured before the fix: one parseable row out of three written.

    Terminated rather than refused. The stale half-row is already
    unrecoverable, so refusing makes an operator hand-repair a file to get back
    a row that no longer exists, while a newline costs nothing and keeps every
    row of the resumed run. Ingest already tolerates and counts one unparseable
    line.
    """

    def _truncated(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "cap.jsonl")
        with open(p, "w") as f:
            f.write(json.dumps({"request_id": "r1", "capture_run": "aaa"}) + "\n")
            f.write('{"request_id": "r2", "capt')      # killed here
        return p

    @staticmethod
    def _counts(path):
        good = bad = 0
        for line in open(path):
            try:
                json.loads(line)
                good += 1
            except ValueError:
                bad += 1
        return good, bad

    def test_the_recorder_repairs_the_tail_on_construction(self):
        from cacheeconomics.recorder import Recorder
        p = self._truncated()
        Recorder(p, key=b"k" * 32)
        with open(p, "a") as f:
            f.write(json.dumps({"request_id": "r3"}) + "\n")
        good, bad = self._counts(p)
        self.assertEqual((good, bad), (2, 1),
                         "the resumed run's row was swallowed by the broken one")

    def test_the_recorder_leaves_a_clean_file_alone(self):
        """The other direction: no spurious blank line on a healthy file."""
        from cacheeconomics.recorder import Recorder
        d = tempfile.mkdtemp()
        p = os.path.join(d, "clean.jsonl")
        with open(p, "w") as f:
            f.write(json.dumps({"request_id": "r1"}) + "\n")
        before = open(p).read()
        Recorder(p, key=b"k" * 32)
        self.assertEqual(open(p).read(), before)

    def test_a_missing_file_is_not_created_by_the_check(self):
        from cacheeconomics.recorder import Recorder
        d = tempfile.mkdtemp()
        p = os.path.join(d, "nope.jsonl")
        Recorder(p, key=b"k" * 32)
        self.assertFalse(os.path.exists(p))

    def test_the_proxy_does_the_same_before_appending(self):
        """Same class, other member. Driven through the real command."""
        p = self._truncated()
        # Popen, not run(): the proxy serves forever, so a timeout on run()
        # kills the test rather than the server.
        proc = subprocess.Popen(
            [sys.executable, "-B", os.path.join(TIER_B, "capture_proxy.py"),
             "--out", p, "--append", "--port", str(_ProxyCase._free_port()),
             "--upstream", "http://127.0.0.1:1"],
            stderr=subprocess.PIPE, text=True,
            env=dict(os.environ, ANTHROPIC_API_KEY="test"))
        self.addCleanup(proc.terminate)
        seen, deadline = "", time.time() + 15
        while time.time() < deadline and "ended mid-row" not in seen:
            line = proc.stderr.readline()
            if not line:
                break
            seen += line
        self.assertIn("ended mid-row", seen)
