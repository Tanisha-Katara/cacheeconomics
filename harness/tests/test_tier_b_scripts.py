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


class TestEverySuffixWeDeriveFromOutIsIgnored(unittest.TestCase):
    """The operator chooses `--out`. They do not choose the suffixes this
    toolchain appends to it, so keeping those out of a commit is our job.

    `count_tokens.py` derives three: `.cache.json`, `.partial-write` and
    `.partial`. The first two were added to `.gitignore` and the third — the
    one written when some rows could not be counted, holding the enriched
    export with request bodies intact — was not. Same defect class, one suffix
    over, which is how the first two came to be added in the first place.

    This asserts the *class* rather than the three names: it reads the suffixes
    out of the source, so a fourth one added later fails here instead of in
    somebody's repository.
    """

    def _derived_suffixes(self):
        import re
        src = open(os.path.join(TIER_B, "count_tokens.py")).read()
        return sorted(set(re.findall(r'args\.out \+ "([^"]+)"', src)))

    def test_the_source_still_derives_the_suffixes_we_think_it_does(self):
        """Guards the guard: if the derivation stops looking like this, the
        test below silently checks nothing."""
        found = self._derived_suffixes()
        self.assertTrue(found, "no derived suffixes found; the pattern moved")
        self.assertIn(".partial", found)
        self.assertIn(".cache.json", found)

    def test_each_one_is_gitignored(self):
        root = os.path.dirname(TIER_B)
        for suffix in self._derived_suffixes():
            with self.subTest(suffix=suffix):
                probe = os.path.join(root, "some-run.jsonl" + suffix)
                r = subprocess.run(["git", "check-ignore", "-q", probe],
                                   cwd=root, capture_output=True)
                self.assertEqual(r.returncode, 0,
                                 f"'{suffix}' is written beside the operator's "
                                 f"--out and git would stage it")

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

    def setUp(self):
        self.dir = tempfile.mkdtemp()


class TestTheDiagnosticCannotOverwriteItsOwnInput(unittest.TestCase):
    """`run_diagnostic.py` derived the counted export with
    `args.path.replace(".jsonl", "-counted.jsonl")`, which is wrong in three
    ways an operator can hit:

      capture               -> capture            output *is* the input
      a.jsonl/b.jsonl       -> a-counted.jsonl/…  a directory rewritten
      my.jsonl.backup.jsonl -> both occurrences replaced

    The first destroys the capture: `count_tokens.py` opens the derived path
    for writing while reading the source. This is the one command that produces
    client evidence, and losing the evidence it was handed is the worst thing
    in its reach.
    """

    def _mod(self):
        return load("run_diagnostic")

    def test_an_extensionless_input_gets_a_distinct_output(self):
        m = self._mod()
        self.assertNotEqual(m._counted_path("capture"), "capture")
        self.assertTrue(m._counted_path("capture").endswith(".jsonl"))

    def test_a_directory_named_like_the_file_is_left_alone(self):
        m = self._mod()
        self.assertEqual(m._counted_path("a.jsonl/b.jsonl"),
                         os.path.join("a.jsonl", "b-counted.jsonl"))

    def test_only_the_extension_is_replaced_not_every_occurrence(self):
        m = self._mod()
        self.assertEqual(m._counted_path("my.jsonl.backup.jsonl"),
                         "my.jsonl.backup-counted.jsonl")

    def test_the_ordinary_case_is_unchanged(self):
        m = self._mod()
        self.assertEqual(m._counted_path("run.jsonl"), "run-counted.jsonl")
        self.assertEqual(m._counted_path("/tmp/x/run.jsonl"),
                         "/tmp/x/run-counted.jsonl")

    def test_no_input_shape_produces_a_colliding_output(self):
        """The property, not the four cases: whatever the shape, the counted
        path is never the path being read."""
        m = self._mod()
        for p in ("run.jsonl", "capture", "run.jsonl.gz", "a.jsonl/b.jsonl",
                  "my.jsonl.backup.jsonl", "/tmp/x/run.jsonl", ".jsonl",
                  "x/y.jsonl", "no-ext-at-all"):
            with self.subTest(path=p):
                self.assertNotEqual(os.path.abspath(m._counted_path(p)),
                                    os.path.abspath(p))

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
