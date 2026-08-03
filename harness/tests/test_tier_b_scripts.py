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

import http.client
import http.server
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

        self.out = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        self.out.close()
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
        self.out = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        self.out.close()
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
