"""The CLI is the only part of this package most people will ever touch.

Two things it is responsible for beyond running the right function: it must not
put the HMAC key anywhere it can leak, and it must not become the place where
the withheld-figures rule quietly stops applying. Both are asserted here rather
than trusted.
"""

import contextlib
import io
import json
import os
import stat
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cacheeconomics import cli                                          # noqa: E402

FIXTURE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "fixtures", "demo-traces.jsonl")


def run(*argv):
    """Return (exit_code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = cli.main(list(argv))
        except SystemExit as e:                       # argparse usage errors
            code = e.code if isinstance(e.code, int) else 2
    return code, out.getvalue(), err.getvalue()


class TestTheKeyNeverTouchesArgv(unittest.TestCase):
    """`ps` shows argv to every user on the box, and the shell writes it to
    history. A key that has been on a command line is a key that has leaked, so
    the flag does not exist -- and this asserts it stays that way."""

    def _option_strings(self, parser):
        """Every flag argparse will actually accept, subcommands included.

        Read off the parser rather than grepped out of `--help`: the help text
        explains *why* there is no --key flag, so a substring search found the
        sentence warning against it and called that a violation.
        """
        out = set()
        for action in parser._actions:
            out.update(action.option_strings)
            # `choices` is None on most actions, a plain tuple on ones like
            # --format, and a name->parser dict only on the subparser action.
            # Only the dict is worth descending into.
            choices = getattr(action, "choices", None)
            if isinstance(choices, dict):
                for sub in choices.values():
                    if hasattr(sub, "_actions"):
                        out |= self._option_strings(sub)
        return out

    def test_there_is_no_key_flag(self):
        flags = self._option_strings(cli.build_parser())
        self.assertNotIn("--key", flags,
                         "argv is visible in ps and saved to shell history")
        self.assertIn("--key-file", flags)

    def test_a_world_readable_key_file_is_refused(self):
        fd, path = tempfile.mkstemp()
        os.write(fd, b"k" * 32)
        os.close(fd)
        try:
            os.chmod(path, 0o644)
            code, _out, err = run("analyze", FIXTURE, "--key-file", path)
            self.assertEqual(code, 1)
            self.assertIn("readable by other users", err)
            self.assertIn("chmod 600", err)
        finally:
            os.unlink(path)

    def test_a_private_key_file_is_accepted(self):
        fd, path = tempfile.mkstemp()
        os.write(fd, b"k" * 32)
        os.close(fd)
        try:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
            code, _out, err = run("analyze", FIXTURE, "--key-file", path,
                                  "--allow-unreconciled")
            self.assertEqual(code, 0, err)
        finally:
            os.unlink(path)

    def test_a_short_key_is_refused_with_the_reason(self):
        fd, path = tempfile.mkstemp()
        os.write(fd, b"short")
        os.close(fd)
        try:
            os.chmod(path, 0o600)
            code, _out, err = run("analyze", FIXTURE, "--key-file", path)
            self.assertEqual(code, 1)
            self.assertIn("minimum", err)
            self.assertIn("openssl rand", err, "say how to make a good one")
        finally:
            os.unlink(path)

    def test_the_body_adapter_will_not_run_unkeyed(self):
        """It derives ids by hashing prompt content, and an unkeyed digest of a
        prompt is confirmable by anyone holding a guess."""
        env = os.environ.pop(cli.KEY_ENV, None)
        try:
            code, _out, err = run("analyze", FIXTURE, "--from", "bodies")
            self.assertEqual(code, 1)
            self.assertIn("needs an HMAC key", err)
        finally:
            if env is not None:
                os.environ[cli.KEY_ENV] = env


class TestFiguresStayWithheldByDefault(unittest.TestCase):
    """The rule the whole project is sold on. A CLI is an easy place for it to
    stop being true, because the convenient default is to print the number."""

    def test_no_invoice_means_no_dollars(self):
        code, out, _err = run("analyze", FIXTURE)
        self.assertEqual(code, 0)
        self.assertIn("FIGURES WITHHELD", out)
        # Per-finding, not just the banner. The money column reads "withheld"
        # where a figure exists and has not been released; the banner alone
        # would still be printed if a single row leaked its number.
        self.assertIn("withheld", out)
        self.assertNotRegex(out, r"~\$[\d,]+/mo")

    def test_the_draft_flag_is_what_releases_them(self):
        code, out, _err = run("analyze", FIXTURE, "--allow-unreconciled")
        self.assertEqual(code, 0)
        # `[figure withheld]` no longer appears anywhere, so asserting its
        # absence had stopped testing anything. The claim is that the released
        # run prints amounts and no row still reads as withheld.
        self.assertNotRegex(out, r"\bwithheld\b\s{2,}")
        self.assertRegex(out, r"~\$[\d,.]+/mo")

    def test_json_output_serialises_a_withheld_figure_as_text(self):
        """Not as a number. A script reading this must not be handed a float it
        would treat as spend -- `raw()` is deliberately unreachable from here."""
        code, out, _err = run("analyze", FIXTURE, "--format", "json")
        self.assertEqual(code, 0)
        payload = json.loads(out)
        for value in payload["spend"].values():
            self.assertIsInstance(value, str)
        self.assertTrue(any("withheld" in v for v in payload["spend"].values()))


class TestTheSubcommandsRun(unittest.TestCase):

    def test_analyze_text(self):
        code, out, _err = run("analyze", FIXTURE, "--allow-unreconciled")
        self.assertEqual(code, 0)
        # The sections a reader walks through, in order. `ingest tier` was the
        # old label for what section 1 now calls depth.
        for label in ("what I read", "how your caching is doing",
                      "what it is costing you", "what to do next",
                      "depth", "prefix efficiency"):
            self.assertIn(label, out)
        # The header is padded to its longest label; it used to render
        # "prefix efficiency17%" with no gap.
        self.assertNotRegex(out, r"prefix efficiency\d")

    def test_analyze_html_to_a_file(self):
        fd, path = tempfile.mkstemp(suffix=".html")
        os.close(fd)
        try:
            code, _out, err = run("analyze", FIXTURE, "--format", "html",
                                  "--out", path, "--allow-unreconciled")
            self.assertEqual(code, 0, err)
            with open(path) as f:
                self.assertIn("<html", f.read().lower())
        finally:
            os.unlink(path)

    def test_bakeoff(self):
        code, out, _err = run("bakeoff", FIXTURE)
        self.assertEqual(code, 0)
        self.assertIn("litellm-auto", out)

    def test_registry_reports_pricing_coverage(self):
        """A surface the registry describes but cannot price is the difference
        between "we support that" and "we can answer a question about it"."""
        code, out, _err = run("registry")
        self.assertEqual(code, 0)
        self.assertIn("anthropic/direct", out)
        self.assertIn("priced_models=", out)

    def test_checks_exit_code_carries_the_verdict(self):
        # The surface is named, because every threshold these checks read is a
        # per-surface fact. `--target-id` used to default to anthropic/direct,
        # so this test passed while asking about a provider nobody had chosen --
        # see TestTheChecksCommandWillNotAnswerForAnUnnamedSurface below.
        bad, _o, _e = run("checks", "--prefix-tokens", "300",
                          "--model", "claude-opus-5", "--breakpoints", "5",
                          "--target-id", "anthropic/direct")
        good, _o, _e = run("checks", "--prefix-tokens", "9000",
                           "--model", "claude-opus-5", "--breakpoints", "2",
                           "--target-id", "anthropic/direct")
        self.assertEqual(bad, 2, "a failing configuration must be detectable")
        self.assertEqual(good, 0)

    def test_a_missing_file_is_a_message_not_a_traceback(self):
        code, _out, err = run("analyze", "/definitely/not/here.jsonl")
        self.assertEqual(code, 1)
        self.assertIn("no such file", err)
        self.assertNotIn("Traceback", err)

    def test_an_unknown_model_abstains_rather_than_passing(self):
        """The registry refuses to guess a minimum, because they are
        non-monotonic across generations. That has to reach the exit code: a
        first mapping sent ABSTAIN to 0, so a pipeline got a green build in which
        the check that catches silently-ignored markers never ran."""
        code, out, err = run("checks", "--prefix-tokens", "9000",
                             "--model", "not-a-real-model", "--breakpoints", "1")
        self.assertEqual(code, 3, "abstain must not read as success")
        self.assertIn("ABSTAIN", out)
        self.assertNotIn("Traceback", err)


class TestNothingHereReachesTheNetwork(unittest.TestCase):
    """Local-first with zero egress is the claim the analysis is sold on, and a
    CLI is where it would quietly stop being true.

    Asserted structurally rather than by example: an import is the only way a
    stdlib-only package gets a socket, and the defect appears with the call that
    has not been written yet.
    """

    NETWORK = {"socket", "http", "urllib", "requests", "httpx", "ftplib",
               "smtplib", "telnetlib", "asyncio.streams", "ssl", "xmlrpc"}

    def test_no_module_in_the_analysis_path_imports_a_network_library(self):
        import ast
        import pathlib

        import cacheeconomics
        pkg = pathlib.Path(cacheeconomics.__file__).parent
        offenders = {}
        for path in sorted(pkg.rglob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for n in names:
                    if n.split(".")[0] in self.NETWORK:
                        offenders.setdefault(path.name, []).append(n)
        self.assertFalse(
            offenders,
            f"network imports in the package: {offenders}. Analysis is "
            f"local-first with zero egress; that is the claim clients are asked "
            f"to trust when they hand over a trace.")


if __name__ == "__main__":
    unittest.main()


class TestCiAssertsThingsTheReportStillSays(unittest.TestCase):
    """CI greps the shipped wheel's output for literal strings.

    That step is the only place the *installed artifact* is checked, and it
    lives outside pytest, so a rename in the report passes the whole suite and
    fails on push. It has now done exactly that: the workflow grepped for
    `[figure withheld]` for one commit after the report stopped emitting it.

    The same shape as the twin-path bugs this codebase keeps finding -- two
    copies of one claim, one of them unreachable from the tests. This makes the
    workflow's literals reachable.
    """

    WORKFLOW = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", ".github", "workflows", "ci.yml"))

    def _report(self):
        from cacheeconomics.analyzer import analyze
        from cacheeconomics.report import render_text
        from cacheeconomics.trace import load_jsonl
        return render_text(analyze(load_jsonl(FIXTURE)))

    @unittest.skipUnless(os.path.exists(WORKFLOW), "workflow not in this checkout")
    def test_every_positive_grep_matches_the_real_report(self):
        import re
        with open(self.WORKFLOW) as f:
            body = f.read()
        # `grep -q "X" out.txt` only -- the negative form is checked below, and
        # a grep against any other file is not making a claim about the report.
        wanted = re.findall(r'^\s*grep -q "([^"]+)" out\.txt\s*$',
                            body, re.MULTILINE)
        self.assertTrue(wanted, "no report greps found; has the step moved?")
        report = self._report()
        for pattern in wanted:
            with self.subTest(pattern=pattern):
                self.assertRegex(report, pattern.replace("\\[", "["),
                                 f"CI greps for {pattern!r}, which this report "
                                 f"no longer says")

    @unittest.skipUnless(os.path.exists(WORKFLOW), "workflow not in this checkout")
    def test_the_negative_grep_is_not_vacuous(self):
        """A pattern that can never match anything passes forever and guards
        nothing. It has to match a *released* report to be worth having."""
        import re

        from cacheeconomics.analyzer import analyze
        from cacheeconomics.report import render_text
        from cacheeconomics.trace import load_jsonl
        with open(self.WORKFLOW) as f:
            body = f.read()
        forbidden = re.findall(r"if grep -qE '([^']+)' out\.txt", body)
        self.assertTrue(forbidden, "no negative grep found; has the step moved?")
        released = render_text(analyze(load_jsonl(FIXTURE), invoice_usd=17.45))
        for pattern in forbidden:
            with self.subTest(pattern=pattern):
                self.assertRegex(released, pattern,
                                 "the forbidden pattern never appears even when "
                                 "figures ARE released, so CI is asserting the "
                                 "absence of something the tool cannot emit")
                self.assertNotRegex(self._report(), pattern)


class TestTheChecksCommandWillNotAnswerForAnUnnamedSurface(unittest.TestCase):
    """The function-level default was closed and the CLI walked around it.

    `checks.check_minimum` and friends stopped defaulting `target_id`, but
    `--target-id` still carried `default="anthropic/direct"` in the argparse
    setup, so the public command handed the fabricated surface straight back in.
    A parameter default is only one of the routes a surface gets named without
    anyone choosing it; an argparse default is another, and `inspect.signature`
    cannot see it.

    Reproduced through the shipped CLI:

        checks --prefix-tokens 768 --model claude-opus-5 --tokens-are-exact

    exited 0 with three PASSes against Anthropic's 512-token minimum. The same
    prefix on openai/direct FAILs against its 1,024.
    """

    ARGS = ("checks", "--prefix-tokens", "768", "--model", "claude-opus-5",
            "--tokens-are-exact")

    def test_it_abstains_and_exits_three_rather_than_passing(self):
        code, out, _err = run(*self.ARGS)
        self.assertEqual(code, 3, "abstain must not read as success")
        self.assertNotIn("PASS", out)
        self.assertEqual(3, out.count("ABSTAIN"),
                         "every check answers per surface, so every one abstains")

    def test_it_says_what_would_settle_it(self):
        _code, out, _err = run(*self.ARGS)
        self.assertIn("no provider surface named", out)
        self.assertIn("target_id", out)

    def test_the_same_prefix_still_gets_opposite_verdicts_once_named(self):
        """Why the abstention is the honest answer rather than a nuisance."""
        ok, _o, _e = run(*self.ARGS, "--target-id", "anthropic/direct")
        bad, _o, _e = run(*self.ARGS, "--target-id", "openai/direct")
        self.assertEqual(ok, 0)
        self.assertEqual(bad, 2)

    def test_naming_the_surface_still_works(self):
        code, out, _err = run(*self.ARGS, "--target-id", "anthropic/direct")
        self.assertEqual(code, 0)
        self.assertIn("PASS", out)


class TestClaudeCodeMakesTheSurfaceAChoice(unittest.TestCase):
    """`cmd_claude_code` did `args.target_id or "anthropic/direct"`.

    That is the *default* path of the command, and a transcript carries no
    provider field anywhere, so the ordinary invocation priced Claude Code
    sessions at Anthropic first-party rates whatever they actually ran against.
    Measured on a 40-request fixture with `--allow-unreconciled`: $6.66 input
    spend, $38.46 uncached, $31.80 saved and $123/mo, all released as DRAFT
    under a rate table nobody chose.

    Silently switching the default to UNATTRIBUTED was measured before it was
    rejected: it empties the report rather than merely withholding dollars. The
    same fixture then reports 0 requests, `input_from_cache` None,
    `prefix_efficiency` None and no findings, in place of 40 requests at 94% and
    94% -- and those two ratios come from the provider's own usage counters and
    need no surface at all. So the choice is made explicit instead.
    """

    def _fixture(self, tmp):
        proj = os.path.join(tmp, "proj")
        os.makedirs(proj)
        with open(os.path.join(proj, "s.jsonl"), "w") as f:
            for i in range(12):
                f.write(json.dumps({
                    "type": "assistant", "sessionId": "s1", "uuid": f"u{i}",
                    "requestId": f"r{i}",
                    "timestamp": f"2026-07-29T09:{i:02d}:00.000Z",
                    "message": {"model": "claude-opus-5", "usage": {
                        "input_tokens": 100, "output_tokens": 10,
                        "cache_read_input_tokens": 20_000,
                        "cache_creation_input_tokens": 1_000,
                        # The per-lifetime split, without which every row is
                        # excluded as an unprovable write and no dollar figure
                        # is printed for a reason that has nothing to do with
                        # the surface -- which would make the assertions below
                        # pass while testing the wrong refusal.
                        "cache_creation": {
                            "ephemeral_5m_input_tokens": 1_000,
                            "ephemeral_1h_input_tokens": 0}}}}) + "\n")
        return tmp

    def test_it_refuses_rather_than_assuming_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out, err = run("claude-code", "--root", self._fixture(tmp),
                                 "--allow-unreconciled")
        self.assertEqual(code, 1)
        self.assertNotIn("$", out, "no dollar figure may be printed at all")
        self.assertNotIn("Traceback", err)

    def test_the_refusal_names_both_ways_forward(self):
        with tempfile.TemporaryDirectory() as tmp:
            _code, _out, err = run("claude-code", "--root", self._fixture(tmp))
        self.assertIn("--assume-anthropic-direct", err)
        self.assertIn("--target-id", err)
        self.assertIn("carries no provider field", err)

    def test_the_assumption_can_be_made_explicitly(self):
        """And produces the report, rather than an empty shell.

        Asserted on the two things that go blank when no surface is priceable,
        because those are what made the silent-UNATTRIBUTED option an
        over-block: `analyze` recomputes its ratios over the priced requests, so
        with none priceable `input_from_cache` and `prefix_efficiency` render as
        "—" and the findings list empties -- for two measurements taken straight
        from the provider's usage counters, which no rate table affects.
        """
        with tempfile.TemporaryDirectory() as tmp:
            code, out, _err = run("claude-code", "--root", self._fixture(tmp),
                                  "--assume-anthropic-direct",
                                  "--allow-unreconciled")
        self.assertEqual(code, 0)
        self.assertIn("DRAFT", out, "the figures were not released at all")
        self.assertRegex(out, r"input from cache\s+\d+%",
                         "the measured ratio came back blank")
        self.assertRegex(out, r"prefix efficiency\s+\d+%")
        self.assertIn("1 finding", out, "the findings list came back empty")

    def test_and_is_still_disclosed_when_it_is_made(self):
        """Opting in states the assumption; it does not retire it."""
        with tempfile.TemporaryDirectory() as tmp:
            _code, out, _err = run("claude-code", "--root", self._fixture(tmp),
                                   "--assume-anthropic-direct",
                                   "--allow-unreconciled")
        flat = " ".join(out.split())
        self.assertIn("surface assumed", flat)
        self.assertIn("--target-id", flat)

    def test_naming_a_different_surface_still_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _out, _err = run("claude-code", "--root", self._fixture(tmp),
                                   "--target-id", "amazon-bedrock/converse",
                                   "--allow-unreconciled")
        self.assertEqual(code, 0)

    def test_the_two_flags_cannot_contradict_each_other(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _out, err = run("claude-code", "--root", self._fixture(tmp),
                                  "--target-id", "amazon-bedrock/converse",
                                  "--assume-anthropic-direct")
        self.assertEqual(code, 1)
        self.assertIn("not both", err)
