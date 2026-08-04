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
import re
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


class TestAnAssumedSurfaceCannotBePublishedAsReconciled(unittest.TestCase):
    """`--assume-anthropic-direct` releases DRAFT figures at best.

    Reconciliation checks a *total* against an invoice. It cannot check the
    *rate table* that total was computed from, and with this flag that table
    came from an assumption -- a Claude Code transcript carries no provider
    field to compare against. So the report could carry
    `released_as='reconciled'`, the label meaning an invoice verified this, over
    dollars whose provenance was a guess.

    Reproduced before the fix, on a 12-request fixture with an invoice equal to
    computed spend: reconciled at 0.0%, and input_usd, if_uncached_usd,
    caching_saved_usd and monthly_input_usd all released as 'reconciled'.

    The assumption was disclosed, but only as prose a renderer adds later: in
    the text report a costed finding and the total appear above the caveat, and
    in HTML the Input spend KPI appears above the standing notes. Neither a
    reader skimming nor a script reading the JSON `release_state` is reached by
    a sentence further down the page.
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
                        "cache_creation": {
                            "ephemeral_5m_input_tokens": 1_000,
                            "ephemeral_1h_input_tokens": 0}}}}) + "\n")
        return tmp

    def _analysis(self, tmp, target_id):
        """An analysis with an invoice that reconciles exactly, so the release
        gate genuinely passes and RECONCILED is what it would otherwise emit."""
        from cacheeconomics.adapters.claude_code import load_sessions
        from cacheeconomics.analyzer import analyze
        ts = load_sessions(root=self._fixture(tmp), target_id=target_id)
        spend = analyze(ts, allow_unreconciled=True).spend["input_usd"].raw()
        return analyze(ts, invoice_usd=spend)

    def _states(self, a):
        """Release provenance of the figures that were actually published.

        Withheld figures are excluded deliberately: this fixture's window is
        minutes long, so `monthly_input_usd` is held back by the projection
        floor and carries no release state at all. The question here is what
        label the *published* figures wear.
        """
        from cacheeconomics.money import Figure
        return {k: v.released_as for k, v in a.spend.items()
                if isinstance(v, Figure) and v.released}

    def test_the_gate_really_does_pass_so_reconciled_is_the_alternative(self):
        """Guard the guard. If the fixture did not reconcile, every figure would
        be withheld for an unrelated reason and the assertions below would hold
        without testing anything."""
        with tempfile.TemporaryDirectory() as tmp:
            a = self._analysis(tmp, "anthropic/direct")
        self.assertEqual(0.0, a.reconciliation["delta_pct"])
        self.assertEqual({"reconciled"}, set(self._states(a).values()))

    def test_the_flag_downgrades_every_spend_figure_to_draft(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = cli._draft_because_the_surface_was_assumed(
                self._analysis(tmp, "anthropic/direct"))
        states = self._states(a)
        self.assertTrue(states, "no figures found; the check would be vacuous")
        self.assertEqual({"draft"}, set(states.values()))

    def test_and_the_reconciliation_figures_too(self):
        """`reconciliation.computed_usd` and `delta_usd` are money a script
        reads, and they shipped as bare reconciled dollars once before."""
        from cacheeconomics.money import Figure
        with tempfile.TemporaryDirectory() as tmp:
            a = cli._draft_because_the_surface_was_assumed(
                self._analysis(tmp, "anthropic/direct"))
        recon = [v.released_as for v in a.reconciliation.values()
                 if isinstance(v, Figure)]
        self.assertTrue(recon)
        self.assertEqual({"draft"}, set(recon))

    def test_both_renderers_stamp_it_without_being_told_separately(self):
        from cacheeconomics import report
        with tempfile.TemporaryDirectory() as tmp:
            a = cli._draft_because_the_surface_was_assumed(
                self._analysis(tmp, "anthropic/direct"))
        self.assertTrue(report._is_draft(a),
                        "release state is read off the figures; it says reconciled")
        for name, fn in (("text", report.render_text),
                         ("html", report.render_html)):
            with self.subTest(renderer=name):
                out = fn(a)
                self.assertIn("DRAFT", out)
                first_draft = out.find("DRAFT")
                first_money = out.find("$")
                self.assertTrue(
                    first_money == -1 or first_draft < first_money,
                    f"{name} printed a dollar figure before the DRAFT stamp")

    def test_the_stamp_says_it_was_the_surface_that_was_assumed(self):
        """A DRAFT banner that says "no invoice supplied" would be false here --
        an invoice *was* supplied and it reconciled."""
        from cacheeconomics import report
        with tempfile.TemporaryDirectory() as tmp:
            a = cli._draft_because_the_surface_was_assumed(
                self._analysis(tmp, "anthropic/direct"))
        reason = report._draft_reason(a)
        self.assertIn("surface was assumed", reason)
        self.assertIn("--target-id", reason)

    def _payload(self, tmp):
        a = cli._draft_because_the_surface_was_assumed(
            self._analysis(tmp, "anthropic/direct"))
        return json.loads(cli.analysis_json(a, tier_name="USAGE_ONLY",
                                            coverage=1.0))

    # Money-shaped fields that are *inputs* rather than computed figures. The
    # client's own invoice is a number they supplied; demanding release
    # provenance for it would be an over-block. Full paths, not leaf names, for
    # the same reason `_state_for` resolves by path.
    _INPUT_ONLY_MONEY = frozenset({"reconciliation.invoice_usd"})

    @classmethod
    def _money_paths(cls, node, path="", key=""):
        """Every path in the decoded JSON whose value is money.

        Detected two ways, because either alone has a hole:

        By *rendering* -- a Figure serialises through `str`, so it is "$..." or
        "[withheld: ...]", and both count, since a withheld figure still has to
        be identifiable as withheld rather than absent.

        And by *name*. Rendering alone missed any money serialised as a raw
        number: `{"projected_usd": 12.5}` in a new or nested section was
        invisible, so a section could publish unprovenanced dollars and this
        walk would report the payload clean. Measured on a synthetic payload
        before this was added.

        Plain floats are still not money on their own -- `window_days` and
        `delta_pct` sit in the same dicts -- which is why the name test is what
        promotes a number, not the type.

        Written out rather than imported from `test_invariants`, which owns the
        general form. Both holes closed here were holes that file had already
        found and fixed; reproducing them independently is the argument for
        saying so in the comments rather than trusting the shape to be obvious.
        """
        out = []
        if isinstance(node, dict):
            for k, v in node.items():
                # Both encodings of provenance are skipped, not just the map.
                # `_state_for` explicitly supports an inline `<leaf>_state`
                # sibling -- so `computed_usd_state` contains "_usd", was walked
                # as a money field in its own right, and was then reported as
                # money with no provenance. One of this walk's own supported
                # shapes could not pass it.
                if k == "release_state" or k.endswith("_state"):
                    continue
                out.extend(cls._money_paths(v, f"{path}.{k}" if path else k, k))
            return out
        if isinstance(node, list):
            for i, v in enumerate(node):
                out.extend(cls._money_paths(v, f"{path}[{i}]", key))
            return out
        if path in cls._INPUT_ONLY_MONEY:
            return out
        by_name = ("_usd" in key
                   and isinstance(node, (int, float, str))
                   and not isinstance(node, bool))
        by_render = isinstance(node, str) and (
            node.startswith("[withheld")
            or bool(re.fullmatch(r"\$-?[\d,]+(?:\.\d+)?", node.strip())))
        if by_name or by_render:
            out.append(path)
        return out

    @classmethod
    def _state_for(cls, payload, path):
        """The release state recorded for the money at `path`, or None.

        Resolved by *full path*. Keying on the leaf name let any field called
        `input_usd`, at any depth, borrow `spend.input_usd`'s state -- so a new
        section carrying reconciled dollars and no provenance of its own read as
        covered. Measured on a synthetic payload: `new_section.input_usd`
        returned 'draft' with nothing local vouching for it.

        The top-level `release_state` map vouches for `spend.*` and nothing
        else. Anything deeper carries its own state, in a sibling
        `release_state` map or inline beside the value.

        That is not an assumption about how the serialiser *might* be written;
        it is what `analysis_json` emits, checked against the real thing rather
        than reasoned about. `_release_state(mapping)` is keyed by field name
        within a section, and it is called three times: once at top level for
        `a.spend`, once inside `_reconciliation_json` for the reconciliation
        block, and once inside each finding. Run over that payload, this
        resolver leaves nothing unresolved.

        A centralised full-path map -- `release_state["reconciliation.
        computed_usd"]` -- would resolve to None here and the marker would stay
        red through a legitimate fix, which is why this is pinned by
        `test_only_spend_is_vouched_for_by_the_top_level_map` rather than left
        as a comment.
        """
        parts = path.split(".")
        leaf = parts[-1]
        if len(parts) == 2 and parts[0] == "spend":
            top = payload.get("release_state") or {}
            if leaf in top:
                return top[leaf]
        node = payload
        for p in parts[:-1]:
            if "[" in p:
                name, idx = p.split("[")
                seq = node.get(name) or []
                i = int(idx.rstrip("]"))
                if i >= len(seq):
                    return None
                node = seq[i]
            else:
                node = node.get(p, {})
            if not isinstance(node, (dict, list)):
                return None
        if isinstance(node, dict):
            local = node.get("release_state")
            if isinstance(local, dict) and leaf in local:
                return local[leaf]
            if f"{leaf}_state" in node:
                return node[f"{leaf}_state"]
        return None

    def test_the_walk_sees_money_that_is_a_number_and_not_a_rendered_string(self):
        """Both holes this walk had, pinned on a synthetic payload.

        Labelled synthetic: it is not output from the tool, it is the shape the
        walk must not miss if a section starts serialising money as a number or
        reuses a field name from `spend`.
        """
        synthetic = {
            "release_state": {"input_usd": "draft"},
            "spend": {"input_usd": "$1.00"},
            "new_section": {"input_usd": "$99.00", "projected_usd": 12.5},
        }
        found = self._money_paths(synthetic)
        self.assertIn("new_section.projected_usd", found,
                      "money serialised as a raw number is invisible")
        self.assertIn("new_section.input_usd", found)

    def test_state_is_resolved_by_path_so_a_leaf_name_cannot_borrow_it(self):
        synthetic = {
            "release_state": {"input_usd": "draft"},
            "spend": {"input_usd": "$1.00"},
            "new_section": {"input_usd": "$99.00"},
        }
        self.assertEqual("draft", self._state_for(synthetic, "spend.input_usd"),
                         "the top-level map must still vouch for spend")
        self.assertIsNone(
            self._state_for(synthetic, "new_section.input_usd"),
            "a field borrowed provenance from a same-named field in spend")

    def test_the_clients_own_invoice_is_not_demanded_to_carry_provenance(self):
        """The other direction. `invoice_usd` is an input the client supplied,
        not a figure this tool computed, so requiring release state for it would
        be an over-block -- and name-based detection would otherwise catch it."""
        self.assertEqual([], self._money_paths(
            {"reconciliation": {"invoice_usd": 1.25}}))

    def test_the_allow_list_is_exactly_one_field(self):
        """Guarded in both directions, because an allow-list only removes
        things. Widening it silently removes fields from every check above, and
        the previous version of these tests asserted only that the one entry was
        honoured -- never that it was the only one."""
        self.assertEqual({"reconciliation.invoice_usd"},
                         set(self._INPUT_ONLY_MONEY))

    def test_the_neighbours_of_the_allow_listed_field_are_still_walked(self):
        """The two Figures that sit in the same dict as `invoice_usd`, and which
        the allow-list must not take with it."""
        synthetic = {"reconciliation": {"invoice_usd": 1.25,
                                        "computed_usd": "$1.25",
                                        "delta_usd": "$0.00"}}
        self.assertEqual(["reconciliation.computed_usd",
                          "reconciliation.delta_usd"],
                         sorted(self._money_paths(synthetic)))

    def test_the_allow_list_is_scoped_by_path_not_by_field_name(self):
        """`invoice_usd` anywhere other than the reconciliation block is not the
        client's invoice and has to be walked like any other money."""
        synthetic = {"some_other_section": {"invoice_usd": "$99.00"}}
        self.assertEqual(["some_other_section.invoice_usd"],
                         self._money_paths(synthetic))

    def test_an_inline_state_key_vouches_without_becoming_money_itself(self):
        """`_state_for` supports an inline `<leaf>_state` sibling, and the walk
        promoted it to a money field because the name contains "_usd" -- so one
        of this walk's own supported encodings could not pass it."""
        synthetic = {"reconciliation": {"computed_usd": "$1.25",
                                        "computed_usd_state": "draft"}}
        self.assertEqual(["reconciliation.computed_usd"],
                         self._money_paths(synthetic))
        self.assertEqual("draft",
                         self._state_for(synthetic, "reconciliation.computed_usd"))

    def test_only_spend_is_vouched_for_by_the_top_level_map(self):
        """Pinned because it is a merge-time contract, not a preference.

        `analysis_json` calls `_release_state(mapping)` per section: once at top
        level for `a.spend`, once inside the reconciliation block, once inside
        each finding. So state for anything outside `spend` is local, and this
        resolver requires it to be. If that ever becomes a centralised full-path
        map, this test fails and says so rather than the marker quietly staying
        red through a real fix.
        """
        local = {"release_state": {"input_usd": "draft"},
                 "spend": {"input_usd": "$1.00"},
                 "reconciliation": {"computed_usd": "$1.00",
                                    "release_state": {"computed_usd": "draft"}}}
        self.assertEqual("draft", self._state_for(local, "spend.input_usd"))
        self.assertEqual("draft",
                         self._state_for(local, "reconciliation.computed_usd"))
        full_path = {"release_state": {"reconciliation.computed_usd": "draft"},
                     "reconciliation": {"computed_usd": "$1.00"}}
        self.assertIsNone(
            self._state_for(full_path, "reconciliation.computed_usd"),
            "the serialiser moved to a centralised full-path map; this resolver "
            "and the xfail markers that depend on it need updating together")

    def test_a_findings_release_state_resolves_through_the_list_index(self):
        """The third section `_release_state` is called for, and the only one
        reached through a list index."""
        synthetic = {"findings": [
            {"code": "EFF-1", "avoidable_usd_month": "$542",
             "release_state": {"avoidable_usd_month": "draft"}}]}
        self.assertEqual(["findings[0].avoidable_usd_month"],
                         self._money_paths(synthetic))
        self.assertEqual("draft", self._state_for(
            synthetic, "findings[0].avoidable_usd_month"))

    def test_the_json_walk_finds_money_outside_the_spend_section(self):
        """Guard the guard. The previous version of this test read
        `payload["release_state"]`, which is built from `a.spend` -- so it
        checked the one section it happened to touch and passed while
        `reconciliation.computed_usd` and `delta_usd` shipped as bare dollar
        strings beside it. If the walk only ever found `spend.*`, the assertions
        below would be the same weak test wearing a stronger name."""
        with tempfile.TemporaryDirectory() as tmp:
            found = self._money_paths(self._payload(tmp))
        self.assertTrue(found, "no money-like fields found; the walk is vacuous")
        self.assertTrue([p for p in found if not p.startswith("spend.")],
                        f"the walk only reached the spend section: {sorted(found)}")

    def test_no_money_field_in_the_json_claims_to_be_reconciled(self):
        """What this track's fix is responsible for, over the whole payload."""
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(tmp)
        claimed = sorted(p for p in self._money_paths(payload)
                         if self._state_for(payload, p) == "reconciled")
        self.assertEqual(
            [], claimed,
            "these say an invoice checked them, over an assumed rate table: "
            + ", ".join(claimed))

    # KNOWN-FAILING here only: this is finding M2 (`analysis_json` builds
    # `release_state` from `a.spend` alone), which Track A has already fixed on
    # main with one `_release_state` helper covering spend, reconciliation and
    # findings. This worktree predates that fix, so `reconciliation.computed_usd`
    # and `reconciliation.delta_usd` come back with no state at all. The
    # underlying Figures *are* draft -- `test_and_the_reconciliation_figures_too`
    # pins that on the Analysis -- so this asserts the serialiser carries it.
    # Delete this marker once Track A's `analysis_json` is merged in.
    @unittest.expectedFailure
    def test_every_money_field_in_the_json_carries_release_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._payload(tmp)
        found = self._money_paths(payload)
        self.assertTrue(found, "no money-like fields found; the check is vacuous")
        missing = sorted(p for p in found
                         if self._state_for(payload, p) is None)
        self.assertEqual(
            [], missing,
            "dollar fields a script cannot tell published from withheld: "
            + ", ".join(missing))

    def _exact_invoice(self, root):
        """The invoice this fixture actually reconciles against.

        Both tests below hardcoded "1.16", which stopped reconciling once the
        fixture gained its write-lifetime split: 82.7% out, so every figure was
        withheld. `test_stating_the_surface_from_knowledge_still_reconciles`
        went on passing anyway, because "no DRAFT stamp" is also true of a report
        that published nothing at all -- it was asserting the right thing about
        the wrong state. That is the same trap that once made a real defect in
        this project read as unreproducible, so the invoice is computed and the
        reconciliation is asserted rather than assumed.
        """
        from cacheeconomics.adapters.claude_code import load_sessions
        from cacheeconomics.analyzer import analyze
        ts = load_sessions(root=root, target_id="anthropic/direct")
        spend = analyze(ts, allow_unreconciled=True).spend["input_usd"].raw()
        # `repr`, not a fixed number of decimal places. `f"{spend:.10f}"` was
        # close enough to look right and left delta_pct at 2.8e-14 rather than
        # 0.0, so the guard below failed on the rounding rather than on anything
        # this class is about.
        return repr(spend)

    def test_the_fixture_reconciles_against_the_computed_invoice(self):
        """Guard the guard for the two tests below: both are about the release
        *label*, and both are meaningless if the gate withheld everything."""
        from cacheeconomics.adapters.claude_code import load_sessions
        from cacheeconomics.analyzer import analyze
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fixture(tmp)
            ts = load_sessions(root=root, target_id="anthropic/direct")
            a = analyze(ts, invoice_usd=float(self._exact_invoice(root)))
        self.assertEqual(0.0, a.reconciliation["delta_pct"])
        self.assertTrue(any(getattr(v, "released", False) for v in a.spend.values()))

    def test_stating_the_surface_from_knowledge_still_reconciles(self):
        """The other direction, so this does not become an over-block.
        `--target-id anthropic/direct` is the same string arrived at by
        knowledge, and nothing about it is assumed."""
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fixture(tmp)
            code, out, _err = run("claude-code", "--root", root,
                                  "--target-id", "anthropic/direct",
                                  "--invoice-usd", self._exact_invoice(root))
        self.assertEqual(code, 0)
        # The short report carries no "$" even when it publishes -- the only
        # finding here is "not costed" -- so the probe for "something was
        # actually released" is the absence of the withheld banner.
        self.assertNotIn("FIGURES WITHHELD", out,
                         "nothing was published, so 'no DRAFT stamp' proves nothing")
        self.assertNotIn("DRAFT", out)

    def test_the_flag_goes_through_the_command_end_to_end(self):
        """The helper is only a floor if `cmd_claude_code` actually applies it."""
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fixture(tmp)
            code, out, _err = run("claude-code", "--root", root,
                                  "--assume-anthropic-direct",
                                  "--invoice-usd", self._exact_invoice(root))
        self.assertEqual(code, 0)
        self.assertNotIn("FIGURES WITHHELD", out,
                         "nothing was published, so the stamp is moot")
        self.assertIn("DRAFT", out)
        self.assertIn("surface was assumed", out)

    def test_it_never_launders_a_withheld_figure_into_a_released_one(self):
        """A downgrade only. Re-releasing everything would turn the figures the
        gate withheld into published drafts, which is the opposite failure."""
        import dataclasses

        from cacheeconomics.money import MEASURED, Figure
        with tempfile.TemporaryDirectory() as tmp:
            base = self._analysis(tmp, "anthropic/direct")
        held = Figure(99.0, MEASURED, released=False,
                      withheld_because="unprovable write lifetime")
        a = cli._draft_because_the_surface_was_assumed(
            dataclasses.replace(base, spend={"held": held, **base.spend}))
        self.assertFalse(a.spend["held"].released)
        self.assertEqual("unprovable write lifetime",
                         a.spend["held"].withheld_because)
        self.assertNotIn("99", str(a.spend["held"]))

    def test_an_already_draft_figure_stays_draft(self):
        import dataclasses

        from cacheeconomics.money import DRAFT, MEASURED, Figure
        with tempfile.TemporaryDirectory() as tmp:
            base = self._analysis(tmp, "anthropic/direct")
        a = cli._draft_because_the_surface_was_assumed(
            dataclasses.replace(base, spend={
                "old": Figure(5.0, MEASURED, released=True, released_as=DRAFT),
                **base.spend}))
        self.assertEqual(DRAFT, a.spend["old"].released_as)

    def test_the_assumption_is_recorded_as_a_blocking_note(self):
        """`blocking_notes` is the list of notes that qualify a published
        figure, which is exactly what this now is."""
        with tempfile.TemporaryDirectory() as tmp:
            a = cli._draft_because_the_surface_was_assumed(
                self._analysis(tmp, "anthropic/direct"))
        self.assertTrue(any("surface was assumed" in n for n in a.blocking_notes))


class TestTheDraftBannerFollowsTheFiguresNotTheFlag(unittest.TestCase):
    """The assumed-surface banner used to be prepended unconditionally.

    Two defects came out of that, and they pull in opposite directions, so both
    are asserted here.

    A report whose reconciliation *failed* has every figure withheld -- there is
    no draft to announce. The note went in anyway, and because `render_text`
    looks for a note beginning "DRAFT" while `render_html` calls `_is_draft`
    (which reads release state off the figures), the two renderers reached
    opposite verdicts about the same report. Measured on a $999 invoice against
    $1.16 of computed spend: `_is_draft` False, HTML gate div absent, text
    banner present.

    And with `--allow-unreconciled` and no invoice, the analyzer's own "figures
    released without invoice reconciliation" sat at notes[0] until this
    prepended in front of it. `report._draft_reason` takes the first DRAFT note,
    so the reader was told instead that passing `--target-id` would make these
    reconciled figures -- which, with no invoice supplied, is false. A true
    explanation had been replaced by a false one.
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
                        "cache_creation": {
                            "ephemeral_5m_input_tokens": 1_000,
                            "ephemeral_1h_input_tokens": 0}}}}) + "\n")
        return tmp

    def _analysis(self, tmp, **kw):
        from cacheeconomics.adapters.claude_code import load_sessions
        from cacheeconomics.analyzer import analyze
        ts = load_sessions(root=self._fixture(tmp), target_id="anthropic/direct",
                           surface_assumed=True)
        if kw.get("invoice_usd") == "exact":
            kw["invoice_usd"] = analyze(ts, allow_unreconciled=True).spend[
                "input_usd"].raw()
        return cli._draft_because_the_surface_was_assumed(analyze(ts, **kw))

    def _verdicts(self, a):
        """What each renderer concludes, by the route each actually uses."""
        from cacheeconomics import report
        return {
            "figures": report._is_draft(a),
            "text": any(n.startswith("DRAFT") for n in a.notes),
            "html": "DRAFT — not for external" in report.render_html(a),
        }

    def test_a_failed_reconciliation_withholds_everything(self):
        """Guard the guard: if this fixture released anything, the case below
        would not be the one it is named for."""
        with tempfile.TemporaryDirectory() as tmp:
            a = self._analysis(tmp, invoice_usd=999.0)
        released = [k for k, v in a.spend.items() if getattr(v, "released", False)]
        self.assertEqual([], released)

    def test_and_then_no_renderer_claims_it_is_a_draft(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self._analysis(tmp, invoice_usd=999.0)
        v = self._verdicts(a)
        self.assertEqual({False}, set(v.values()),
                         f"renderers disagree about the draft verdict: {v}")

    def test_the_renderers_agree_on_every_reachable_release_state(self):
        """The general form, over all three ways this command reaches the
        renderers. Two paths deciding one verdict from two different signals is
        the defect this repo has the longest history with."""
        cases = {
            "reconciles": dict(invoice_usd="exact"),
            "fails to reconcile": dict(invoice_usd=999.0),
            "no invoice, override": dict(allow_unreconciled=True),
            "no invoice, no override": {},
        }
        for label, kw in cases.items():
            with self.subTest(case=label):
                with tempfile.TemporaryDirectory() as tmp:
                    a = self._analysis(tmp, **kw)
                v = self._verdicts(a)
                self.assertEqual(
                    1, len(set(v.values())),
                    f"{label}: renderers disagree about the draft verdict: {v}")

    def test_the_true_reason_is_not_replaced_by_the_surface_one(self):
        from cacheeconomics import report
        with tempfile.TemporaryDirectory() as tmp:
            a = self._analysis(tmp, allow_unreconciled=True)
        reason = report._draft_reason(a)
        self.assertIn("without invoice reconciliation", reason,
                      "the analyzer's own reason was displaced")
        self.assertIn("surface was assumed", reason,
                      "the surface assumption went unmentioned")

    def test_and_does_not_promise_reconciliation_that_cannot_happen(self):
        """With no invoice supplied, "pass --target-id and these become
        reconciled figures" is false: they would still be drafts."""
        from cacheeconomics import report
        with tempfile.TemporaryDirectory() as tmp:
            a = self._analysis(tmp, allow_unreconciled=True)
        self.assertNotIn("become reconciled figures", report._draft_reason(a))

    def test_there_is_exactly_one_draft_banner(self):
        """Composed into the existing reason, not stacked beside it. Two DRAFT
        notes means `_draft_reason` silently picks one and drops the other."""
        with tempfile.TemporaryDirectory() as tmp:
            a = self._analysis(tmp, allow_unreconciled=True)
        self.assertEqual(1, sum(1 for n in a.notes if n.startswith("DRAFT")))

    def test_a_reconciling_invoice_still_gets_the_surface_reason(self):
        """The path with no competing reason still has to state this one."""
        from cacheeconomics import report
        with tempfile.TemporaryDirectory() as tmp:
            a = self._analysis(tmp, invoice_usd="exact")
        self.assertTrue(report._is_draft(a))
        self.assertIn("surface was assumed", report._draft_reason(a))


class TestTheAssumedSurfaceNoteKeysOnProvenanceNotOnTheString(unittest.TestCase):
    """`--target-id anthropic/direct` is the same string as an assumed one.

    The adapter emitted its "Provider surface assumed to be anthropic/direct"
    blocking note on `target_id == "anthropic/direct"`, so a user who stated the
    surface from knowledge was told it was an assumption. The figures were
    correctly reconciled; only the prose was wrong -- which is the same defect as
    the one this round is about, in the half nobody was looking at.
    """

    def _ts(self, tmp, **kw):
        from cacheeconomics.adapters.claude_code import load_sessions
        proj = os.path.join(tmp, "proj")
        os.makedirs(proj)
        with open(os.path.join(proj, "s.jsonl"), "w") as f:
            for i in range(4):
                f.write(json.dumps({
                    "type": "assistant", "sessionId": "s1", "uuid": f"u{i}",
                    "requestId": f"r{i}",
                    "timestamp": f"2026-07-29T09:0{i}:00.000Z",
                    "message": {"model": "claude-opus-5", "usage": {
                        "input_tokens": 100, "output_tokens": 10,
                        "cache_read_input_tokens": 20_000,
                        "cache_creation_input_tokens": 1_000}}}) + "\n")
        return load_sessions(root=tmp, target_id="anthropic/direct", **kw)

    def test_a_stated_surface_is_not_reported_as_assumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            ts = self._ts(tmp)
        self.assertTrue(ts.requests, "fixture produced no requests")
        self.assertEqual([], [n for n in ts.blocking_notes if "assumed" in n])

    def test_an_assumed_surface_still_is(self):
        with tempfile.TemporaryDirectory() as tmp:
            ts = self._ts(tmp, surface_assumed=True)
        self.assertTrue(any("assumed" in n for n in ts.blocking_notes))

    def test_the_two_differ_only_in_how_the_surface_was_arrived_at(self):
        """Same root, same target_id, same rows -- only the provenance differs,
        which is precisely what cannot be recovered from the value."""
        with tempfile.TemporaryDirectory() as tmp:
            stated = self._ts(tmp, surface_assumed=False)
        with tempfile.TemporaryDirectory() as tmp:
            assumed = self._ts(tmp, surface_assumed=True)
        self.assertEqual([r.target_id for r in stated.requests],
                         [r.target_id for r in assumed.requests])
        self.assertNotEqual(stated.blocking_notes, assumed.blocking_notes)

    def test_the_command_states_it_only_for_the_flag(self):
        """End to end, both ways round."""
        import tempfile as tf
        with tf.TemporaryDirectory() as tmp:
            self._ts(tmp)          # writes the transcript
            _c, stated, _e = run("claude-code", "--root", tmp,
                                 "--target-id", "anthropic/direct",
                                 "--allow-unreconciled")
            _c, assumed, _e = run("claude-code", "--root", tmp,
                                  "--assume-anthropic-direct",
                                  "--allow-unreconciled")
        self.assertNotIn("surface assumed", " ".join(stated.split()))
        self.assertIn("surface assumed", " ".join(assumed.split()))
