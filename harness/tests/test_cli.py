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


class TestEveryDollarFieldInTheJsonCarriesItsReleaseState(unittest.TestCase):
    """`--format json` is what a script reads, and a string is not a state.

    `release_state` was built from `a.spend` and nothing else. Measured on a
    two-request trace with a reconciling invoice: the payload carried seven
    money-shaped fields and three of them -- `findings[0].avoidable_usd_month`,
    `reconciliation.computed_usd`, `reconciliation.delta_usd` -- had no state
    anywhere beside them. A consumer could not tell a published figure from a
    withheld or a draft one, which is the machine-readable half of the DRAFT
    banner and the whole reason the gate exists.

    The fields are found by scanning the decoded payload for anything that looks
    like money, rather than by naming the sections. Naming them is what shipped
    the defect: a test called "every figure" checked the two sections its author
    had in mind.
    """

    def _payload(self, *extra):
        code, out, err = run("analyze", FIXTURE, "--format", "json", *extra)
        self.assertEqual(code, 0, err)
        return json.loads(out)

    @staticmethod
    def _money_paths(node, path=""):
        """Every path whose value looks like a rendered `Figure`.

        A Figure serialises through `str`, so it is either "$..." or
        "[withheld: ...]". Both are money: a withheld one still has to be
        identifiable as withheld rather than simply absent.

        Plain numbers are deliberately excluded. `window_days` and `delta_pct`
        sit in the same dicts, and flagging them would make this permanently red
        over a defect that does not exist -- which is how a check gets switched
        off and then protects nothing.
        """
        import re
        out = []
        if isinstance(node, str):
            if node.startswith("[withheld") or re.fullmatch(
                    r"\$-?[\d,]+(?:\.\d+)?", node.strip()):
                out.append(path)
        elif isinstance(node, dict):
            for k, v in node.items():
                if k == "release_state":
                    continue
                out += TestEveryDollarFieldInTheJsonCarriesItsReleaseState \
                    ._money_paths(v, f"{path}.{k}" if path else k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                out += TestEveryDollarFieldInTheJsonCarriesItsReleaseState \
                    ._money_paths(v, f"{path}[{i}]")
        return out

    @staticmethod
    def _state_for(payload, path):
        """The release state recorded for the money at `path`, or None.

        Two placements are accepted, because the payload legitimately uses two.
        `spend` carries its map at the root, keyed by field name -- that is the
        published shape and consumers read it. The nested sections carry theirs
        beside the values, which the root map could not express: every finding
        has an `avoidable_usd_month` and one flat map cannot key them apart.

        Where a section puts its map is that section's business. Having neither
        is the defect, and that is what this returns None for.
        """
        parts = path.split(".")
        leaf = parts[-1]
        node = payload
        for p in parts[:-1]:
            if "[" in p:
                name, idx = p.split("[")
                node = node[name][int(idx.rstrip("]"))]
            else:
                node = node[p]
        states = node.get("release_state") if isinstance(node, dict) else None
        if isinstance(states, dict) and leaf in states:
            return states[leaf]
        root = payload.get("release_state") or {}
        return root[leaf] if leaf in root else None

    def test_there_are_money_fields_to_check(self):
        found = self._money_paths(self._payload("--allow-unreconciled"))
        self.assertTrue(found, "no money-like fields found; this would pass "
                               "while checking nothing")

    def test_every_money_field_has_release_state(self):
        for extra in (("--allow-unreconciled",), ("--invoice-usd", "17.45"), ()):
            payload = self._payload(*extra)
            with self.subTest(args=extra or ("(no invoice)",)):
                missing = [p for p in self._money_paths(payload)
                           if self._state_for(payload, p) is None]
                self.assertEqual(
                    [], missing,
                    "dollar fields with no release state, so a script cannot "
                    "tell a published figure from a withheld or draft one:\n"
                    "    " + "\n    ".join(missing))

    def test_the_state_distinguishes_a_draft_from_an_invoice_checked_figure(self):
        """Presence of the key is not the claim. A state map that answered ""
        everywhere would satisfy the test above and tell a consumer nothing."""
        draft = self._payload("--allow-unreconciled")
        checked = self._payload("--invoice-usd", "17.45")
        states = {p: self._state_for(draft, p) for p in self._money_paths(draft)}
        self.assertIn("draft", set(states.values()),
                      f"--allow-unreconciled marked nothing as a draft: {states}")
        self.assertIn("reconciled",
                      {self._state_for(checked, p)
                       for p in self._money_paths(checked)})

    def test_the_caveats_come_before_every_dollar_bearing_section(self):
        """The ordering the other two renderers already keep.

        Text and HTML both print the caveats above the first figure they
        qualify. This one put every figure ahead of all of them and carried the
        caveats nowhere at all -- `blocking_notes` was absent from the payload,
        so a consumer had to recover them by matching prose in `notes`, which is
        the thing a machine-readable surface exists to avoid.

        Asserted on key order rather than presence. JSON objects preserve
        insertion order through `json.dumps`, and order is the whole claim: a
        caveat a consumer reaches after the figure is doing the same amount of
        work as one printed below it.
        """
        payload = self._payload("--allow-unreconciled")
        keys = list(payload)
        self.assertIn("caveats", keys, "the payload carries no caveats field")
        money_sections = [k for k in ("spend", "reconciliation", "findings")
                          if k in keys]
        self.assertTrue(money_sections)
        for section in money_sections:
            with self.subTest(section=section):
                self.assertLess(keys.index("caveats"), keys.index(section),
                                f"{section} is serialised before the caveats "
                                f"that qualify it")

    def test_the_caveats_are_the_ones_the_analysis_recorded(self):
        """Presence in the right place is not enough; it has to be the same
        list the other two renderers print, or this is a third opinion."""
        from cacheeconomics.analyzer import analyze, spend_caveats
        from cacheeconomics.trace import load_jsonl
        a = analyze(load_jsonl(FIXTURE), allow_unreconciled=True)
        payload = self._payload("--allow-unreconciled")
        self.assertEqual(list(spend_caveats(a)), payload["caveats"])

    def test_forwardability_is_one_boolean_rather_than_a_scan(self):
        draft = self._payload("--allow-unreconciled")
        checked = self._payload("--invoice-usd", "17.45")
        self.assertTrue(draft["draft"])
        self.assertFalse(checked["draft"])

    def test_a_withheld_figure_says_so_rather_than_claiming_a_release(self):
        """The run with no invoice releases nothing, so no field may report a
        provenance. A state that survived the gate would be worse than none."""
        payload = self._payload()
        for p in self._money_paths(payload):
            with self.subTest(field=p):
                self.assertEqual("", self._state_for(payload, p),
                                 "withheld figure carries release provenance")


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
        bad, _o, _e = run("checks", "--prefix-tokens", "300",
                          "--model", "claude-opus-5", "--breakpoints", "5")
        good, _o, _e = run("checks", "--prefix-tokens", "9000",
                           "--model", "claude-opus-5", "--breakpoints", "2")
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
