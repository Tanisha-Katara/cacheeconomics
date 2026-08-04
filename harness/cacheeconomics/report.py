"""The forwarding object.

The CLI is how the report gets made; this file is the thing a champion sends to
their VP. It is written for someone who did not run it and will not run it.

Every figure carries its evidence class, coverage is stated before any number,
and if reconciliation misses the gate the dollar figures are withheld rather
than printed with a caveat underneath.
"""

from __future__ import annotations

import html
import textwrap
from datetime import datetime, timezone

from .analyzer import Analysis, spend_caveats
from .money import Figure
from .trace import Tier

_CSS = """
:root{color-scheme:light;--bg:#fbfbf9;--panel:#fff;--ink:#111110;--ink2:#4d4b46;
--ink3:#7c7972;--rule:#dfddd6;--fill:#f2f1ed;--ok:#0b7f37;--okbg:#eaf5ee;
--warn:#9b6b0b;--warnbg:#fff5df;--crit:#b63229;--critbg:#fff0ee;--accent:#2457a6}
*{box-sizing:border-box}
html{-webkit-font-smoothing:antialiased}
body{margin:0;background:var(--bg);color:var(--ink);
font:400 16px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.report{max-width:1060px;margin:0 auto;padding:68px 34px 110px}
.hero{display:grid;grid-template-columns:minmax(0,1fr) 280px;gap:34px;align-items:end;
padding-bottom:28px}
.eyebrow,.label{font:600 11px/1 ui-monospace,SFMono-Regular,Menlo,monospace;
letter-spacing:.12em;text-transform:uppercase;color:var(--ink3)}
h1{font-size:clamp(38px,5vw,58px);line-height:1.02;letter-spacing:-.035em;
font-weight:560;margin:12px 0 16px;max-width:13ch}
h2{font-size:clamp(24px,3vw,32px);line-height:1.15;letter-spacing:-.025em;
font-weight:560;margin:0}
h3{font-size:18px;line-height:1.25;font-weight:650;margin:0;color:var(--ink)}
p{margin:0;color:var(--ink2);max-width:68ch}
.lede{font-size:19px;line-height:1.55;color:var(--ink2);max-width:58ch}
.meta{display:grid;gap:10px;border:1px solid var(--rule);background:var(--panel);
padding:18px;border-radius:6px}
.meta div{display:flex;justify-content:space-between;gap:16px;border-bottom:1px solid var(--rule);
padding-bottom:9px}
.meta div:last-child{border-bottom:0;padding-bottom:0}
.meta span:first-child{color:var(--ink3)}
.meta span:last-child{font-weight:560;color:var(--ink);text-align:right}
.verdict{border:1px solid var(--rule);background:var(--panel);border-radius:6px;padding:22px;
display:grid;grid-template-columns:minmax(0,1.3fr) minmax(220px,.7fr);gap:24px;margin:12px 0 28px}
.verdict strong{display:block;color:var(--ink);font-size:22px;line-height:1.25;margin-top:8px}
.impact{border-left:3px solid var(--crit);padding-left:16px}
.impact.ok{border-left-color:var(--ok)}
.impact .amount{font-size:42px;line-height:1;font-weight:560;letter-spacing:-.035em;
font-variant-numeric:tabular-nums;color:var(--ink)}
.impact .caption{margin-top:8px;color:var(--ink3);font-size:14px}
.status{display:flex;align-items:center;gap:10px;margin-top:16px;color:var(--ink2)}
.dot{width:9px;height:9px;border-radius:99px;background:var(--ok);flex:none}
.dot.warn{background:var(--warn)}.dot.crit{background:var(--crit)}
.section{border-top:1px solid var(--rule);padding-top:18px;margin-top:54px}
.section-head{display:flex;justify-content:space-between;gap:24px;align-items:end;margin-bottom:22px}
.section-head p{font-size:15px;max-width:46ch}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:var(--rule);
border:1px solid var(--rule);border-radius:6px;overflow:hidden}
.kpi{background:var(--panel);padding:20px 18px;min-height:126px}
.kpi .value{font-size:31px;line-height:1.1;font-weight:560;letter-spacing:-.025em;
font-variant-numeric:tabular-nums;margin:15px 0 7px}
.kpi .help{font-size:14px;color:var(--ink2)}
.strip{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-top:16px}
.strip .box{border:1px solid var(--rule);background:var(--panel);border-radius:6px;padding:17px}
.box strong{display:block;margin-bottom:6px;color:var(--ink)}
.findings{display:grid;gap:16px}
.finding{border:1px solid var(--rule);border-left-width:4px;background:var(--panel);
border-radius:6px;padding:20px}
.finding.high{border-left-color:var(--crit)}.finding.medium{border-left-color:var(--warn)}
.finding.low{border-left-color:var(--ink3)}
.finding-head{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:16px;align-items:start}
.badges{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:10px}
.tag{font:600 10px/1 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.09em;
text-transform:uppercase;padding:6px 8px;border:1px solid var(--rule);border-radius:999px;
color:var(--ink3);background:var(--bg);white-space:nowrap}
.tag.measured{color:var(--ok);background:var(--okbg);border-color:transparent}
.tag.modeled{color:var(--warn);background:var(--warnbg);border-color:transparent}
.tag.risk{color:var(--ink2)}
.finding-money{text-align:right;min-width:138px;font-variant-numeric:tabular-nums}
.finding-money b{display:block;font-size:24px;line-height:1.1;font-weight:560;color:var(--ink)}
.finding-money span{display:block;font-size:12px;color:var(--ink3);margin-top:5px}
.finding-body{display:grid;grid-template-columns:minmax(0,1fr) minmax(220px,.42fr);gap:24px;
margin-top:16px}
.block-label{font:600 10px/1 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.1em;
text-transform:uppercase;color:var(--ink3);margin-bottom:8px}
.fix{border-left:2px solid var(--accent);padding-left:14px;color:var(--ink2);font-size:15px}
.total{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;border-top:1px solid var(--rule);
padding-top:20px;margin-top:4px}
.total .amount{font-size:42px;line-height:1;font-weight:560;letter-spacing:-.035em}
.note{font-size:14px;color:var(--ink3)}
table{border-collapse:collapse;width:100%;font-size:15px;background:var(--panel);
border:1px solid var(--rule);border-radius:6px;overflow:hidden}
th{text-align:left;font:600 10px/1 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.1em;
text-transform:uppercase;color:var(--ink3);padding:13px 16px;border-bottom:1px solid var(--rule)}
td{padding:14px 16px;border-bottom:1px solid var(--rule);color:var(--ink2)}
tr:last-child td{border-bottom:0}td:first-child{color:var(--ink);font-weight:560}
.gate{padding:16px 18px;border:1px solid var(--crit);background:var(--critbg);border-radius:6px}
footer{border-top:1px solid var(--rule);margin-top:72px;padding-top:18px;font-size:13px;color:var(--ink3)}
@media(max-width:850px){.report{padding:48px 22px 80px}.hero,.verdict,.finding-body{grid-template-columns:1fr}
.kpis,.strip{grid-template-columns:1fr 1fr}.finding-head{grid-template-columns:1fr}
.finding-money{text-align:left}.section-head{display:block}.section-head p{margin-top:10px}}
@media(max-width:560px){.kpis,.strip{grid-template-columns:1fr}.meta div{display:block}
.meta span:last-child{text-align:left;display:block;margin-top:3px}}
"""

_SEV = {"high": "var(--crit)", "medium": "var(--warn)", "low": "var(--ink3)"}


def _pct(v):
    return "—" if v is None else f"{v*100:.0f}%"


def _usd(v):
    """Render a Figure. A withheld one renders as withheld, not as a number.

    The renderers no longer carry a gate. Two of them once disagreed about the
    same reconciliation rule, so the rule moved into the value: str() on an
    unreleased Figure returns "[withheld: ...]" and float() on one raises.
    """
    if v is None:
        return "—"
    if isinstance(v, Figure):
        return str(v)
    # Plain float: the reconciliation numbers. Those always render, including
    # when the gate fails, because they are the evidence for why it failed.
    return f"${v:,.2f}"


def _is_draft(a: Analysis) -> bool:
    """Does this analysis carry figures released without an invoice?

    Read off the figures rather than off the notes. Matching a note by its text
    is how the HTML and text renderers came to disagree: one searched the notes
    and the other rendered them wherever they happened to sit.
    """
    from .money import DRAFT, Figure
    return any(isinstance(v, Figure) and v.released_as == DRAFT
               for v in a.spend.values())


def _draft_reason(a: Analysis) -> str:
    return next((n for n in a.notes if n.startswith("DRAFT")),
                "Figures were released without invoice reconciliation.")


def render_html(a: Analysis, client: str = "", window_label: str = "") -> str:
    e = html.escape
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    client_name = e(client) or "Diagnostic"
    cov = a.coverage
    cov_line = (f"{cov['analysed']:,} of {cov['total']:,} requests analyzed "
                f"({_pct(cov['fraction'])})")
    if cov["excluded"]:
        cov_line += ". Excluded: " + ", ".join(f"{v:,} {k}" for k, v in cov["excluded"].items())

    saved = a.spend.get("caching_saved_usd")
    impact_class = "impact"
    if saved is not None and saved.released:
        if saved.raw() >= 0:
            impact_class = "impact ok"
            verdict = f"Caching saved {_usd(saved)} during the measured window."
            verdict_detail = "The current prefix is earning back its write cost."
        else:
            verdict = f"Caching added {_usd(abs(saved))} during the measured window."
            verdict_detail = ("The cache is being written more often than it is being read. "
                              "Until the prefix is stable, caching can cost more than plain input.")
    else:
        verdict = "Dollar impact is withheld until reconciliation passes."
        verdict_detail = ("The structure can still be diagnosed, but no spend figure should be "
                          "used until computed usage ties to the invoice.")

    total = a.total_avoidable_month
    total_line = (_usd(total) if total.released else "withheld")
    total_detail = ("per month, modeled pessimistically. Findings overlap, so this is an upper "
                    "bound on the sum, not a promise.")

    recon_label = "Not supplied"
    recon_ok = None
    recon_text = "No invoice was supplied, so spend was not independently reconciled."
    if a.reconciliation:
        r = a.reconciliation
        recon_ok = bool(r["within_ship_gate"])
        # One place that decides how a null difference reads, because there were
        # three copies of `abs(delta_pct)` in this file and fixing two of them
        # left the zero-invoice case still crashing.
        _delta = r.get("delta_pct")
        # Only reached with a valid invoice below, so `delta_pct is None` no
        # longer has to stand in for four different rejections.
        _diff = (f"Difference: {abs(_delta):.1f}%"
                 if _delta is not None else "No difference could be computed")
        # An invoice the analyzer rejected is not money, so it must not be
        # formatted as money. `_usd()` raised ValueError on a non-numeric one and
        # took the whole HTML deliverable down with it, and a negative or
        # non-finite one rendered as "The invoice is zero" -- naming a blocker
        # that is not the real one. The analyzer grew `invalid_invoice` last
        # change and this renderer was not taught to read it: the same guard in
        # one path and not its twin, which is the shape this branch keeps
        # producing.
        _invalid = r.get("invalid_invoice")
        if _invalid:
            recon_label = "Failed"
            recon_text = {
                "zero": "The invoice supplied is zero, so computed spend cannot "
                        "be reconciled against it. Usually the export and the "
                        "bill do not describe the same period.",
                "negative": f"The invoice supplied is negative "
                            f"({r.get('invoice_usd')}). A credit or refund is not "
                            f"a spend total, and reading one as the denominator "
                            f"turns any mismatch into a negative percentage.",
                "not-finite": "The invoice supplied is not a finite number, so "
                              "computed spend cannot be reconciled against it.",
            }.get(_invalid,
                  "The invoice supplied is not a number, so computed spend "
                  "cannot be reconciled against it.")
            recon_text = (f"Computed input spend {_usd(r['computed_usd'])}. "
                          + recon_text)
        elif recon_ok:
            recon_label = "Passed"
            recon_text = (f"Computed input spend {_usd(r['computed_usd'])} against invoice "
                          f"{_usd(r['invoice_usd'])}. {_diff}, "
                          "inside the 5% publication gate.")
        else:
            # Which condition actually failed. The card said "outside the 5%
            # publication gate" for every failure, so an invoice matching the
            # priced subtotal exactly, blocked by one row on an unknown model,
            # rendered as "Difference: 0.0%, outside the 5% publication gate" --
            # a sentence that reads as a defect in the tool. Same fix the text
            # report already had; the two renderers diverged on it.
            b = r.get("blockers") or {}
            excluded = [(n, label) for n, label in (
                (b.get("unprovable_lifetime", 0), "with cache writes of unprovable lifetime"),
                (b.get("unpriceable_model", 0), "on a model or surface with no recorded price"),
                (b.get("undated", 0), "with no usable timestamp"),
                (b.get("skipped_rows", 0), "the loader could not read"),
                (b.get("no_usage", 0), "carrying no usage fields"),
                (b.get("failed_but_billed", 0), "that failed but still billed")) if n]
            recon_label = "Failed"
            if not b.get("delta", True) and excluded:
                recon_text = (
                    f"Computed {_usd(r['computed_usd'])} against invoice "
                    f"{_usd(r['invoice_usd'])}. {_diff}, but that subtotal excludes "
                    + "; ".join(f"{n} request(s) {label}" for n, label in excluded)
                    + " — a subset that happens to agree, not a reconciliation.")
            else:
                recon_text = (f"Computed {_usd(r['computed_usd'])} against invoice "
                              f"{_usd(r['invoice_usd'])}. {_diff}, "
                              "outside the 5% publication gate.")
    status_dot_class = "dot warn" if recon_ok is False else "dot"

    parts = [
        "<!doctype html><html><head><meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        f"<title>{client_name} cache economics assessment</title>",
        f"<style>{_CSS}</style></head><body><main class=report>",
        "<header class=hero><div>",
        "<div class=eyebrow>Cache economics assessment</div>",
        "<h1>What your prompt cache is actually doing.</h1>",
        f"<p class=lede>{verdict} {verdict_detail}</p>",
        "</div><div class=meta>",
        f"<div><span>Client</span><span>{client_name}</span></div>",
        f"<div><span>Window</span><span>{e(window_label) if window_label else 'Not specified'}</span></div>",
        f"<div><span>Generated</span><span>{generated}</span></div>",
        f"<div><span>Ingest tier</span><span>{e(str(a.tier))}</span></div>",
        "</div></header>",
    ]
    # Before the hero, and therefore before any dollar figure. This used to
    # arrive with the ordinary notes in the Standing section: measured on the
    # demo trace, the first "$" was at character 6,554 and the first "DRAFT" at
    # 14,510, so a forwarded report read as client-ready for eight thousand
    # characters. The text renderer already led with it; only HTML did not,
    # which is the twin-path shape one more time.
    #
    # Inserted rather than rendered as an empty string when absent, so a
    # reconciled report is byte-identical to what it was.
    if _is_draft(a):
        # Index 4: immediately after `</head><body><main class=report>` and
        # before `<header class=hero>`. Index 1 put a div inside <head>, which
        # renders nowhere -- a stamp that is present in the source and invisible
        # on the page is worse than no stamp, because it passes a substring test.
        parts.insert(4, f"<div class=gate><strong>DRAFT — not for external "
                        f"use.</strong> {e(_draft_reason(a))}</div>")

    parts.append("<section class=verdict><div>")
    parts.append("<div class=label>Executive readout</div>")
    parts.append(f"<strong>{verdict}</strong>")
    parts.append(f"<p style='margin-top:10px'>{verdict_detail}</p>")
    parts.append(f"<div class=status><span class='{status_dot_class}'></span>"
                 f"<span>Reconciliation: {e(recon_label)}. Coverage: {e(cov_line)}.</span></div>")
    parts.append(f"</div><div class='{impact_class}'>")
    parts.append("<div class=label>Recoverable upper bound</div>")
    parts.append(f"<div class=amount>{e(total_line)}</div>")
    parts.append(f"<div class=caption>{e(total_detail)}</div>")
    parts.append("</div></section>")

    # coverage first: a number without it is not interpretable
    parts.append("<section class=section><div class=section-head>")
    parts.append("<div><div class=label>01 / Data quality</div><h2>Can we trust the numbers?</h2></div>")
    parts.append("<p>Coverage and invoice reconciliation are shown before any recommendation, "
                 "because a cost figure without those checks is just arithmetic.</p></div>")
    parts.append("<div class=strip>")
    parts.append(f"<div class=box><strong>Coverage</strong><p>{e(cov_line)}.</p></div>")
    recon_class = "gate" if recon_ok is False else "box"
    parts.append(f"<div class={recon_class}><strong>Reconciliation: {e(recon_label)}</strong>"
                 f"<p style='margin-top:6px'>{e(recon_text)}</p></div>")
    parts.append("</div>")
    if cov["fraction"] is not None and cov["fraction"] < 0.95:
        parts.append("<p class=note>A material share of requests could not be analysed. "
                     "Figures below describe the analysed subset only.</p>")
    parts.append("</section>")

    # headline
    parts.append("<section class=section><div class=section-head>")
    parts.append(f"<div><div class=label>02 / Current position</div>"
                 f"<h2>{e(_headline(a))}</h2></div>")
    parts.append("<p>These are the ratios finance and engineering can both read: how much input "
                 "came from cache, how often writes paid off, and what the run costs at your rate.</p>")
    parts.append("</div><div class=kpis>")
    cells = [
        ("Input from cache", _pct(a.ratios["input_from_cache"]),
         "Share of input tokens served at the cheap read rate."),
        ("Prefix efficiency", _pct(a.ratios["prefix_efficiency"]),
         "Of every token written to cache, the share later read."),
        ("Requests", f"{a.ratios['requests']:,}",
         f"Observed over {a.window_days:.1f} days." if a.window_days else "Observed requests."),
    ]
    cells.append(("Input spend", _usd(a.spend.get("monthly_input_usd")),
                  "Per month, extrapolated from this window."))
    for k, v, s in cells:
        parts.append(f"<div class=kpi><div class=label>{e(k)}</div><div class=value>{e(v)}</div>"
                     f"<div class=help>{e(s)}</div></div>")
    parts.append("</div></section>")

    # findings
    parts.append("<section class=section><div class=section-head>")
    # Was the literal "Fix the prefix before changing lifetime." on every report,
    # which is advice and was wrong wherever the top finding was not a prefix
    # one. The findings are already ranked; name the first one instead.
    _top = a.findings[0].title if a.findings else "Nothing to act on."
    parts.append(f"<div><div class=label>03 / Findings - {len(a.findings)}</div>"
                 f"<h2>{e(_top)}</h2></div>")
    parts.append("<p>Findings are ranked by severity and monthly impact. Each one names the "
                 "mechanism, the evidence class, and the safest next action.</p></div>")
    parts.append("<div class=findings>")
    if not a.findings:
        parts.append("<p>No material cache-related waste identified. That is a valid outcome "
                     "and it means there is nothing here worth changing.</p>")
    for f in a.findings:
        money = ("<div class=finding-money><b>"
                 f"~{e(_usd(f.avoidable_usd_month))}</b><span>avoidable / month</span></div>"
                 if f.avoidable_usd_month else "")
        basis_class = "measured" if f.evidence_class == "measured" else "modeled"
        parts.append(
            f"<article class='finding {e(f.severity)}'><div class=finding-head><div>"
            f"<div class=badges><span class=tag>{e(f.code)}</span>"
            f"<span class='tag {basis_class}'>{e(f.evidence_class)}</span>"
            f"<span class='tag risk'>quality risk: {e(f.quality_risk)}</span></div>"
            f"<h3>{e(f.title)}</h3></div>{money}</div>"
            "<div class=finding-body><div><div class=block-label>Evidence</div>"
            f"<p>{e(f.detail)}</p></div><div><div class=block-label>Action</div>"
            f"<div class=fix>{e(f.fix)}</div></div></div></article>")
    if a.total_avoidable_month.released:
        parts.append("<div class=total><span class=amount>"
                     f"{e(_usd(a.total_avoidable_month))}</span>"
                     f"<span class=note>{e(total_detail)}</span>"
                     "<span class='tag modeled'>modeled</span></div>")
    parts.append("</div></section>")

    # what this is not
    parts.append("<section class=section><div class=section-head>"
                 "<div><div class=label>04 / Standing</div><h2>What each number means.</h2></div>"
                 "<p>The report separates observations from projections. A modeled number is useful, "
                 "but it is not proof that savings appeared after a change.</p></div>"
                 "<table><thead><tr><th>Class</th><th>Means</th></tr></thead><tbody>"
                 "<tr><td>Measured</td><td>Computed from usage fields actually returned by the API.</td></tr>"
                 "<tr><td>Modeled</td><td>Projected from those measurements under stated assumptions.</td></tr>"
                 "<tr><td>Verified</td><td>Observed in production after a change shipped. "
                 "No figure in this report is verified, because this engagement changed nothing.</td></tr>"
                 "</tbody></table>")
    for n in a.notes:
        parts.append(f"<p class=note>{e(n)}</p>")
    parts.append("</section>")

    # Conditional, because the unconditional version was false on the workflow
    # this project recommends. `run_diagnostic.py` calls `count_tokens.py`
    # unless `--estimate-only` is passed, and counting sends prompt prefixes to
    # a tokenizer -- that is the whole point of it, and it is what takes the
    # segment split from 19.2% median error to 0.2%. A client reading only this
    # report was told nothing left, in a report produced by a run that sent
    # their prompts to a provider.
    #
    # The analysis half stays true either way and is still worth saying: this
    # package imports no network library and a test asserts it. What changes is
    # that the report no longer speaks for a step it did not perform.
    if a.tokens_counted:
        privacy = ("Analysis ran locally and this package opens no sockets. "
                   "Segment sizes here were counted rather than estimated, and "
                   "counting is a separate step that sends prompt prefixes to a "
                   "tokenizer endpoint. Segments are identified by keyed hash, "
                   "not content.")
    else:
        privacy = ("Analysis ran locally. No prompt content left the "
                   "environment; segments are identified by keyed hash, not "
                   "content.")
    parts.append(f"<footer>{privacy} KCG Consulting LLC</footer></main></body></html>")
    return "\n".join(parts)


_WIDTH = 78

# What the four prices are, in words, before any of them is used. Somebody
# running this for the first time has no reason to know that "cache read" is a
# billing class rather than an event, and every number below is a ratio between
# these four. Stated once, at the top, so nothing after it has to explain itself.
_PRIMER = (
    "Your provider charges four different prices for the same token. Sending "
    "text fresh costs 1x. Reading it back out of the cache costs 0.1x. Putting "
    "it into the cache costs 1.25x, or 2x for the version that survives an "
    "hour. Your bill adds all four together and shows you one number called "
    "\"input tokens\".",
    "This pulls that number back apart, from logs you already have, and tells "
    "you which of the four you have been buying.",
)


def _fill(text: str, width: int) -> list:
    # break_on_hyphens off, break_long_words off: the default settings split
    # `--allow-unreconciled` across two lines at the hyphen, so the report's own
    # instructions could not be copied out of it.
    return textwrap.wrap(str(text), width=max(8, width),
                         break_on_hyphens=False, break_long_words=False) or [""]


def _wrap(text: str, width: int, indent: str = "") -> list:
    return [indent + line for line in _fill(text, width - len(indent))]


def _rule(n: int, title: str, width: int = _WIDTH) -> list:
    head = f"── {n} · {title} "
    return ["", head + "─" * max(0, width - len(head)), ""]


def _cols(rows: list, widths: list, indent: str = "  ") -> list:
    """One table, last column wrapped and hung under itself.

    Written by hand rather than pulled in from anywhere: this package has no
    dependencies and a test asserts it, and a table is not worth breaking that
    for.
    """
    out = []
    for row in rows:
        head, tail = row[:-1], row[-1]
        stem = indent + "".join(f"{c:<{w}}" for c, w in zip(head, widths[:-1]))
        lines = _fill(tail, widths[-1])
        out.append(stem + lines[0])
        pad = " " * len(stem)
        out.extend(pad + line for line in lines[1:])
    return out


def _impact(f) -> str:
    """The money column, in words a first-time reader can act on.

    "[figure withheld]" is accurate and tells somebody who has never seen this
    tool anything at all. The two reasons a row carries no number are different
    and have different remedies, so they read differently and the legend under
    the table says which is which.
    """
    fig = f.avoidable_usd_month
    if fig is None:
        return "not costed"
    if fig.released:
        return f"~${fig:,.0f}/mo"
    return "withheld"


def _headline(a: Analysis) -> str:
    """One plain sentence saying how the cache is doing, before any percentage.

    Both renderers used to open on an asserted headline -- the HTML one on the
    literal string "The cache is active, but underused." -- which was written
    once against one demo trace and then printed over every trace after it,
    including runs where the cache was doing fine. An assertion that does not
    read its own data is worse than no headline, because it looks like a
    finding.

    Efficiency rather than `input_from_cache`: the latter is near 100% on any
    heavy cache user and says nothing about whether the caching worked, which is
    the misreading CAC-1 exists to correct.
    """
    eff = a.ratios.get("prefix_efficiency")
    if eff is None:
        return "Not enough cache activity here to say whether caching is working."
    if eff >= 0.75:
        return ("Caching is working: most of what gets written to cache is read "
                "back before it expires.")
    if eff >= 0.5:
        return (f"Caching is working, but {_pct(1 - eff)} of what gets written to "
                f"cache expires or is discarded before anything reads it.")
    return (f"Most of what gets written to cache is never read: {_pct(1 - eff)} of "
            f"written tokens are paid for at a premium and then thrown away.")


def render_text(a: Analysis, *, detail: bool = False) -> str:
    """The same gate as the HTML report, because the text one forwards just as easily.

    These two renderers diverged once: HTML withheld dollars on a failed
    reconciliation while text printed them through Finding.__str__ and then
    totalled them. A number that survives in either renderer has escaped the
    gate, and the text output is the one that ends up pasted into an email.

    Laid out as a sequence someone can read top to bottom: what the four prices
    are, what was read, how the cache is doing, what it is costing, what to do.
    It used to open on four unlabelled values and then print every finding as a
    150-word paragraph, which on a real run is five screens of prose with the
    two actionable lines buried inside it.

    `detail=False` keeps the reasoning behind the table. Nothing is lost -- the
    same strings render with `--detail`, and the HTML report has room and always
    carries them.
    """
    # The gate is whatever the figures themselves say. Recomputing it from the
    # reconciliation dict is how these two renderers disagreed in the first
    # place, and it also reads a missing invoice as a pass.
    gate_ok = a.total_avoidable_month.released or a.spend.get("input_usd") is not None \
        and a.spend["input_usd"].released

    out = ["", "  cacheeconomics · what your prompt caching actually costs",
           "  " + "─" * (_WIDTH - 2)]
    # Before anything else on a released-without-an-invoice run. This used to be
    # the first of five notes at the bottom, which is where a warning about
    # forwarding a document does the least good.
    draft = next((n for n in a.notes if n.startswith("DRAFT")), None)
    if draft:
        out += [""] + _wrap(draft, _WIDTH, "  ")
    for para in _PRIMER:
        out += [""] + _wrap(para, _WIDTH, "  ")

    out += _rule(1, "what I read")
    reqs = a.ratios.get("requests") or a.coverage.get("total") or 0
    span = f" over {a.window_days:.0f} days" if a.window_days else ""
    out += _cols([
        ("volume", f"{reqs:,} requests{span}"),
        ("could be read", f"{a.coverage['analysed']:,} of {a.coverage['total']:,} "
                          f"({_pct(a.coverage['fraction'])})"),
        ("depth", {Tier.USAGE_ONLY: "token counts only. Enough for spend and "
                                    "cadence; the prompts themselves are not here, "
                                    "so nothing can say which *part* of a prompt "
                                    "costs you",
                   }.get(a.tier, f"{a.tier} — prompt structure is available, so "
                                 f"findings can name the part of the prompt at "
                                 f"fault")),
        ("privacy", "read on this machine. Nothing was sent anywhere"),
    ], [16, _WIDTH - 18])

    out += _rule(2, "how your caching is doing")
    out += _wrap(_headline(a), _WIDTH, "  ") + [""]
    out += _cols([("measure", "value", "in plain english"),
                  ("─" * 19, "─" * 6, "─" * 48)],
                 [20, 8, _WIDTH - 30])
    out += _cols([
        ("input from cache", _pct(a.ratios["input_from_cache"]),
         "of everything you sent, the share billed at the cheap 0.1x read rate "
         "instead of full price"),
        ("prefix efficiency", _pct(a.ratios["prefix_efficiency"]),
         "of every token you paid to put into cache, the share something read "
         "back before it expired. This is the one that says whether caching is "
         "working"),
    ], [20, 8, _WIDTH - 30])
    if a.reconciliation:
        r = a.reconciliation
        pct = r.get("delta_pct")
        # `delta_pct is None` stood in for four different rejections, so a
        # negative, non-finite or non-numeric invoice printed "invoice is zero,
        # (OUTSIDE the 5% gate)" -- naming a blocker that is not the real one,
        # and naming the gate as the reason when the invoice never reached it.
        # The analyzer and the HTML renderer both learned `invalid_invoice`; this
        # is the twin that did not, and it is the one that gets pasted into an
        # email.
        invalid = r.get("invalid_invoice")
        if invalid:
            reason = {"zero": "the invoice supplied is zero",
                      "negative": f"the invoice supplied is negative "
                                  f"({r.get('invoice_usd')})",
                      "not-finite": "the invoice supplied is not a finite number",
                      }.get(invalid, "the invoice supplied is not a number")
            out += _cols([("reconciliation", "—",
                           f"not attempted: {reason}")], [20, 8, _WIDTH - 30])
        else:
            out += _cols([("reconciliation",
                           f"{abs(pct):.1f}%" if pct is not None else "—",
                           ("the gap between what this priced and the invoice you "
                            "gave it. " if pct is not None
                            else "no difference could be computed. ")
                           + ("Inside the 5% gate, so the dollar figures below are "
                              "published" if gate_ok else
                              "Outside the 5% gate, so no dollar figure is "
                              "published"))],
                         [20, 8, _WIDTH - 30])
        if r.get("unpriced_requests"):
            # Broken out by cause. One label for three different exclusions told
            # an operator to go hunting for cache lifetimes when the row was
            # actually on a model with no recorded price, or carried no usable
            # timestamp -- three different fixes behind one sentence.
            b = r.get("blockers") or {}
            parts = [f"{n} {label}" for n, label in (
                (b.get("unprovable_lifetime", 0), "with cache writes of unprovable lifetime"),
                (b.get("unpriceable_model", 0), "on a model or surface with no recorded price"),
                (b.get("undated", 0), "with no usable timestamp"),
                (b.get("skipped_rows", 0), "the loader could not read"),
                (b.get("no_usage", 0), "carrying no usage fields"),
                (b.get("failed_but_billed", 0), "that failed but still billed")) if n]
            out += _cols([("unpriced", f"{r['unpriced_requests']:,}",
                           ("requests this could not price: "
                            + "; ".join(parts)) if parts
                           else "requests this could not price")],
                         [20, 8, _WIDTH - 30])

    n = len(a.findings)
    out += _rule(3, f"what it is costing you — {n} finding{'' if n == 1 else 's'}, "
                    f"worst first" if n else "what it is costing you")
    if not gate_ok:
        # Say which reason. "Insufficient reconciliation" was printed even when
        # the real cause was that no invoice existed at all, which sends the
        # reader to fix the wrong thing.
        why = (a.spend["input_usd"].withheld_because
               if a.spend.get("input_usd") is not None else "not reconciled")
        # The literal marker stays. It is what a reader greps for before
        # forwarding a report, and three tests treat it as the signal that no
        # number escaped, so burying it inside a friendlier sentence would have
        # removed the thing that makes the sentence trustworthy.
        out += _wrap(f"FIGURES WITHHELD — {why}.", _WIDTH, "  ")
        out.append("")
        out += _wrap("The findings themselves still stand. This is a rule about "
                     "publishing money, not a doubt about the analysis. Step 1 "
                     "at the end turns the numbers on.", _WIDTH, "  ")
        out.append("")
    if not a.findings:
        out += _wrap("Nothing to act on. No rule here fired on this data.",
                     _WIDTH, "  ")
    # Cost before title, deliberately. The first question anybody brings to this
    # is "where is the money going", so the money column sits where the eye
    # lands rather than at the far right of a wrapped title.
    _c = [4, 10, 15, _WIDTH - 33]
    out += _cols([("#", "severity", "what it costs", "what is happening"),
                  ("─" * 3, "─" * 9, "─" * 14, "─" * (_c[3] - 1))], _c)
    for i, f in enumerate(a.findings, 1):
        out += _cols([(str(i), f.severity.upper(), _impact(f), f.title)], _c)
        if f.fix:
            out += _cols([("", "do this", f.fix)], [4, 10, _WIDTH - 16])
        if detail:
            out += _cols([("", "why", f.detail)], [4, 10, _WIDTH - 16])
        out += _cols([("", "basis", f"{f.code} · {f.evidence_class} · confidence "
                                    f"{f.confidence} · quality risk "
                                    f"{f.quality_risk}")], [4, 10, _WIDTH - 16])
        out.append("")
    if a.total_avoidable_month.released:
        out += _cols([("", "TOTAL", f"~${a.total_avoidable_month:,.0f}/month, "
                                    f"modeled and deliberately pessimistic")],
                     [4, 10, _WIDTH - 16])
        out.append("")
    # The two blanks in the cost column mean different things and have different
    # remedies. Printed only where they appear, so this is a legend rather than
    # boilerplate.
    legend = []
    if any(f.avoidable_usd_month is None for f in a.findings):
        legend.append("'not costed' — this rule names a mechanism and does not "
                      "produce a dollar amount at this depth. Section 1 says "
                      "what depth you gave it.")
    if any(f.avoidable_usd_month is not None and not f.avoidable_usd_month.released
           for f in a.findings):
        legend.append("'withheld' — a figure exists and has not passed the "
                      "reconciliation gate. Step 1 below releases it.")
    for line in legend:
        out += _wrap(line, _WIDTH, "  ")
    # Caveats on the numbers directly above them. These used to sit in the
    # notes block at the very bottom, so a figure and the sentence saying what
    # it excludes were four sections apart, and folding that block away would
    # have separated them entirely.
    for note in spend_caveats(a):
        out.append("")
        out += _wrap(f"caveat: {note}", _WIDTH, "  ")
    if not detail and a.findings:
        out.append("")
        out += _wrap("Every row above is the short version. Re-run with "
                     "--detail for the reasoning behind each one, the counts, "
                     "and what was excluded from them.", _WIDTH, "  ")

    steps = _next_steps(a)
    if steps:
        out += _rule(4, "what to do next")
        for i, step in enumerate(steps, 1):
            out += _cols([(f"{i}.", step)], [4, _WIDTH - 6])
            out.append("")

    # Notes carry the coverage facts: unpriced writes, partial structure, which
    # records were excluded and why. They belong in the report and they do not
    # belong in the reader's way, so they sit with the reasoning under --detail.
    #
    # The draft stamp is the exception and does not travel with them. It says
    # "not for external use" about the document it is printed on, which is worth
    # nothing at the bottom of a long report and is the first thing somebody
    # about to forward one needs to see. It goes to the top, and a test asserts
    # it is there in the default view -- nothing checked that before, which is
    # how it nearly left with this section.
    if a.notes and detail:
        out += _rule(5, "the fine print — what this is based on, and what it "
                        "could not see")
        for n in a.notes:
            out += _cols([("·", n)], [4, _WIDTH - 6])
    elif a.notes:
        out.append("")
        rest = len(a.notes) - len(spend_caveats(a))
        if rest:
            out += _wrap(f"{rest} note(s) on provenance and coverage are printed "
                         f"with --detail.", _WIDTH, "  ")
    out.append("")
    return "\n".join(out)


def _next_steps(a: Analysis) -> list:
    """What to run now, given what this particular report could not answer.

    The report used to end on a wall of caveats. Every one of them was true and
    none of them said what to do about it, so a first run read as a list of
    refusals with no way forward -- the tool telling you it would not answer,
    five times, and then stopping.

    Ordered by what unblocks the most. A missing invoice blocks every dollar
    figure, so it goes first; the tier only decides which *kinds* of finding are
    reachable at all.
    """
    steps = []
    withheld = not (a.spend.get("input_usd") and a.spend["input_usd"].released)

    if withheld and not a.reconciliation:
        steps.append("Get a dollar figure: re-run with --invoice-usd <amount> "
                     "from the provider bill covering this window. Every number "
                     "stays hidden until it reconciles to within 5% of money "
                     "that actually left an account.")
        steps.append("Or, for an internal look before the bill arrives: add "
                     "--allow-unreconciled. It releases the figures and stamps "
                     "the report DRAFT, which is not something to forward.")
    elif withheld and a.reconciliation:
        steps.append("Figures are withheld and the reason is printed above. "
                     "Fix that rather than working around it -- each of those "
                     "gates exists because it once let a wrong number out.")

    if a.tier is Tier.USAGE_ONLY:
        steps.append("Reach the structural findings: this input carries usage "
                     "counters but not prompt structure, so nothing here can "
                     "say *which part* of the prompt costs you. Export request "
                     "bodies from your gateway and re-run with --from bodies, "
                     "or point the agent at tier-b/capture_proxy.py if you "
                     "cannot export.")
    elif a.tier is Tier.INFERRED and not a.tokens_counted:
        steps.append("Put dollar figures on the structural findings: run "
                     "tier-b/run_diagnostic.py instead, which counts tokens "
                     "first. Segment sizes are estimated here and the estimate "
                     "is 19.2% off at the median.")

    acted = [f for f in a.findings if f.fix and f.severity in ("high", "medium")]
    if acted:
        top = acted[0]
        steps.append(f"Act on {top.code} first ({top.title.lower()}). It is the "
                     f"highest-severity finding here, and the 'do this' line in "
                     f"its row is the change to make. Re-run with --detail for "
                     f"the reasoning behind it.")
    elif a.findings:
        steps.append("Nothing here is high severity. The findings above are "
                     "measurements rather than problems.")
    return steps
