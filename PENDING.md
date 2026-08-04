# Deferred work

Known-and-not-done, each with whatever measurement exists. The point of this
file is that "we decided not to do this yet" and "we forgot" look identical in
a codebase unless one of them is written down.

Two places in the source already pointed here — `analyzer.py` and
`tests/test_invariants.py` — before this file existed, which is the same defect
the invariant harness was written to catch: a claim in prose that nothing could
contradict. Both references are now true.

**Rule for this file:** an entry without a measurement is a rumour. If you defer
something, attach what you actually observed, not what you suspect.

---

## 1. ~~`Figure.release()` upgrades DRAFT to RECONCILED on re-release~~ — FIXED

Kept as a record of why it moved from deferred to fixed within one stage.

Deferred first on the grounds that it was unreachable — all three call sites
take a withheld figure straight from `_monthly` — and that the `RECONCILED`
default was deliberate and documented.

Fixed anyway, because the invariant written to cover Figure round-trips had a
`release(True)` table entry that passed `as_=f.released_as` and therefore never
called the path its own label named. Given the choice between a test that
reports coverage it does not have and fixing a latent sharp edge, the fix is
cheaper and honest. `release()` now keeps existing provenance when the figure
is already released, and still defaults to `RECONCILED` for a withheld one —
which was the whole point of the documented default.

---

## 2. Projection gating for per-finding figures

`_monthly` has seven callers; six are per-finding `avoidable_usd_month` and
only `monthly_input_usd` was gated by the projection floor.

**Measured:** exactly **11** tests fail when the gate is extended to finding
figures, measured with a read-only pytest plugin. All 11 assert *other* gates —
structural coverage, alignment, counted tokens, size agreement — and merely
need some released figure to exist. None is about projections. An earlier note
in `analyzer.py` said "ten"; 11 is the measured number.

**Status: IN FLIGHT, not deferred.** Being closed in Track A of the current
round via a split figure (`avoidable_usd_window`, measured and never
extrapolated, beside the gated `avoidable_usd_month`). Recorded here because
`analyzer.py:2084` points at this file for it.

---

## 3. EFF-1 contradicts TTL-1 on rolling conversations

The two rules can both fire on the same trace and recommend opposing actions.

**Measured:** not yet quantified. Surfaced in adversarial review; no
reproduction fixture has been built, so the size of the disagreement — and
whether it is a real contradiction or two correct statements about different
subsets — is unknown.

**Do not act on this entry until it has a reproduction.** Recorded so it is not
lost, not because it is understood.

---

## 4. TTL-1 / TTL-2 premium asymmetry

The two lifetime rules do not treat the write premium symmetrically.

**Measured:** not yet quantified. Same status as item 3 — surfaced in review,
no fixture, no number.

---

## 5. Surface defaults: 8 members, not the 2 the review found

**This entry was stale within one commit of being written** and is corrected
here. It previously said INV-4 walks only `plugin` and does not catch
`Recorder`. INV-4 was widened in the same commit, and the claim was false by
the time it was committed — the same "prose nothing can contradict" failure
this file exists to record, occurring in the file that records it. Caught by
external review, not by me.

**Measured, current:** INV-4 walks the package recursively via
`pkgutil.walk_packages` and inspects 164 callables. Eight default `target_id`
to the named surface `'anthropic/direct'`:

    adapters.claude_code.load_sessions
    checks.check_breakpoint_budget
    checks.check_minimum
    checks.run_all
    cost.ttl_crossover
    plugin.CachePlugin.on_request
    recorder.Recorder.__init__
    trace.Request.__init__

Adversarial review named two of these by hand (`on_request`, `Request`). The
invariant found the other six, the last only after switching from
`iter_modules` to `walk_packages` — `iter_modules` does not descend into
subpackages, so `adapters.claude_code` was invisible.

**Not cosmetic.** Minimums differ by surface: 512 on `anthropic/direct`, 1024
on `openai/direct`. Measured: `check_minimum(768, 'claude-opus-5')` returns
**PASS** by default and **FAIL** when the real surface is named.

**Status: IN FLIGHT** — all 8 assigned to Track B of the current round.

---

## 6. MIN-1 and RT-MIN have no estimation allowance

`check_minimum` compares an estimated prefix against a hard minimum. The
static linter uses a symmetric ±10% band; the plugin uses the measured 2.81×
worst overestimate.

**Measured:** `check_minimum(600, 'claude-opus-5')` returns **PASS** —
"600 tokens clears the 512 minimum" — while 600 estimated tokens could really
be 214 at the measured worst overestimate, well below 512. The ±10% band spans
540–660 and never straddles 512, so it does not even abstain.

**Status: IN FLIGHT for `check_minimum`** (Track C, finding L3). MIN-1 and
RT-MIN inherit the same thinness and are **not** covered by that fix.

**Caveat carried forward:** `ESTIMATOR_WORST_OVERESTIMATE = 2.81` rests on 26
prefixes over six bodies. It is a floor on the error, not a proof of its bound.
