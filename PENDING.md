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

## 1. `Figure.release()` upgrades DRAFT to RECONCILED on re-release

`release(ok, as_="")` defaults `as_` to `RECONCILED`, so calling it on a figure
that is *already* released as `DRAFT` silently relabels it as invoice-checked.

**Measured:** unreachable today. All three call sites
(`analyzer.py:123`, `analyzer.py:2115`, `money.release_map`) operate on a
withheld figure straight from `_monthly`, whose output is `released=False`.
Verified by inspection of every `.release(` site in `harness/cacheeconomics/`.

**Why deferred rather than fixed:** the `RECONCILED` default is deliberate and
documented — it exists so figures that predate the distinction are not
relabelled as drafts. Changing it to "preserve existing provenance if already
released" is strictly safer, but nothing exercises the path, and a speculative
fix to a documented choice is how a design acquires two rules.

**When to pick it up:** the moment any caller releases an already-released
figure. `abs()` had the identical shape and *was* reachable; it was fixed in
`d394523`.

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

## 5. `Recorder.target_id` has a surface default

The same class as `on_request` and `trace.Request`, both closed in the current
round. `Recorder` was named in review and not closed with them.

**Measured:** confirmed present by signature inspection. INV-4 does not
currently catch it — the invariant walks `plugin`'s public callables, and
`Recorder` is not among them, which is a known gap in that invariant's
discovery mechanism rather than evidence of absence.

**Extending INV-4 to cover the recorder is the fix, not a one-line default
change** — otherwise the next entry point repeats it.

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
