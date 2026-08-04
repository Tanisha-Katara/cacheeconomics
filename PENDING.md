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

## 2. ~~Projection gating for per-finding figures~~ — CLOSED

**My measurement of this was wrong and is corrected here.** I stated, in the
approved plan and in this file, that "exactly 11" tests break when the gate is
extended, measured with a read-only pytest probe. Track A found a naive gate
breaks **17**, and the correct implementation breaks **9**. My probe never
modelled the interaction that explains the difference: a naive gate OVERWRITES
an existing refusal, replacing "segment sizes disagree with what was billed"
with the weaker projection reason. `_withhold_projection` only downgrades a
figure that is currently *released*, which cut the migration to 9 — all
positive controls for other gates, none about projections.

Closed in Track A with a split figure: `avoidable_usd_window`, measured over the
observed window and never extrapolated, beside the gated `avoidable_usd_month`.

**A second defect was found inside the fix, and it was also mine.** The floor was
computed once from the WHOLE trace's window and request count, then applied to
every finding — so a finding resting on a tiny subset published a monthly figure
whenever unrelated traffic cleared the floor. Measured: a 10-request, 2.3-day
trace whose only two cache writers went out **1 second apart** published EFF-1 at
$6.43/mo and FAN-1 at $14.79/mo, both `reconciled`, totalling $21.21/mo — on a
2-request sample, while the 8 filler requests that carried the floor contributed
exactly $0. Each rule now passes the timestamps of the requests that actually
moved its figure, and `sample_times` is required with no default.

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
`pkgutil.walk_packages`, unwraps classmethod/staticmethod/functools
descriptors, and inspects 165 callables. Eight default `target_id`
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

**Status: IN FLIGHT for `check_minimum`** (Track B, finding L3 — the track
split moved `checks.py` to B when the surface-default class widened). MIN-1 and
RT-MIN inherit the same thinness and are **not** covered by that fix.

**Caveat carried forward:** `ESTIMATOR_WORST_OVERESTIMATE = 2.81` rests on 26
prefixes over six bodies. It is a floor on the error, not a proof of its bound.

---

## 7. `sweep_report.py` derives `-counted.jsonl` with a naive `str.replace`

`counted()` at `tier-b/sweep_report.py:37` is `path.replace(".jsonl",
"-counted.jsonl")`, which rewrites EVERY occurrence. `run_diagnostic._counted_path`
had the identical bug and was fixed; this copy was not.

**Measured:** input `run.jsonl.bak/trace.jsonl` becomes
`run-counted.jsonl.bak/trace-counted.jsonl` — the directory component is
corrupted, so the tool reads from or writes to a path that is not the one
intended.

**Why not fixed now:** outside Track D's assigned files, and this is a
twin-path defect whose whole lesson is that fixing one copy silently is what
created it. The two derivations should become ONE function both scripts call,
which is a change to two files and needs its own reproduction.

**Contained, not urgent:** the new `.gitignore` rule `*-counted.*` covers the
output wherever it lands, so this is a correctness bug, not an egress one.

---

## 8. Five shipped registry surfaces record no breakpoint budget or lookback

**Measured** against `harness/cacheeconomics/data/providers.json`, 8 surfaces:

    amazon-bedrock/converse      lookback_blocks = null
    openai/direct                max_breakpoints = null, lookback_blocks = null
    openai/bedrock               both NOT RECORDED
    deepseek/direct              lookback_blocks NOT RECORDED
    google/gemini-explicit       both NOT RECORDED

Until Track C's fix these gaps disabled live checks **in silence** — the
operator saw a quiet dashboard and read it as healthy. After it, they emit a
low-severity `RT-NOSURFACE` naming the missing key. That is the correct
behaviour and it is still not the same as knowing the values.

**Deliberately NOT fixed by inventing values.** This repo's rule is that a
registry row carries a dated source and a contested row is never published as
fact. Filling five rows with plausible numbers to silence an alert is precisely
the failure this project exists to avoid, and it would be undetectable
afterwards.

**What picking this up looks like:** find each value in the provider's own
documentation, record it with the source and the date observed, and only then
does the alert go quiet — because the answer exists, not because the question
was suppressed.

---

## 9. `TraceSet`'s structural fields default to "measured and fine"

A `TraceSet` constructed directly — which the public API allows — answers every
structural question affirmatively without anything having measured it:

    structural_coverage: float = 1.0
    tokens_counted: float = 1.0
    token_sums_reconciled: bool = True
    token_sums_publishable: bool = True

So `TraceSet(requests=reqs, tier=Tier.INSTRUMENTED)` declares counted tokens and
publishable segment sums that no loader established. Both the analyzer and the
bake-off gate read these, so a gate that fails closed on `trace=None` fails
**open** on a bare `TraceSet`.

**Measured** (Track E, applied temporarily then reverted): flipping the four
defaults to `0.0/0.0/False/False` fails **26 tests, 11 of them analyzer tests
that never call the simulator** — `TestStructuralClaimsNeedMeasuredSegmentation`,
`TestStructuralCoverageGatesStructuralMoney`,
`TestCoverageThatGatesMoneyIsWeightedByMoney`,
`TestStructuralMoneyNeedsCountedTokens`.

That count is the finding, not an obstacle to it: **the analyzer shares this
hole rather than inheriting it from the simulator.** Eleven analyzer tests
currently rely on a permissive default to reach the code they exercise.

**Partially backstopped, which is why it has not bitten yet.** A false
declaration is caught from the requests for three of them —
`token_sums_publishable` by `misscaled`, `structural_coverage` by `unstructured`
reaching `omitted`, and `structural_coverage_billed` is a property that cannot
be declared at all. **Not** backstopped: `tokens_counted`, `tier`, `alignment`,
`skipped_rows`. `tokens_counted=1.0` is the real hole — nothing on a `Request`
carries the information needed to contradict it.

**Exact change when picked up:** `harness/cacheeconomics/trace.py`, `class
TraceSet`, the four field defaults above. Expect to reclassify 11 analyzer
tests, which is the actual work — the one-line default change is not.

**Do not do this inside another track.** It moves analyzer behaviour, and the
tests it breaks are the ones that assert the analyzer's own trust gates.

---

## 10. `min_cacheable_tokens` cannot be inspected on a contested row

`registry.capability(target_id, name, allow_contested=False)` takes an
`allow_contested` flag; `registry.min_cacheable_tokens(target_id, model)` does
not. So on a contested surface — shipped: `openai/bedrock`,
`google/gemini-explicit` — a caller cannot distinguish "this row records a
minimum, but the row is disputed" from "this row records no minimum at all".

**Consequence, found by Track C:** `RT-NOSURFACE` can name the dispute but not
whether the value also needs recording. It now reports the key as *not
inspected* rather than guessing, which is correct — "present" understates a row
that needs values, "absent" invents a gap — but it is an answer the operator
cannot act on precisely.

**Why not fixed there:** reproducing the registry's `inherits_minimums_from`
walk inside `monitor.py` would be a second copy of the registry's own knowledge,
and every second copy in this codebase has drifted. The fix belongs in
`registry.py`: either give `min_cacheable_tokens` the same `allow_contested`
parameter `capability` has, or add a diagnostic that reports row-level contested
status separately from per-key presence.

**Scope:** `registry.py` only, plus whatever calls it. Small, but it is the
public registry API and no track owned it this round.

---

## 11. FIVE BRANCHES COMPLETE AND UNVERIFIED EXTERNALLY

**Status: not merged. Do not merge without reading this entry.**

Five worktree branches carry substantial fixes. Every one is committed, tested
and self-reported as complete. **None has passed a clean external review**, and
the external reviewer became unavailable mid-run:

    Codex usage limit reached; resets 2026-08-11 19:42

Confirmed by direct probe, not inferred from a single failure.

### Why this entry exists rather than a merge

The plan this round was executed under states: *if a stage's review cannot run,
that stage is not done — it is marked unverified externally here and does not
merge. It is never quietly counted as passing.* This is that record.

Two independent reasons merging is not justified:

1. **No track ever returned a clean review.** 48 scoped reviews across 5 tracks;
   every one found at least one real defect. The final verdict on every track
   was `needs-attention`.
2. **The most recent work on every track is unreviewed.** Tracks C and E have
   strong final rounds that no reviewer has seen. Merging them rests on my
   assessment alone, which is the exact failure this round was convened to fix.

### Branch state at the point of the outage

    worktree-agent-a595754573f25fde6   Track A   4a12b92   projections, TTL-1, multiplier validation
    worktree-agent-abe30441cb92d4802   Track B   b1039fc   surface defaults, assumed_inputs, cost.py bool
    worktree-agent-a316a0a62e83df9b7   Track C   d632025   monitor abstentions, TTL narrowing, budget guard
    worktree-agent-a4bfc871983f7a67b   Track D   1f15ecd   tier-b counting, provenance, egress rule
    worktree-agent-a1b3f34e94aac1182   Track E   9f0e8d1   bake-off trust gates, by-agent, DRAFT rendering

`main` is untouched at the INV-6b commit: 1327 passing, CI green.

### Open findings at the outage, by track

- **A** — reviewed through round 5; round 6 unreviewed.
- **B** — reviewed through round 8; round 9 (`cost.py` bool fix) unreviewed.
- **C** — reviewed through round 7; round 8 unreviewed.
- **D** — those two HIGH findings are now CLOSED at `4a55fb7`, after this entry
  was first written. `locally_vouched_serveable()` requires the exact id in the
  price table AND the first-party endpoint, with everything else falling to an
  explicit `--assume-endpoint-serves`; and `countable()` moved into the package
  so the writer's refusal and the loader's acceptance are ONE predicate rather
  than two that agreed by luck.

  Its own revert-proof then found that **two of its four fixes had no test at
  all** — reverting each left the file green — and it closed that gap in a
  separate commit so both are legible. Unreviewed externally like everything
  after Track B's round 9.
- **E** — reviewed through round 6; round 7 unreviewed.

### Merge order and actions, when verification resumes

Order: **B → A → C → E → D** (B first: 22 files, the widest surface).

Actions that must accompany the merge:

    retire KNOWN_JSON_UNPROVENANCED and KNOWN_SILENT_ABSTENTIONS in test_invariants.py
    delete the xfail markers each track flagged (each carries a comment naming the trigger)
    rebuild web/harness-bundle.js ONCE, after all five  (Track E changed BakeOff.__str__,
      so rendered output differs, not only internals)
    full suite, three disclosure verifiers, CLI smoke, generated fixtures unchanged
    a FINAL external review of the merged result — five individually-correct
      diffs can still be jointly wrong, and no per-track review looks at the seams

**The last item is not optional.** It is the one review that covers what the
per-track reviews structurally cannot.

---

## 12. `registry._load` accepts NaN and Infinity from disk

Python's JSON parser accepts the non-standard `NaN`, `Infinity` and `-Infinity`
literals by default. **Verified through `registry._load` itself**, not merely
`json.loads`: a registry file containing them returns
`{'read': nan, 'write_5m': inf, 'write_1h': -1}`.

**Downstream, measured before the consumers were guarded:**

    NaN       ->  usd = nan   rendered "$nan"; --format json emits a bare NaN,
                              which is INVALID JSON and breaks any consumer
    Infinity  ->  usd = inf
    -1.0      ->  usd = -5.00  a negative bill
    0.0       ->  usd = 0.00   writes priced free

**Every consumer is now guarded** — `cost.is_multiplier` requires finite and
`> 0`, and `tiers._surface` refuses a present-but-invalid multiplier — so this
is defence in depth rather than a live hole. It is still the stronger fix,
because it stops the value entering the process rather than catching it at each
use, and "each use" is a set that has already been wrong twice this round.

**The change:** `json.load(f, parse_constant=...)` raising on the three
constants. Verified that `parse_constant` closes it.

**Why unowned:** `registry.py` was assigned to Track C for one additive change
(`allow_contested`), and this is a different concern in the same file. Track B
found it, guarded every consumer it owns, and correctly declined to edit a file
outside its remit.
