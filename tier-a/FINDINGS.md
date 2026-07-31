# Tier A — static findings, three open-source agentic deployments

**Date:** 2026-07-28 · **Method:** source only, no API spend, no execution
**Evidence class:** every claim below is a *fact about source code*, verifiable by reading the cited file and line. Nothing here is Measured or Modeled. Dollar figures require Tier B.

---

## Correction logged before anything else

The initial candidate screen searched `OpenHands/OpenHands` (82,407★) and found **0 `cache_control` hits**, which looked like a headline: "the most popular open-source agent does no prompt caching."

**That was wrong.** `OpenHands/OpenHands` is the TypeScript frontend. Agent logic lives in `OpenHands/software-agent-sdk` (938★, Python), which caches deliberately — including a dedicated `test_prompt_caching_cross_conversation.py`.

Recording this because it is exactly the failure the method exists to prevent: a repo-level absence of a string is not evidence of an absent behaviour. Had this shipped, the case study would have opened with a false claim about a major project.

---

## The convergent finding

**All three projects emit `{"type": "ephemeral"}` with no `ttl` key. That is the 5-minute cache. Not one of them ever writes a 1-hour cache.**

| Project | Where | 1h TTL anywhere? |
|---|---|---|
| OpenHands SDK | `llm/message.py:200,214,384` | **No** — grep for `ttl` / `1h` / `3600` returns nothing |
| SWE-agent | `agent/history_processors.py:59,63,67` | **No** |
| browser-use | `llm/anthropic/serializer.py:57` | **No** — and see below |

**browser-use is the sharpest case.** `llm/anthropic/chat.py:195-196` reads *both* `ephemeral_5m_input_tokens` and `ephemeral_1h_input_tokens` off the response and surfaces `prompt_cache_creation_1h_tokens` in its usage model (`chat.py:216`). It instruments a metric it can never produce, because `serializer.py:57` only ever emits `CacheControlEphemeralParam(type='ephemeral')`.

**Why this could matter, by the write-premium arithmetic:** a 5-minute write costs 1.25× base input; a 1-hour write costs 2×. A static prefix rewritten twice within an hour has cost 2.5× — already more than a single 1-hour write. So a deployment whose static prefix goes cold ≥2× per hour would be losing money on the 5-minute default.

**Whether any of these three crosses that threshold is unknown and unknowable from source.** The cadence depends on request patterns, not code. Everything above the line is a source fact; the cost consequence is a hypothesis. Tier B measures it.

### Evidence-class discipline

Rows marked **High — fact about source** must be verifiable by reading the cited file and line, with no assumption about how the software is deployed or used. Rows depending on request cadence, step duration, concurrency, or user counts are **Hypothesis** regardless of how plausible they seem, and carry no savings claim until measured.

An earlier draft of this file violated that rule — OH-1 and SA-1 asserted deployment frequency and step duration as source facts. They are split below. This matters beyond tidiness: a maintainer receiving a patch justified by an unmeasured materiality claim is being asked to take a change on the author's assumptions, which is exactly the failure the plan's measured/modeled/verified taxonomy exists to prevent.

---

## Per-project findings

### OpenHands SDK — 938★ · MIT · `openhands-sdk/openhands/sdk/`

Implementation is `llm/llm.py:2603` `_apply_prompt_caching()`. Two breakpoints of the four available:

1. **System block 0** marked; if a second (dynamic) block exists it is *explicitly unmarked* (`llm.py:2617-2618`) "to enable cross-conversation cache sharing." This is the relocation instinct, arrived at independently and correctly.
2. **Last user/tool message** marked, so the prefix extends each turn (`llm.py:2626-2631`) — the rolling marker.

| # | Finding | Confidence | Fix |
|---|---|---|---|
| OH-1 | `_apply_prompt_caching()` marks the static system block with `{"type": "ephemeral"}` and no `ttl` key, i.e. a 5-minute lifetime — while the surrounding comment states the block is unmarked-dynamic-separated specifically "to enable cross-conversation cache sharing" | **High — fact about source** | Candidate: `"ttl": "1h"` |
| OH-1h | *Hypothesis, not a fact:* on a multi-user deployment that prefix is re-written more than twice an hour, past the point where the 1h write premium (2×) beats repeated 5m writes (1.25× each). **Materiality is unestablished** — it depends on request cadence, which cannot be read from source | **Hypothesis — needs Tier B** | — |
| OH-2 | Two of four breakpoints used. No intermediate breakpoint in long tool loops, so the rolling marker is exposed to the **20-block lookback window** — a turn appending >20 content blocks silently loses the prior cache | Needs Tier B to confirm block counts | Intermediate breakpoint every ~15 blocks |
| OH-3 | Capability gating is correct, and `model_features.py:141` explicitly excludes Gemini because "explicit `cache_control` markers freeze its cache" | **Good practice — no finding** | — |

OH-3 is worth stating publicly: they independently discovered that Gemini's named-resource cache model breaks annotate-a-span abstractions. Independent confirmation of the control-model taxonomy.

### SWE-agent — 19,943★ · MIT

`agent/history_processors.py` implements explicit `_set_cache_control` / `_clear_cache_control` (lines 46–67), applied as clear-then-set (lines 293–299) — a deliberate rolling breakpoint, not accidental.

| # | Finding | Confidence | Fix |
|---|---|---|---|
| SA-1 | `_set_cache_control` emits `{"type": "ephemeral"}` with no `ttl` key at all three call sites (lines 59, 63, 67). No 1-hour path exists in the code | **High — fact about source** | Candidate: `"ttl": "1h"` on the static prefix |
| SA-1h | *Hypothesis, not a fact:* SWE-bench steps sometimes exceed five minutes, so the prefix goes cold mid-trajectory. **Step-duration distribution is not observable from source.** Attempted to settle this for free — `TrajectoryStep` does carry `execution_time: float` (`sweagent/types.py`), but the public SWE-bench submissions in [`SWE-bench/experiments`](https://github.com/SWE-bench/experiments) contain only `README.md`, `metadata.yaml` and `results` — **no `.traj` files**. No public timing data found; remains unmeasured | **Hypothesis — needs Tier B** | — |
| SA-2 | `models.py:870` counts `cache_control` occurrences and logs `n_cache_control` — defensive instrumentation consistent with having previously hit Anthropic's 4-breakpoint ceiling (cf. Franklin #73, where 5 emitted breakpoints returned HTTP 400) | Medium — inferred from defensive code | Treat the budget as a constrained allocation |
| SA-3 | `models.py:684-691` strips `cache_control` into `messages_no_cache_control`, but only to feed a client-side token counter — a documented workaround for a LiteLLM bug, not a request mutation. **Not a defect** | Resolved in Tier A | — |
| SA-4 | **SWE-agent never reads `response.usage`.** Grep for `response.usage` / `cache_read_input` / `cache_creation_input` across `models.py` returns **zero hits**. `tokens_sent` comes from `litellm.utils.token_counter` on a cache_control-stripped copy (`models.py:690-694`, consumed at `:780`). A cache read, a cache write, and uncached input are therefore **indistinguishable in the project's own telemetry** | **High — fact about source** | Read `response.usage` and split the three token classes |

**Scope limit on SA-4, stated deliberately:** cost is computed via `litellm.cost_calculator.completion_cost(response, ...)` (`models.py:744`), and LiteLLM's calculator *does* read usage. So the *dollar* figure is probably cache-aware. The defect is in the token statistics, not necessarily the cost — do not overstate it.

### browser-use — 107,111★ · MIT

`llm/anthropic/serializer.py` gates caching through `_serialize_cache_control(use_cache)`, marking the last content block (`serializer.py:154`, `use_cache and is_last`).

| # | Finding | Confidence | Fix |
|---|---|---|---|
| BU-1 | Reads and reports 1-hour cache-creation tokens it can never generate (`chat.py:195-196,216` vs `serializer.py:57`) | **High — fact about source** | See BU-1a — the fix is wider than one helper |
| BU-1a | **Three** sites construct `CacheControlEphemeralParam`, and only one is the helper: `serializer.py:57` (in `_serialize_cache_control`), `serializer.py:118` (direct, string-content path), `chat.py:362` (direct). Patching only the helper leaves two paths emitting 5-minute and looks like a complete fix | **High — fact about source** | Consolidate all three behind one TTL-aware helper, or patch each |
| BU-2 | **CORRECTED.** An earlier draft claimed "images are not cacheable." That is **false** — Anthropic's docs list "Images & Documents: Content blocks in the `messages.content` array, in user turns" as cacheable, and OpenHands already marks `images[-1]` with `cache_control`. The real concern is **volatility, not cacheability**: a browser agent's screenshot changes every step, so it is `ALWAYS`-tier content, and any volatile block positioned before a breakpoint invalidates the stable content behind it | Needs Tier B | Keep per-step screenshots after the last breakpoint |

---

## Methodology note: cross-run cache contamination

Recorded because it was caught by the harness rather than by inspection, and because it would have quietly corrupted a published number.

The replication run (`20260728T192418Z`) started 56 minutes after its predecessor. The prefix builder was deterministic, so both runs sent byte-identical content. The 1-hour arm therefore opened on a **HIT** against the *previous run's* cache, while the 5-minute arm — long expired — opened on a WRITE. One arm started warm, the other cold, and every downstream comparison was meaningless.

The expectation gate caught it (`t0-write/1h: got HIT, expected WRITE`) and refused to emit a cost verdict. Without that gate the run would have produced a tidy, entirely invalid dollar figure showing the 1h arm even further ahead.

Two things follow. The prefix is now seeded with the run id, so it stays byte-identical *within* a run and differs *between* runs. And the anomaly is itself a measurement: **the 1-hour cache demonstrably survived a 56-minute gap**, which is the boundary this project had otherwise taken from documentation rather than observation.

**That observation is now backed by a tracked artifact** — `tier-b/evidence/20260728T192418Z.json`, exported under `--allow-anomalies`. It records `valid: false`, `usable_as: "observation only — NOT a valid cost comparison"`, and has its arm totals suppressed entirely rather than published behind a caveat nobody reads. The per-call rows are retained in full, so the 56-minute figure is derivable by anyone: the run's `started` timestamp minus its predecessor's, against a `t0-write/1h` row showing HIT and 16,442 read tokens.

Exporting a knowingly invalid run needed a deliberate carve-out. Errors, missing cells and timing violations remain unexportable, because those mean the run did not happen as described. An expectation anomaly is different: the run happened exactly as recorded, it simply did not behave as assumed, and that is worth keeping so long as the artifact says so loudly and refuses to hand over a dollar figure.

## Tier B go / no-go

**GO on all three.** Justification: the convergent 5m-only finding is real and one-line-fixable, but whether it *saves money* depends entirely on inter-request gap distribution — which cannot be read from source. That is the measurement Tier B exists to make, and it is the difference between "we noticed something" and a case study.

Arm-B risk confirmed as anticipated: OpenHands routes via LiteLLM (154 hits) so arm B is likely a config flag; **browser-use has its own Anthropic client and does not route via LiteLLM**, so arm B needs a shim or gets dropped and disclosed.

**Prediction registered before measurement, so it can be wrong:** the 1h change wins clearly on OpenHands (multi-user, cross-conversation prefix), is marginal on SWE-agent (depends on step latency), and is smallest on browser-use — not because images cannot be cached (they can), but because a large per-step screenshot means a greater share of each request is genuinely volatile regardless of TTL. If the results contradict this, the prediction is published alongside them.

**Correction log.** Two claims in earlier drafts of this file were wrong and are recorded rather than quietly edited: the OpenHands "zero caching" headline (wrong repo — frontend, not SDK), and BU-2's "images are not cacheable" (contradicted by Anthropic's own docs). Both were caught before publication. The pattern in each case was asserting from a plausible inference instead of the primary source.

---

## Disclosure status

Nothing published. No maintainer contacted yet. Per plan: findings go to maintainers with an offered PR **before** anything public. OH-1, SA-1, and BU-1 are each a one-line change and all three are good first PRs.
