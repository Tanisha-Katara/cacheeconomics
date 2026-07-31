# Reply draft — BerriAI/litellm#35011

Re-verified against `main` on **30 Jul 2026**, immediately before sending. Every
number in the comment below still holds:

| Claim | Re-checked | Status |
|---|---|---|
| entries claiming `supports_prompt_caching` | 623 | unchanged |
| ...of those with no `prompt_cache_min_tokens` | 499 | unchanged |
| `claude-fable-5` keys disagreeing | 512 on `claude-fable-5`, 1024 on `anthropic.`, `us.`, `eu.`, `global.` | unchanged |
| the 29 proposed keys still unfilled upstream | 29 of 29 | unchanged |

**The dependency worth stating in the comment.** Of the 29, twenty-six are
derivable from another key in the file with no ambiguity at all. The other three
— `azure_ai/claude-fable-5`, `vertex_ai/claude-fable-5`,
`vertex_ai/claude-fable-5@default` — are *not*, because the file contradicts
itself on Fable 5. Deriving them requires Problem 1 to be settled first.

That makes the two problems ordered rather than independent, which is worth
saying: someone filling in the 29 without resolving Fable 5 first has a
one-in-ten chance of propagating 1024 to three more keys, and 1024 is the wrong
value. Anthropic's published minimum for Fable 5 is 512
(`platform.claude.com/docs/en/build-with-claude/prompt-caching`, checked
2026-07-28), which is what the bare `claude-fable-5` key already carries.

Verification script and its output are reproducible from
`contrib/litellm/proposed-minimums.json` against the upstream file; the counts
above come from re-running it, not from the earlier draft.

The one thing this reply has to do is separate the 29 keys that can be filled in
from inside the file from the ~470 that cannot. If those get filled with guesses
the file becomes worse than it is now — a wrong minimum silently disables
caching, which is the exact failure the issue is about, whereas a missing one
only means "unknown".

---

## The comment

> That would be great — thank you for picking it up. I have the values and a test
> ready, so take whatever is useful.
>
> One thing worth splitting before you start, because it changes the size of the
> PR a lot.
>
> **Problem 1 — four keys.** `claude-fable-5` is 512 on the direct entry and 1024
> on `anthropic.claude-fable-5`, `us.`, `eu.` and `global.`. All four are the
> Bedrock-shaped keys for the same model, so 512 is the value.
>
> **Problem 2 — 29 keys, not 499.** The 499 figure is how many entries claim
> `supports_prompt_caching: true` with no minimum recorded. Only **29** of those
> are Anthropic models whose minimum is already in this file on another key, so
> those can be filled in from within the file itself with no external source.
> `claude-haiku-4-5` is the clearest example: eighteen keys, fourteen carry 4096,
> four carry nothing.
>
> Worth doing in that order, because they are not independent: 26 of the 29 are
> unambiguous, but `azure_ai/claude-fable-5`, `vertex_ai/claude-fable-5` and
> `vertex_ai/claude-fable-5@default` inherit whichever answer Problem 1 gets. Fill
> them before settling Fable 5 and you propagate 1024 to three more keys.
>
> The remaining ~470 are mostly non-Anthropic models, and I would leave those
> alone. I have not found published minimums for them, and a guessed minimum is
> worse than a missing one: too high and `is_prompt_caching_valid_prompt()`
> returns `False` for a prefix that would have cached, so the caller silently
> skips caching and pays full rate on every request. A missing value at least
> reads as unknown.
>
> Here are the 29, derived only from other keys in the same file:
>
> ```json
> <paste contrib/litellm/proposed-minimums.json>
> ```
>
> There is also a test that asserts every key for one model agrees on its
> minimum, which is what would have caught the Fable 5 split — happy to hand that
> over too, or open it as a separate PR if you would rather keep yours to the
> data change.
>
> Either way, shout if you want me to take any of it off your hands.

---

## Notes

- **Do not offer to take it over.** He asked first and volunteered. Handing him
  the data and the test is more useful than competing for the PR, and Gate 6 is
  about the contribution landing, not about whose name is on it.
- **This is contributor engagement, not maintainer engagement.** Gate 6 asks for
  a maintainer response. A contributor picking it up is a good early signal and
  does not close the gate — the gate closes when a maintainer merges or comments.
- **If he does fill in all 499**, that is worth a polite, specific objection on
  the PR rather than silence, with the `is_prompt_caching_valid_prompt()`
  mechanism spelled out. It is the same argument as the issue, applied to the
  fix.
