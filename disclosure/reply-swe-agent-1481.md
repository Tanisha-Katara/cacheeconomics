# Reply draft — SWE-agent/SWE-agent#1481

Responding to @EvolveAegis, who added 20 trials of local-estimate versus
provider-reported usage across two providers, plus a downstream consequence I
had missed.

Their precheck observation is the lead. Everything they cite was checked first:
commit `3ea751c` exists, the precheck is where they say it is, and
`response.usage` appears zero times in `models.py` at that commit.

---

The precheck is the part I missed, and it is worse than what I filed. It reads
the same estimate, so when the estimate runs low the context check fires late and
a run carries on past the limit it should have stopped at.

`models.py:701`:

```python
elif input_tokens > self.model_max_input_tokens > 0:
    raise ContextWindowExceededError(msg)
```

`input_tokens` there is `litellm.utils.token_counter` from fifteen lines earlier,
on the cache_control-stripped copy. At your glm median of 0.75 that puts the
boundary about a third of a window out. The run fails later and somewhere less
obvious.

Two different bugs with the same fix, and I would keep them apart.

Mine was that a cache read, a cache write and uncached input are
indistinguishable in the project's telemetry. On Anthropic those bill at 0.1x,
1.25x and 1x, so a trajectory can shift heavily between them while `tokens_sent`
barely moves.

Yours is that the total is wrong too, on providers where the local tokenizer
diverges. The kimi output ratio of 0.0 to 0.26 is a third thing again: reasoning
tokens a text-based estimate cannot see at all.

All three go away if the ledger books `response.usage` when it is there.
Confirming your count: `grep -c 'response\.usage'` on `models.py` at `3ea751c`
returns 0.

One caveat on my side. My measurements were Anthropic-only, so I cannot
corroborate the glm and kimi ratios, only that the mechanism you describe is the
one I found. The 20/20 direction is more convincing than anything I have on the
totals.

Method and per-call rows for the Anthropic side:
https://github.com/Tanisha-Katara/cacheeconomics/tree/main/tier-b/evidence
