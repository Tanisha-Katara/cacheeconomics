# Reply draft — SWE-agent/SWE-agent#1481

Responding to @EvolveAegis, who added 20 trials of local-estimate versus
provider-reported usage across two providers, plus a downstream consequence I
had missed.

Insight first: their precheck observation turns this from an accounting bug into
a correctness bug. Their numbers were verified before this was written — commit
`3ea751c` exists, the precheck is where they say it is, and `response.usage`
appears zero times in `models.py` at that commit.

---

The precheck consequence makes this a correctness bug rather than an accounting
one, and that is a better argument for the fix than the one I filed.

At `models.py:701`:

```python
elif input_tokens > self.model_max_input_tokens > 0:
    raise ContextWindowExceededError(msg)
```

`input_tokens` comes from `litellm.utils.token_counter` fifteen lines earlier, on
the cache_control-stripped copy. So a systematic under-count does not only
misreport the ledger, it moves the context-window boundary. At your glm median of
0.75 the check fires about a third of a window late. A run that should have
stopped instead keeps going and fails somewhere less obvious.

Worth keeping the two findings apart, because they are different failure modes
that happen to share a fix.

Mine was that a cache read, a cache write and uncached input are
indistinguishable in the project's telemetry. On Anthropic those bill at 0.1x,
1.25x and 1x, so a trajectory can shift heavily between them while `tokens_sent`
barely moves.

Yours is that the total is wrong too, on providers where the local tokenizer
diverges. The kimi output ratio of 0.0 to 0.26 is a third thing again: reasoning
tokens a text-based estimate cannot see at all.

Three separate ways the ledger departs from what was billed, all fixed by booking
`response.usage` when it is present. Confirming your count: `grep -c
'response\.usage'` on `models.py` at `3ea751c` returns 0.

One caveat on my side. My measurements were Anthropic-only, so I cannot
corroborate the glm and kimi ratios, only that the mechanism you describe is the
one I found. The 20/20 direction is more convincing than anything I have on the
totals.

Method and per-call rows for the Anthropic side:
https://github.com/Tanisha-Katara/cacheeconomics/tree/main/tier-b/evidence
