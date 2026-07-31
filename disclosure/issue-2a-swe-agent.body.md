`models.py` never reads `response.usage`. Grepping for `response.usage`, `cache_read_input`, `cache_creation_input`, or `prompt_tokens_details` returns zero hits.

Input tokens get counted client-side instead:

```python
# models.py:684-694
messages_no_cache_control = copy.deepcopy(messages)
for message in messages_no_cache_control:
    if "cache_control" in message:
        del message["cache_control"]
    ...
input_tokens: int = litellm.utils.token_counter(
    messages=messages_no_cache_control, ...
)
```

and that's the value that reaches the stats at `models.py:780`:

```python
self._update_stats(input_tokens=input_tokens, output_tokens=output_tokens, cost=cost)
```

So a cache read, a cache write, and uncached input all look identical in `tokens_sent`. They bill at 0.1x, 1.25x and 1x, but each one bumps the same counter by the same amount. Nothing in SWE-agent's own telemetry would tell you whether the prefix caching in `history_processors.py` is working well, working badly, or quietly not working at all.

I should be careful not to overstate this. `cost` comes from `litellm.cost_calculator.completion_cost(response, ...)` at `models.py:744`, and LiteLLM's calculator does read usage, so the dollar figure is probably right. It's the token statistics that are blind, not necessarily the cost.

Reading `response.usage` and recording the three classes separately is a small change. It turns cache behaviour from invisible to measurable, and it's what would let you judge the companion TTL issue on your own runs instead of taking an outsider's word for it.

(Filed alongside a second issue about the 1-hour cache TTL. This one is the prerequisite: without usage-class visibility you can't evaluate that one on your own runs.)

Filed by Tanisha Katara, CEO, KCG Consulting LLC.


Analysed at commit `3ea751c087f32b16e039a2233dd6eefecef325d5`
