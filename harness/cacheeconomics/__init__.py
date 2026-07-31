"""cacheeconomics — a dated, sourced registry of prompt-cache behaviour,
a cost model that separates the four token classes, checks that catch the
failures providers do not report, and an analyzer that turns real traces into
a report someone can forward.

Deliberately not here: instrumentation and prefix diffing. Several projects
already do those well. What none of them carry is provenance.
"""
from . import registry, cost, checks, trace, analyzer, report, monitor  # noqa: F401
from .cost import Usage, Spend, price, ratios, ttl_crossover     # noqa: F401
from .checks import Status, Result, run_all, worst               # noqa: F401
from .trace import Tier, TraceSet, load_jsonl                    # noqa: F401
from .analyzer import Analysis, Finding, analyze                 # noqa: F401
from .monitor import Alert, Monitor                              # noqa: F401

__version__ = "0.2.0"
