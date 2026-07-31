"""The full allocator: tier merge under budget, relocation, multi-target.

Build item 8, and the thing allocator-lite deliberately is not. Three
differences, each of which is a reason it was held back:

**It spends the whole marker budget.** Allocator-lite finds the deepest stable
boundary, places one marker, and leaves the other three unused along with
everything past the first volatile segment. This partitions the prompt into
tiers and solves for the cuts, so a semi-stable middle section can hold its own
entry instead of being stranded behind the first thing that moved.

**It moves content.** Relocation and placement are solved together rather than
in sequence, because they are the same decision: a segment is worth moving
exactly when moving it lets a marker sit deeper.

**It reads the surface from the registry.** Budget, lifetimes and multipliers
all come from the target row, so Bedrock's checkpoint model, an implicit-prefix
surface with no markers at all, and a surface whose read multiplier this cost
model cannot express each produce an abstention with the reason attached rather
than an Anthropic-shaped answer applied to something that is not Anthropic.

**It is not a bake-off arm.** Gate 1 measures four arms, and this is the thing
Gate 1 decides whether to build. Adding it to `POLICIES` would put the verdict
inside the experiment: the full allocator would beat the baseline, the gate
would pass, and the gate would have measured itself. It has to be requested by
name.
"""

from __future__ import annotations

from . import tiers
from .allocate import Plan
from .relocate import DEFAULT_APPLIED_RISKS, Move
from .trace import Request


def allocator_full(req: Request, rates: dict | None = None,
                   gaps: list | None = None,
                   moves: list[Move] | None = None,
                   order: list[int] | None = None,
                   risks: tuple = DEFAULT_APPLIED_RISKS,
                   relocate: bool = True, **_) -> Plan:
    """Marker placement solved as a partition, over a possibly-relocated order.

    `rates` is per-segment change rate and `gaps` the observed inter-request
    gaps, both for the pool this request belongs to. Neither has a default: with
    no rates every segment is treated as changing on every request, and with no
    gaps no entry can be shown to survive to the next request. Both refusals
    produce a plan with no markers and a note saying which input was missing,
    which is the honest answer to "should I cache this" when nobody has looked
    at the traffic.
    """
    applied = [m for m in (moves or []) if m.applicable and m.risk in risks] \
        if relocate else []
    if order is None:
        order = list(applied[-1].new_order) if applied else sorted(
            s.index for s in req.segments)
    elif not relocate:
        order = sorted(s.index for s in req.segments)

    by_i = {s.index: s for s in req.segments}
    emission = [by_i[i] for i in order if i in by_i]
    emission += [s for s in sorted(req.segments, key=lambda s: s.index)
                 if s.index not in set(order)]
    if not emission:
        return Plan("allocator-full", [], {}, ["no segments to place markers in"])

    notes = [f"moved segment {m.segment_index} ({m.label}) down [{m.risk}, {m.scope}]"
             for m in applied]
    wire_order = [s.index for s in emission]

    try:
        alloc = tiers.allocate(emission, rates or {}, target_id=req.target_id,
                               model=req.model, gaps=gaps or [])
    except tiers.Unsupported as e:
        return Plan("allocator-full", [], {},
                    notes + [f"no allocation attempted: {e}"], wire_order)

    marker_indices = [emission[t.marker_position].index for t in alloc.tiers]
    ttls = {emission[t.marker_position].index: t.ttl for t in alloc.tiers}

    notes += list(alloc.notes)
    for t in alloc.tiers:
        notes.append(
            f"tier through segment {emission[t.marker_position].index} "
            f"({emission[t.marker_position].label or emission[t.marker_position].role}): "
            f"{t.tokens:,} tokens, {t.ttl}, served from cache "
            f"{t.hit_probability:.0%} of the time")
    if alloc.tiers and alloc.saving_ratio is not None:
        notes.append(
            f"modelled input-token cost {alloc.saving_ratio:.1%} below sending "
            f"the prompt uncached, before any write premium is reconciled "
            f"against an invoice")
    if alloc.searched:
        notes.append("candidates scored: " + ", ".join(
            f"{label} {'not feasible' if cost is None else format(cost, ',.0f')}"
            for label, cost in alloc.searched))
    if any(m.eval_required for m in applied):
        notes.append("EVAL REQUIRED: this plan changes prompt ordering. The saving "
                     "is Modeled and not claimable until a behavioural eval shows "
                     "no material regression.")

    plan = Plan("allocator-full", marker_indices, ttls, notes,
                wire_order if applied else None)
    plan.allocation = alloc
    return plan


def allocator_full_no_relocation(req: Request, **kw) -> Plan:
    """Tier merge only, leaving the prompt in the order it was authored.

    Worth having separately because the two halves have very different
    deployment costs: this one is a config change, and the relocating version
    needs a behavioural eval before anybody can claim its saving.
    """
    kw.pop("relocate", None)
    return allocator_full(req, relocate=False, **kw)


# Deliberately not merged into `allocate.POLICIES`. See the module docstring:
# the bake-off decides whether this should exist, so it cannot be in the
# bake-off.
GATED_POLICIES = {"allocator-full": allocator_full,
                  "allocator-full-no-reloc": allocator_full_no_relocation}
