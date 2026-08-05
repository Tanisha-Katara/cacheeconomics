"""Relocation-lite: move the volatile block down, when that is provably safe.

The highest-value transform available, and the most dangerous. A 90-token
session header sitting above 34,000 stable tokens makes every one of those
tokens uncacheable; moving it down recovers all of them. No amount of marker
placement can do that — the marker has nowhere good to go.

It is dangerous because prompt sections are not commutative. Moving content
changes instruction priority, recency, tool selection, and system-versus-user
authority. A cheaper prompt that behaves differently is not a saving, it is a
regression with a cost report attached.

So this module proposes and never applies. Every move carries the tokens it
unlocks, a risk class, the mechanism that preserves authority, whether a
behavioural eval is required, and how to roll it back.

The distinction that makes this useful rather than permanently blocked: moving
a segment to the end of *its own authority block* is a different act from moving
it out of that block. The first changes instruction order within the system
prompt. The second demotes system content to user authority and needs a
documented mechanism the model may not have. Collapsing the two into "a move"
blocks the safe case for the unsafe case's reasons.

Deliberately NOT here, and left to the full allocator:
  - reordering stable content among itself (no cache benefit, all risk)
  - splitting a segment to move only part of it
  - reordering conversation history
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from . import registry
from .allocate import Plan, _place
from .trace import UNATTRIBUTED, Request, Segment

# Two authority classes. Within one, order changes instruction priority. Across
# one, it changes how strongly the model weighs the content at all.
AUTHORITY_ROLES = {"system", "tools"}


def _authority(role: str) -> str:
    """How strongly the model weighs this content."""
    return "system" if role in AUTHORITY_ROLES else "conversation"


def _container(role: str) -> str:
    """Which top-level field of the request this segment is emitted in.

    Distinct from authority, and conflating the two produced a proposal nobody
    could ship. `tools` and `system` share an authority class, so treating that
    class as one movable block let a volatile tool be reordered after system
    content -- but on the wire `tools` is its own top-level field emitted before
    `system`, and there is no way to interleave them. The relocation arm was
    pricing a saving for a prompt no client can send.
    """
    if role == "tools":
        return "tools"
    if role == "system":
        return "system"
    return "messages"


@dataclass(frozen=True)
class Move:
    """One proposed relocation. Never applied automatically."""
    segment_index: int
    label: str
    tokens_moved: int
    tokens_unlocked: int
    risk: str                 # low | medium | high | blocked
    reason: str
    scope: str = ""           # within-container | cross-authority | history-reorder
    mechanism: str = ""
    eval_required: bool = True
    rollback: str = "revert the ordering change; no data migration, no state"
    blocked_by: str = ""
    new_order: tuple = field(default=(), repr=False)

    @property
    def applicable(self) -> bool:
        return self.risk != "blocked"

    def __str__(self):
        head = f"[{self.risk}] move segment {self.segment_index} ({self.label})"
        body = (f"    unlocks {self.tokens_unlocked:,} cacheable tokens by moving "
                f"{self.tokens_moved:,}\n    {self.reason}")
        if self.scope:
            body += f"\n    scope: {self.scope}"
        if self.mechanism:
            body += f"\n    mechanism: {self.mechanism}"
        if self.blocked_by:
            body += f"\n    blocked: {self.blocked_by}"
        else:
            body += f"\n    eval required: {'yes' if self.eval_required else 'no'}"
            body += f"\n    rollback: {self.rollback}"
        return head + "\n" + body


def observed_reordering(reqs: list[Request]) -> set:
    """Segments the shipping code has genuinely emitted on both sides of another.

    This is the movability evidence, and it has to be *relative* order. An
    earlier version recorded each segment's ordinal position and treated
    variance as proof the placement was free. Positions are derived by sorting
    on `index`, so emission order always follows index and the ordinal can only
    shift when an earlier segment is absent. That made "this segment is
    optional" indistinguishable from "this segment moves", and the latter was
    then used to waive the eval requirement on a relocation.

    Pairwise relative order is immune to that: segment i counts as movable only
    if some segment j has been observed both before and after it. Under the
    current schema nothing satisfies that, so this correctly returns empty and
    no move is waived on false evidence. It starts producing signal the moment
    an adapter records true wire order rather than a derived index.
    """
    rel = defaultdict(set)
    for r in reqs:
        order = [x.index for x in sorted(r.segments, key=lambda x: x.index)]
        pos = {idx: p for p, idx in enumerate(order)}
        for a in order:
            for b in order:
                if a != b:
                    rel[(a, b)].add(pos[a] < pos[b])
    return {a for (a, _b), seen in rel.items() if len(seen) > 1}


def stable_prefix_tokens(order, segs: dict, volatility: dict) -> int:
    """Tokens in the leading run of stable segments — the cacheable span.

    The same definition the allocator uses to place a marker, so a move's
    reported gain is exactly what the simulator will price.
    """
    total = 0
    for i in order:
        if volatility.get(i, 1) > 1:
            break
        total += segs[i].tokens
    return total


def scopes_of(reqs: list[Request]) -> list[tuple]:
    """Every (target_id, model) pair present. Relocation safety is per scope."""
    return sorted({(r.target_id, r.model) for r in reqs})


def _authority_mechanism(model: str, target_id: str) -> tuple[bool, str]:
    """Can system-authority content move into messages[] and stay authoritative?"""
    try:
        row = registry.target(target_id)
    except registry.RegistryError:
        return False, ""
    apr = row.get("authority_preserving_relocation") or {}
    if not apr.get("supported"):
        return False, ""
    if model in (apr.get("excluded") or []):
        return False, ""
    if model not in (apr.get("models") or []):
        return False, ""       # absent means unproven, not permitted
    return True, apr.get("mechanism", "")


def _mechanism_for_all(scopes: list[tuple]) -> tuple[bool, str, str]:
    """A cross-authority move is safe only if every scope in the group permits it.

    Groups are not homogeneous. A real 29-day trace analysed here had four
    sessions switch model mid-conversation, so a group can contain both Opus 5,
    which supports authority-preserving relocation, and Sonnet 5, which the
    registry explicitly excludes. Clearing the move on whichever request
    happened to be first would hand a client a prompt reordering that silently
    demotes system content to user authority on part of their traffic — and
    price it as a saving.

    Returns (permitted, mechanism, name of the first scope that refused).
    """
    mech = ""
    for target_id, model in scopes:
        ok, m = _authority_mechanism(model, target_id)
        if not ok:
            return False, "", f"{model} on {target_id}"
        mech = mech or m
    return bool(scopes), mech, ""


def _classify(i: int, order: list[int], segs: dict, volatility: dict,
              reordered: set, scopes: list[tuple]) -> Move | None:
    """Where segment i should go, and what that move costs in risk.

    Tries the least invasive placement that actually unlocks tokens: to the end
    of its own authority block first, out of the block only if that is not
    enough.
    """
    s = segs[i]
    base = stable_prefix_tokens(order, segs, volatility)
    rest = [j for j in order if j != i]
    mine = _authority(s.role)

    # Candidate 1: last position still inside this segment's own *container*.
    #
    # Container, not authority. A tools segment stays among tools and a system
    # segment stays among system blocks, because those are separate top-level
    # fields and no ordering of the request can interleave them.
    #
    # The messages container is excluded entirely: it is a transcript, so moving
    # a turn within it changes recency and the order events appear to have
    # happened in. That is reordering history, which this module does not do at
    # any risk class.
    my_container = _container(s.role)
    within = None
    if my_container in ("system", "tools"):
        same_block = [p for p, j in enumerate(rest)
                      if _container(segs[j].role) == my_container]
        if same_block:
            p = max(same_block)
            within = rest[:p + 1] + [i] + rest[p + 1:]

    # Candidate 2: the very end of the request, crossing out of the block.
    across = rest + [i]

    gain_within = (stable_prefix_tokens(within, segs, volatility) - base) if within else 0
    gain_across = stable_prefix_tokens(across, segs, volatility) - base

    if gain_within <= 0 and gain_across <= 0:
        return None                       # moving it recovers nothing

    label = s.label or s.role

    # Already demonstrably movable: the shipping code has emitted this segment
    # on both sides of another one, so its placement is not load-bearing.
    #
    # Applies to the within-container candidate only. Observing that a segment
    # moves inside the system block says nothing about whether it may leave the
    # block, and an earlier version let this branch return a cross-authority
    # move at low risk with no eval — skipping the mechanism check entirely,
    # including for models the registry explicitly excludes.
    if i in reordered and gain_within > 0:
        return Move(i, label, s.tokens, gain_within, "low",
                    "the shipping code already emits this segment at more than one "
                    "position relative to its neighbours, so its placement within "
                    "the field is not load-bearing",
                    scope="within-container", eval_required=False,
                    new_order=tuple(within))

    # Prefer staying inside the authority block. Instruction order changes;
    # authority does not, and no special mechanism is needed.
    if gain_within > 0:
        return Move(i, label, s.tokens, gain_within, "medium",
                    f"moved to the end of the {my_container} field, so it stays "
                    f"{mine}-authority and only its position among other "
                    f"{my_container} content changes",
                    scope="within-container",
                    mechanism=f"reorder within the {my_container} field",
                    eval_required=True, new_order=tuple(within))

    # Only worth crossing the boundary if staying inside recovers nothing.
    # Tools are not system text. They share an authority class, which is why
    # round 17 separated container from authority for the within-block case --
    # but this branch was still keyed on authority, so a volatile tool
    # definition could be moved into messages[] via the role:system mechanism.
    # That mechanism relocates *instructions*; a tool definition moved into a
    # message stops being a tool at all, and the order was priced as a saving
    # for a prompt no client can send.
    if my_container == "tools":
        return Move(i, label, s.tokens, gain_across, "blocked",
                    "recovering these tokens needs this tool definition below the "
                    "marker, which would move it out of the tools field entirely",
                    scope="cross-container",
                    blocked_by="tools are a separate top-level field with no "
                               "authority-preserving relocation mechanism; a tool moved "
                               "into messages[] is no longer a tool")

    if mine == "system":
        ok, mech, refused_by = _mechanism_for_all(scopes)
        if not ok:
            extra = (f" (this group spans {len(scopes)} model/surface combinations; "
                     f"a move has to be safe for all of them)" if len(scopes) > 1 else "")
            return Move(i, label, s.tokens, gain_across, "blocked",
                        "recovering these tokens needs this segment below the marker, "
                        "which would move it out of the system block and demote it to "
                        "user authority",
                        scope="cross-authority",
                        blocked_by=f"{refused_by} has no recorded authority-preserving "
                                   f"relocation mechanism{extra}")
        return Move(i, label, s.tokens, gain_across, "medium",
                    "system-authority content moved below the marker without demoting it",
                    scope="cross-authority", mechanism=mech,
                    eval_required=True, new_order=tuple(across))

    # A conversation segment moved to the end crosses no authority boundary; it
    # reorders the transcript. Different act, different risk, so it gets its own
    # name rather than being filed under cross-authority.
    return Move(i, label, s.tokens, gain_across, "high",
                "recovering these tokens needs this turn moved later in the "
                "conversation, which changes recency and the order events appear "
                "to have happened in",
                scope="history-reorder", eval_required=True, new_order=tuple(across))


# Risk classes applied by default. `high` is proposed and reported but never
# simulated as a saving: an unevaluated behavioural change is not a result.
DEFAULT_APPLIED_RISKS = ("low", "medium")


def _reject_rates(volatility: dict) -> None:
    """Refuse a change-rate map where a distinct-value count is required.

    A count is an integer of at least one: a segment that was observed at all
    took at least one value. So anything below 1, and anything fractional, is a
    rate. The dangerous case is a rate map holding only 0.0 and 1.0, which
    passes for counts and reverses the meaning of every entry.
    """
    bad = [(i, v) for i, v in (volatility or {}).items()
           if not isinstance(v, int) or isinstance(v, bool) or v < 1]
    if bad:
        i, v = bad[0]
        raise TypeError(
            f"propose() wants distinct-value counts from observed_volatility(), "
            f"and got what looks like a change rate: segment {i} = {v!r}. The "
            f"two are inverses -- rate 1.0 means it changes every request, "
            f"count 1 means it never does -- so this would propose no moves on "
            f"the worst prompt in the trace.")


def propose(reqs: list[Request], volatility: dict, model: str | None = None,
            target_id: str | None = None,
            risks: tuple = DEFAULT_APPLIED_RISKS) -> tuple[list[Move], list[int]]:
    """Moves worth making, and the emission order they produce.

    Only ever considers the first volatile segment. The cacheable prefix ends
    there by definition, so nothing behind it can extend the prefix, and moving
    anything else is risk without benefit. Once it moves, the next volatile
    segment becomes the blocker and the loop repeats.

    `volatility` is `observed_volatility`'s **counts of distinct values**, not
    `observed_change_rates`' **rates**. The two invert each other -- a rate of
    1.0 means the segment changes on every request, a count of 1 means it never
    changes -- so handing this the wrong one silently proposes no moves on the
    most volatile prompt there is. Guarded rather than documented, because the
    author of this function made that exact substitution within minutes of the
    two existing side by side.
    """
    if not reqs:
        return [], []
    _reject_rates(volatility)
    # One order is emitted for the whole group, so every request in it has to
    # agree on what an index means. Building `segs` by last-writer-wins across
    # heterogeneous prompts let index 0 be `system` in requests that carry a
    # system prompt and `user` in requests that do not -- and the resulting
    # order was then applied to both, reordering conversation history under the
    # label of a safe system move.
    #
    # Fail closed. Splitting the group by prompt shape is the better answer and
    # needs the simulator to carry more than one order at a time; until then,
    # proposing nothing beats proposing a reorder for the wrong prompt.
    # Role and container are not enough to call two prompts the same shape. A
    # volatile session header and a stable safety policy are both system/system,
    # so index 0 passed the check, segs was built last-writer-wins, and a move
    # could be proposed for content that was not the blocker in half the group.
    # The section label is what distinguishes them, and it is the same field
    # deliberately excluded from *volatility* identity -- there it hid drift,
    # here it is the only thing that says two prompts have the same schema.
    shapes: dict[int, set] = defaultdict(set)
    for r in reqs:
        for sg in r.segments:
            shapes[sg.index].add((sg.role, _container(sg.role), sg.label or sg.role))
    conflicting = sorted(i for i, kinds in shapes.items() if len(kinds) > 1)
    if conflicting:
        i = conflicting[0]
        roles = ", ".join(sorted({k[2] for k in shapes[i]}))
        return ([Move(i, f"index {i}", 0, 0, "blocked",
                      "this group mixes prompt shapes, so one ordering cannot be applied "
                      "to all of it safely",
                      blocked_by=f"segment index {i} is {roles} in different requests"
                                 f"{'' if len(conflicting) == 1 else f', and {len(conflicting) - 1} other index(es) also disagree'}"
                                 f". Applying one order across them would move content "
                                 f"between containers in the requests that disagree.")],
                sorted(shapes))

    segs: dict[int, Segment] = {s.index: s for r in reqs for s in r.segments}
    reordered = observed_reordering(reqs)
    # Derived from the whole group by default. An explicit model/target is a
    # single-scope override for callers that already know the group is uniform.
    #
    # `UNATTRIBUTED`, not `anthropic/direct`. A caller who supplied `model` and
    # omitted `target_id` had a first-party surface invented for them, and the
    # surface is what `_authority_mechanism` reads to decide whether
    # system-authority content may leave the system block at all. So the
    # override answered a different question from the derived path on identical
    # requests. Measured, on twelve UNATTRIBUTED claude-opus-5 requests:
    #
    #   propose(reqs, vol, model='claude-opus-5')  -> [medium] cross-authority,
    #       mechanism 'role:system message inside messages[]'
    #   propose(reqs, vol)                         -> [blocked], "claude-opus-5
    #       on unknown/unattributed has no recorded authority-preserving
    #       relocation mechanism"
    #
    # The medium one is a recommendation to rewrite a prompt using a mechanism
    # recorded for one named surface, handed to somebody whose surface nobody
    # stated. `_authority_mechanism` already refuses an unknown surface -- it
    # returns (False, "") when `registry.target` raises -- so passing the
    # absence of a surface through makes the override agree with the derived
    # path instead of overruling it.
    scopes = ([(target_id or UNATTRIBUTED, model)] if model
              else scopes_of(reqs))
    order = sorted(segs)
    moves: list[Move] = []
    seen: set[int] = set()

    while True:
        blocker = next((i for i in order if volatility.get(i, 1) > 1), None)
        if blocker is None or blocker in seen:
            break
        seen.add(blocker)
        m = _classify(blocker, order, segs, volatility, reordered, scopes)
        if m is None:
            break                         # moving it recovers nothing; stop
        moves.append(m)
        if m.applicable and m.risk in risks:
            order = list(m.new_order)
        else:
            break                         # cannot get past this one
    return moves, order


def relocation_lite(req: Request, volatility: dict | None = None,
                    cadence_seconds: float | None = None,
                    moves: list[Move] | None = None,
                    order: list[int] | None = None,
                    risks: tuple = DEFAULT_APPLIED_RISKS, **_) -> Plan:
    """Marker placement over a relocated order. The fourth bake-off arm.

    Separate from allocator-lite on purpose. Two arms answer two questions:
    allocator-lite says whether placement alone beats automatic injection, and
    this one says whether placement plus safe moves does. Folded together they
    would produce one number answering neither, and the two have very different
    deployment costs — one is a config change, the other needs an eval.
    """
    volatility = volatility or {}
    moves = moves or []
    applied = [m for m in moves if m.applicable and m.risk in risks]
    if order is None:
        order = list(applied[-1].new_order) if applied else sorted(
            s.index for s in req.segments)

    by_i = {s.index: s for s in req.segments}
    emission = [by_i[i] for i in order if i in by_i]
    notes = [f"moved segment {m.segment_index} ({m.label}) down [{m.risk}, {m.scope}]"
             for m in applied]
    plan = _place(req, emission, "relocation-lite", volatility, cadence_seconds,
                  order=order, extra_notes=notes)
    if any(m.eval_required for m in applied):
        plan.notes.append("EVAL REQUIRED: this arm changes prompt ordering. The saving "
                          "is Modeled and not claimable until a behavioural eval shows "
                          "no material regression.")
    return plan
