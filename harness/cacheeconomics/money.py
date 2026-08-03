"""A dollar figure that cannot be printed until it has been released.

Two adversarial reviews in a row found the same class of defect: a number
reaching an output through a path that forgot to check the reconciliation gate.
The text renderer printed figures the HTML renderer withheld. The finding rules
ran over requests the spend calculation had excluded. Each was fixed at the
site, and each fix left every *future* site free to make the same mistake,
because the rule lived in the caller's head rather than in the value.

So the rule moves into the value. A `Figure` renders as "[withheld: reason]"
until something explicitly releases it, and `float()` on an unreleased one
raises. A renderer that forgets the gate now produces a visibly withheld figure
or a loud error, instead of a plausible number nobody questions.

The escape hatch is `raw()` — deliberately ugly, deliberately greppable. It
exists because ranking and summing have to happen before release, and it is the
one place a reviewer needs to look to audit whether a guard was bypassed.
"""

from __future__ import annotations


# Epistemic status, from the plan. Every number carries one.
MEASURED = "measured"    # observed in historical usage fields
MODELED = "modeled"      # projected; always a range, pessimistic as headline
VERIFIED = "verified"    # observed in production after a change shipped

BASES = (MEASURED, MODELED, VERIFIED)


class WithheldFigure(Exception):
    """Raised when an unreleased figure is used as a number."""


class Figure:
    """A dollar amount plus whether anyone is allowed to see it.

    Deliberately not a dataclass, and `__slots__` deliberately leaves no
    `__dict__`. As a dataclass every generic helper walked straight past the
    gate: `dataclasses.asdict(f)` and `vars(f)` both returned `{'_usd': 123.45,
    ...}`, `json.dumps(f, default=lambda o: o.__dict__)` -- the most common
    serializer default anyone writes -- published the number, and
    `dataclasses.replace(f, released=True)` turned a withheld figure into "$123"
    in one line with no `raw()` at the call site to grep for.

    `__str__`, `__repr__` and `__format__` all honour the gate, so the value now
    has no route out except `raw()`, which is the audit point this module
    exists to provide.
    """

    __slots__ = ("_usd", "basis", "released", "withheld_because")

    def __init__(self, usd: float, basis: str = MODELED, *,
                 released: bool = False,
                 withheld_because: str = "not released"):
        if basis not in BASES:
            raise ValueError(f"basis must be one of {BASES}, got {basis!r}")
        object.__setattr__(self, "_usd", usd)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "released", bool(released))
        object.__setattr__(self, "withheld_because", withheld_because)

    def __setattr__(self, name, value):
        raise AttributeError(
            f"Figure is immutable; use release() rather than setting {name!r}. "
            f"Mutating `released` in place is how a withheld number gets "
            f"published without passing the gate.")

    def __eq__(self, other):
        if not isinstance(other, Figure):
            return NotImplemented
        return (self._usd == other._usd and self.basis == other.basis
                and self.released == other.released
                and self.withheld_because == other.withheld_because)

    def __hash__(self):
        return hash((self._usd, self.basis, self.released))

    def __reduce__(self):
        """Pickle through the constructor, not through a state dict."""
        return (Figure, (self._usd, self.basis),
                {"released": self.released,
                 "withheld_because": self.withheld_because})

    def __setstate__(self, state):
        object.__setattr__(self, "released", state["released"])
        object.__setattr__(self, "withheld_because", state["withheld_because"])

    def raw(self) -> float:
        """The underlying number, gate or no gate.

        For arithmetic and ranking that must happen before the gate is decided.
        Never for output. If this appears in a renderer, that is the bug this
        module exists to prevent.
        """
        return self._usd

    @property
    def amount(self) -> float:
        if not self.released:
            raise WithheldFigure(
                f"this figure is withheld ({self.withheld_because}) and must not be "
                f"published. Use raw() if you genuinely need the number for internal "
                f"arithmetic.")
        return self._usd

    def release(self, ok: bool, because: str = "") -> "Figure":
        """The only way to change release state, and it makes a new Figure."""
        return Figure(self._usd, self.basis, released=bool(ok),
                      withheld_because=("" if ok
                                        else (because or self.withheld_because)))

    def __float__(self) -> float:
        return self.amount

    def __repr__(self) -> str:
        """The dataclass auto-repr printed `_usd` in plain text.

        `__str__` and `__format__` both honour the gate; `repr` did not, and repr
        is what a traceback, a pytest diff, a log line and `print(list_of_figures)`
        all reach for. So the one path nobody writes deliberately was the one that
        published. There is no `raw()` at those call sites to grep for either,
        which is the property this module is built on.
        """
        if not self.released:
            return (f"Figure(withheld={self.withheld_because!r}, "
                    f"basis={self.basis!r})")
        return (f"Figure({self._usd!r}, basis={self.basis!r}, released=True)")

    def __str__(self) -> str:
        if not self.released:
            return f"[withheld: {self.withheld_because}]"
        return f"${self._usd:,.0f}" if abs(self._usd) >= 100 else f"${self._usd:,.2f}"

    def __format__(self, spec: str) -> str:
        # Deliberately ignores the format spec when withheld. A renderer asking
        # for "{:,.0f}" on a withheld figure gets the withheld text, not a
        # TypeError it might catch and paper over.
        if not self.released:
            return f"[withheld: {self.withheld_because}]"
        return format(self._usd, spec) if spec else str(self)

    def __bool__(self) -> bool:
        return self._usd != 0

    def __abs__(self) -> "Figure":
        """Magnitude, keeping the release state.

        Needed because the sign is often carried by the surrounding words: a
        report says "caching COST $0.83", not "caching COST $-0.83". Dropping
        abs() during a refactor produced exactly that sentence.
        """
        return Figure(abs(self._usd), self.basis, released=self.released,
                      withheld_because=self.withheld_because)


def measured(usd: float) -> Figure:
    return Figure(usd, MEASURED)


def modeled(usd: float) -> Figure:
    return Figure(usd, MODELED)


def draft_override_applies(invoice_supplied: bool, allow_unreconciled: bool) -> bool:
    """Whether `allow_unreconciled` may release figures here.

    It covers a *missing* invoice and nothing else. An invoice that was supplied
    and did not reconcile is evidence the figures are wrong, and an override
    meaning "I have not tied this to a bill yet" must not also mean "I tied it
    to a bill and it disagreed".

    One function because this rule is enforced in two modules and they had
    already diverged. The analyzer required `recon is None`; `bake_off` wrote
    `allow_unreconciled or reconciled is True`, so a $999,999 invoice against
    $0.27 of computed spend released anyway, as did a negative one and a NaN.
    The gate was fixed in the analyzer during the same session that left the
    simulator's copy untouched -- the exact twin-path shape this package's tests
    exist to catch, in the one pair nothing was comparing.
    """
    return allow_unreconciled and not invoice_supplied


def release_map(mapping: dict, ok: bool, because: str = "") -> dict:
    """Release every Figure in a dict, leaving non-Figures alone.

    Release is one decision applied to everything at once. Doing it per output
    is what produced two renderers that disagreed about the same gate.
    """
    return {k: (v.release(ok, because) if isinstance(v, Figure) else v)
            for k, v in mapping.items()}
