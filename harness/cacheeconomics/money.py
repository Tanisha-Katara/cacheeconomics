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

# How a released figure earned its release. Withheld figures carry "".
RECONCILED = "reconciled"   # checked against an invoice the client supplied
DRAFT = "draft"             # released by --allow-unreconciled; not for forwarding
RELEASES = (RECONCILED, DRAFT)


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

    `__str__`, `__repr__` and `__format__` all honour the gate.

    What this does NOT claim, because the first version of this docstring did
    and it was wrong: that `raw()` is the only way to reach the number. It is
    not. `f._usd` reads it, `type(f)._usd.__get__(f, type(f))` reads it through
    the slot descriptor, and `f.__reduce_ex__(4)` carries it in the pickle
    payload. A single-underscore attribute is a convention, not a wall, and
    claiming otherwise is worse than the gap because it stops anyone looking.

    The property that actually holds, and the one worth having: no *generic*
    or *accidental* route publishes the value. Anything that reaches it names
    it, which is what makes `_usd` and `raw()` both greppable at review time.
    The routes closed here were the ones nobody writes deliberately -- a
    serializer default, a dataclass helper, `vars`, an in-place assignment --
    and those are how a withheld number actually escaped in practice.
    """

    __slots__ = ("_usd", "basis", "released", "withheld_because", "released_as",
                 "projected")

    def __init__(self, usd: float, basis: str = MODELED, *,
                 released: bool = False,
                 withheld_because: str = "not released",
                 released_as: str = "",
                 projected: bool = False):
        if basis not in BASES:
            raise ValueError(f"basis must be one of {BASES}, got {basis!r}")
        if released_as and released_as not in RELEASES:
            raise ValueError(f"released_as must be one of {RELEASES}, "
                             f"got {released_as!r}")
        object.__setattr__(self, "_usd", usd)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "released", bool(released))
        object.__setattr__(self, "withheld_because", withheld_because)
        # Whether this number was extrapolated past the observed window.
        #
        # It exists to make a class *enumerable*. The projection floor was
        # written as a check one caller performed, so "every monthly figure is
        # gated" was a sentence in a commit message with nothing able to
        # contradict it -- and it was false: the headline monthly spend was
        # gated while six per-finding monthly figures were not. A flag set at
        # the single extrapolation site turns that claim into something a test
        # can walk the object graph and check, which is the only reason to
        # widen this class.
        #
        # Set by `_monthly`, never by hand at a call site. A caller that has to
        # remember is the thing being replaced.
        object.__setattr__(self, "projected", bool(projected))
        # How it earned release, not merely that it did. `released` was a bare
        # bool, so a figure released by `--allow-unreconciled` was byte-identical
        # to one an invoice had checked -- same value, same basis, same string --
        # and no renderer *could* mark one and not the other. Measured on the
        # demo trace: both rendered "$229" with nothing to tell them apart.
        object.__setattr__(self, "released_as",
                           (released_as or RECONCILED) if released else "")

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
                and self.withheld_because == other.withheld_because
                and self.released_as == other.released_as
                and self.projected == other.projected)

    def __hash__(self):
        return hash((self._usd, self.basis, self.released, self.released_as,
                     self.projected))

    def __reduce__(self):
        """Pickle through the constructor, not through a state dict."""
        return (Figure, (self._usd, self.basis),
                {"released": self.released,
                 "withheld_because": self.withheld_because,
                 "released_as": self.released_as,
                 "projected": self.projected})

    def __setstate__(self, state):
        object.__setattr__(self, "released", state["released"])
        object.__setattr__(self, "withheld_because", state["withheld_because"])
        object.__setattr__(self, "released_as", state.get("released_as", ""))
        object.__setattr__(self, "projected", state.get("projected", False))

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

    def release(self, ok: bool, because: str = "", *, as_: str = "") -> "Figure":
        """The only way to change release state, and it makes a new Figure.

        `as_` records *how* it was released. When it is not given, an
        already-released figure keeps the provenance it has and a withheld one
        becomes RECONCILED.

        That second half is the original rule and the reason for it stands:
        RECONCILED is what every caller meant before the distinction existed,
        and defaulting to DRAFT would relabel every honest figure. But the rule
        was written for figures that had never been released, and applied to
        every figure, so `draft.release(True)` silently upgraded a figure the
        client was told was unchecked into one an invoice had verified.

        `__abs__` had the identical shape and was fixed first, because it was
        reachable. This one is not reachable today -- all three call sites take
        a withheld figure straight from `_monthly` -- and it was deferred on
        that basis. It is fixed now because keeping it meant keeping a test
        whose label named the path while its body stepped around it, which is
        worse than either fixing or failing honestly.
        """
        keep = self.released_as if (self.released and not as_) else as_
        return Figure(self._usd, self.basis, released=bool(ok),
                      withheld_because=("" if ok
                                        else (because or self.withheld_because)),
                      released_as=(keep or RECONCILED) if ok else "",
                      projected=self.projected)

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

        Carries `released_as` and `projected`, which it did not. Measured
        before this line changed: `abs(Figure(-12.34).release(True,
        as_=DRAFT))` came back `released_as='reconciled'`, so taking the
        magnitude of a draft figure laundered it into one an invoice had
        checked. It did not surface a wrong banner today only because
        `_is_draft` reads `a.spend` rather than this transient copy -- the
        defect was one caller away from mattering, and "a `Figure` method
        silently drops provenance" is the same shape as everything else in
        this file's history.
        """
        return Figure(abs(self._usd), self.basis, released=self.released,
                      withheld_because=self.withheld_because,
                      released_as=self.released_as, projected=self.projected)


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


def release_map(mapping: dict, ok: bool, because: str = "", *,
                as_: str = "") -> dict:
    """Release every Figure in a dict, leaving non-Figures alone.

    Release is one decision applied to everything at once. Doing it per output
    is what produced two renderers that disagreed about the same gate.
    """
    return {k: (v.release(ok, because, as_=as_) if isinstance(v, Figure) else v)
            for k, v in mapping.items()}
