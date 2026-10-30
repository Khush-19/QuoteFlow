"""Deterministic cost summaries and side-by-side comparison. No LLM involved."""

from __future__ import annotations

import re
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal

from pydantic import BaseModel

from .schema import Charge, CostSummary, FreightQuote, Money, UnknownItem

CENT = Decimal("0.01")
PRICED_STATUSES = {"included", "estimated"}

# (key, label, pattern) - first match wins, so specific patterns come first.
CANONICAL: list[tuple[str, str, str]] = [
    ("baf", "Bunker / fuel surcharge", r"\bbaf\b|bunker|fuel"),
    ("pss", "Peak season surcharge", r"peak season|\bpss\b"),
    ("duties_taxes", "Import duties & taxes", r"dut(y|ies)|\bvat\b|\btax"),
    ("origin_thc", "Origin terminal handling", r"\bothc\b|(origin|export).*(thc|terminal handling)|(thc|terminal handling).*(origin|export)"),
    ("dest_thc", "Destination terminal handling", r"\bdthc\b|(dest|import).*(thc|terminal handling)|(thc|terminal handling).*(dest|import)"),
    ("export_customs", "Export customs clearance", r"export.*(customs|clearance)|(customs|clearance).*export"),
    ("import_customs", "Import customs clearance", r"import.*(customs|clearance)|(customs|clearance).*(import|destination)"),
    ("insurance", "Cargo insurance", r"insurance"),
    ("delivery", "Delivery / on-carriage", r"on-carriage|delivery|trucking|haulage|drayage"),
    ("pickup", "Pickup / pre-carriage", r"pick ?-?up|pre-carriage|collection"),
    ("documentation", "Documentation", r"\bdoc|b/l|bill of lading"),
    ("main_freight", "Main freight", r"ocean freight|sea freight|air freight|freight charge|base rate|\bfreight\b"),
    ("destination_charges", "Destination charges (general)", r"destination charges|destination fees"),
]
LABELS = {k: label for k, label, _ in CANONICAL}
# A blanket line like "destination charges excluded" speaks for these specific items.
UMBRELLAS = {"destination_charges": {"dest_thc", "delivery", "import_customs"}}

Flag = Literal["priced_both", "gap_a", "gap_b", "unknown_both"]
CellStatus = Literal["included", "excluded", "not_mentioned", "estimated", "at_cost", "absent"]


def canonical_key(charge: Charge) -> str:
    name = charge.name.lower()
    for key, _label, pattern in CANONICAL:
        if re.search(pattern, name):
            return key
    slug = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return f"{charge.category}:{slug}"


def quantize(amount: Decimal) -> Decimal:
    return amount.quantize(CENT, rounding=ROUND_HALF_UP)


def _sum_by_currency(pairs: list[tuple[Decimal, str]]) -> list[Money]:
    totals: dict[str, Decimal] = defaultdict(Decimal)
    for amount, currency in pairs:
        totals[currency] += amount
    return [Money(amount=quantize(v), currency=c) for c, v in sorted(totals.items())]


def convert(moneys: list[Money], target: str, rates: dict[str, Decimal]) -> tuple[Money | None, list[str]]:
    """rates[c] = how many `target` units one unit of c is worth (user supplied).
    Returns (total, currencies missing a rate)."""
    missing = [m.currency for m in moneys if m.currency != target and m.currency not in rates]
    if missing:
        return None, missing
    total = sum((m.amount if m.currency == target else m.amount * rates[m.currency] for m in moneys), Decimal(0))
    return Money(amount=quantize(total), currency=target), []


def unknown_reason(c: Charge):
    if c.status in ("excluded", "at_cost", "not_mentioned"):
        return c.status
    if c.amount is None:
        return "amount_missing"
    if c.currency is None:
        return "currency_missing"
    return None


def summarize(quote: FreightQuote, target: str | None = None, rates: dict[str, Decimal] | None = None) -> CostSummary:
    rates = rates or {}
    included, estimated, unknown = [], [], []
    for c in quote.charges:
        reason = unknown_reason(c)
        if reason:
            unknown.append(UnknownItem(name=c.name, reason=reason))
        elif c.status == "included":
            included.append((c.line_total, c.currency))
        elif c.status == "estimated":
            estimated.append((c.line_total, c.currency))

    calculated = _sum_by_currency(included)
    estimates = _sum_by_currency(estimated)
    quoted = (
        Money(amount=quantize(quote.quoted_total), currency=quote.quoted_total_currency)
        if quote.quoted_total is not None and quote.quoted_total_currency
        else None
    )

    mismatch = None
    if quoted and len(calculated) == 1 and calculated[0].currency == quoted.currency:
        diff = quoted.amount - calculated[0].amount
        if diff != 0:
            mismatch = Money(amount=diff, currency=quoted.currency)

    target = target or (quoted.currency if quoted else (calculated[0].currency if calculated else None))
    converted, note = None, None
    if target and (calculated or estimates):
        converted, missing = convert(calculated + estimates, target, rates)
        if missing:
            note = f"No exchange rate given for {', '.join(sorted(set(missing)))} -> {target}; totals not combined."
        elif any(m.currency != target for m in calculated + estimates):
            note = f"Converted to {target} using user-supplied rates."

    return CostSummary(
        quoted_total=quoted,
        calculated_totals=calculated,
        estimated_extras=estimates,
        unknown_items=unknown,
        total_mismatch=mismatch,
        converted_total=converted,
        converted_note=note,
        is_complete=not unknown and converted is not None,
    )


# --- Side-by-side -------------------------------------------------------------


class Cell(BaseModel):
    status: CellStatus
    name: str | None = None
    amount: Money | None = None
    page: int | None = None
    note: str | None = None

    @property
    def priced(self) -> bool:
        return self.status in PRICED_STATUSES and self.amount is not None


class Row(BaseModel):
    key: str
    label: str
    a: Cell
    b: Cell
    flag: Flag


class Comparison(BaseModel):
    rows: list[Row]
    context_warnings: list[str]  # route / equipment / mode / incoterm differences
    scope_mismatch: bool = False  # route, equipment or mode differ: not like-for-like

    def gaps(self) -> list[Row]:
        return [r for r in self.rows if r.flag in ("gap_a", "gap_b")]


def _cells(quote: FreightQuote) -> dict[str, Cell]:
    cells: dict[str, Cell] = {}
    for c in quote.charges:
        key = canonical_key(c)
        amount = Money(amount=quantize(c.line_total), currency=c.currency) if c.line_total is not None and c.currency else None
        cell = Cell(status=c.status, name=c.name, amount=amount, page=c.evidence.page)
        if key in cells:  # two lines mapping to one key: add amounts if both priced in same currency
            prev = cells[key]
            if prev.priced and cell.priced and prev.amount.currency == amount.currency and prev.status == cell.status:
                cell = prev.model_copy(update={"amount": Money(amount=prev.amount.amount + amount.amount, currency=amount.currency), "name": f"{prev.name} + {c.name}"})
            elif prev.priced:
                continue
        cells[key] = cell
    return cells


def _lookup(cells: dict[str, Cell], key: str) -> Cell:
    if key in cells:
        return cells[key]
    for umbrella, children in UMBRELLAS.items():
        if key in children and umbrella in cells:
            u = cells[umbrella]
            return Cell(status=u.status, name=u.name, page=u.page, note=f"covered by '{u.name}'")
    return Cell(status="absent")


def _differs(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    return norm(a) not in norm(b) and norm(b) not in norm(a)


def compare(a: FreightQuote, b: FreightQuote) -> Comparison:
    ca, cb = _cells(a), _cells(b)
    keys = list(dict.fromkeys([*ca, *cb]))
    # Umbrella rows are only interesting if nothing else represents them.
    covered = {k for u, ch in UMBRELLAS.items() for k in ch if k in keys}
    rows = []
    for key in keys:
        if key in UMBRELLAS and UMBRELLAS[key] & covered:
            continue
        cell_a, cell_b = _lookup(ca, key), _lookup(cb, key)
        if cell_a.priced and cell_b.priced:
            flag = "priced_both"
        elif cell_b.priced:
            flag = "gap_a"
        elif cell_a.priced:
            flag = "gap_b"
        else:
            flag = "unknown_both"
        label = LABELS.get(key) or (cell_a.name or cell_b.name or key)
        rows.append(Row(key=key, label=label, a=cell_a, b=cell_b, flag=flag))

    warnings = []
    for field, label in [("origin", "Origin"), ("destination", "Destination"), ("equipment", "Equipment")]:
        if _differs(getattr(a, field), getattr(b, field)):
            warnings.append(f"{label} differs: '{getattr(a, field)}' vs '{getattr(b, field)}'.")
    if a.mode != b.mode and "unknown" not in (a.mode, b.mode):
        warnings.append(f"Transport mode differs: {a.mode} vs {b.mode}.")
    scope_mismatch = bool(warnings)
    if a.incoterm and b.incoterm and a.incoterm.upper() != b.incoterm.upper():
        warnings.append(f"Incoterms differ ({a.incoterm} vs {b.incoterm}); scope of charges may differ.")
    return Comparison(rows=rows, context_warnings=warnings, scope_mismatch=scope_mismatch)
