"""Rule-based recommendation, plus an optional LLM-phrased explanation of it.

The verdict and every number come from here. The LLM only rewrites the facts as prose,
and its text is discarded if it contains a number that is not in the facts.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

from .compare import Comparison, Row, compare, convert, summarize
from .evidence import is_trusted, numbers_in
from .llm import LLMConfig, chat_json, json_payload
from .schema import CostSummary, ExtractedQuote, Money

TOO_CLOSE_PCT = Decimal("0.03")


def fmt(m: Money | None) -> str:
    return "unknown" if m is None else f"{m.currency} {m.amount:,.2f}"


def mismatch_text(diff: Money) -> str:
    """diff = stated total - sum of priced lines."""
    direction = "lower" if diff.amount < 0 else "higher"
    return f"Stated total is {fmt(Money(amount=abs(diff.amount), currency=diff.currency))} {direction} than the sum of its priced lines."


class FillIn(BaseModel):
    label: str
    amount: Money
    borrowed_from: Literal["a", "b"]
    source_page: int | None


class Recommendation(BaseModel):
    verdict: Literal["a", "b", "too_close", "cannot_compare"]
    headline: str
    confidence: Literal["high", "medium", "low"]
    currency: str | None
    known_a: Money | None  # priced + estimated, per the quote itself
    known_b: Money | None
    adjusted_a: Money | None  # known + the other quote's price for each gap
    adjusted_b: Money | None
    fill_ins_a: list[FillIn]
    fill_ins_b: list[FillIn]
    unknown_in_both: list[str]
    reasons: list[str]
    caveats: list[str]


class Analysis(BaseModel):
    """Everything the UI needs, computed in one place."""

    summary_a: CostSummary
    summary_b: CostSummary
    comparison: Comparison
    recommendation: Recommendation


def _fill_ins(rows: list[Row], side: Literal["a", "b"], target: str, rates: dict) -> tuple[list[FillIn], list[str]]:
    gap_flag, other = ("gap_a", "b") if side == "a" else ("gap_b", "a")
    out, missing = [], []
    for r in rows:
        if r.flag != gap_flag:
            continue
        cell = getattr(r, other)
        money, miss = convert([cell.amount], target, rates)
        if money is None:
            missing += miss
            continue
        out.append(FillIn(label=r.label, amount=money, borrowed_from=other, source_page=cell.page))
    return out, missing


def _plus(base: Money, items: list[FillIn]) -> Money:
    return Money(amount=base.amount + sum((f.amount.amount for f in items), Decimal(0)), currency=base.currency)


def recommend(
    a: ExtractedQuote,
    b: ExtractedQuote,
    target: str | None = None,
    rates: dict[str, Decimal] | None = None,
    today: date | None = None,
) -> Analysis:
    rates = rates or {}
    today = today or date.today()
    qa, qb = a.quote, b.quote
    target = target or qa.quoted_total_currency or qb.quoted_total_currency
    if target is None:
        currencies = {c.currency for c in qa.charges + qb.charges if c.currency}
        target = sorted(currencies)[0] if currencies else None
    sa, sb = summarize(qa, target, rates), summarize(qb, target, rates)
    comp = compare(qa, qb)
    name_a = qa.carrier_or_forwarder or "Quote A"
    name_b = qb.carrier_or_forwarder or "Quote B"

    reasons: list[str] = []
    caveats: list[str] = list(comp.context_warnings)

    for name, s in ((name_a, sa), (name_b, sb)):
        if s.total_mismatch is not None:
            calc = s.calculated_totals[0]
            caveats.append(
                f"{name}: stated total {fmt(s.quoted_total)} does not equal its priced lines {fmt(calc)}. "
                f"{mismatch_text(s.total_mismatch)} Ask the forwarder to confirm."
            )
        if s.estimated_extras:
            caveats.append(f"{name} includes estimates ({', '.join(fmt(m) for m in s.estimated_extras)}), not fixed prices.")

    untrusted = [(q.source_file, c.label) for q in (a, b) for c in q.evidence_checks if not is_trusted(c)]
    if untrusted:
        caveats.append(f"{len(untrusted)} extracted item(s) could not be verified against the PDF text.")

    for q, name in ((qa, name_a), (qb, name_b)):
        if q.validity_until and q.validity_until < today:
            caveats.append(f"{name} expired on {q.validity_until.isoformat()}.")

    unknown_both = [r.label for r in comp.rows if r.flag == "unknown_both"]
    fills_a, miss_a = _fill_ins(comp.rows, "a", target, rates) if target else ([], [])
    fills_b, miss_b = _fill_ins(comp.rows, "b", target, rates) if target else ([], [])

    def result(verdict, headline, confidence, adjusted_a=None, adjusted_b=None):
        return Analysis(
            summary_a=sa, summary_b=sb, comparison=comp,
            recommendation=Recommendation(
                verdict=verdict, headline=headline, confidence=confidence, currency=target,
                known_a=sa.converted_total, known_b=sb.converted_total, adjusted_a=adjusted_a, adjusted_b=adjusted_b,
                fill_ins_a=fills_a, fill_ins_b=fills_b, unknown_in_both=unknown_both, reasons=reasons, caveats=caveats,
            ),
        )

    missing = sorted(set(miss_a + miss_b))
    if sa.converted_total is None or sb.converted_total is None or missing:
        notes = {n for n in (sa.converted_note, sb.converted_note) if n}
        if missing:
            notes.add(f"No exchange rate given for {', '.join(missing)} -> {target}.")
        reasons += sorted(notes) or ["One of the quotes has no priced charges."]
        return result("cannot_compare", "Cannot compare totals yet: add the missing exchange rate(s).", "low")

    adj_a, adj_b = _plus(sa.converted_total, fills_a), _plus(sb.converted_total, fills_b)

    if sa.quoted_total and sb.quoted_total and sa.quoted_total.currency == sb.quoted_total.currency:
        lower = name_a if sa.quoted_total.amount < sb.quoted_total.amount else name_b
        reasons.append(
            f"Headline totals: {name_a} {fmt(sa.quoted_total)} vs {name_b} {fmt(sb.quoted_total)}; "
            f"{lower} looks cheaper at first glance."
        )
    for fills, name, other in ((fills_a, name_a, name_b), (fills_b, name_b, name_a)):
        if fills:
            items = ", ".join(f"{f.label} ({fmt(f.amount)})" for f in fills)
            reasons.append(f"{name} does not price {items}; {other} does.")
    reasons.append(
        f"Like-for-like estimate (priced lines + estimates, with each quote's gaps filled at the other quote's price): "
        f"{name_a} {fmt(adj_a)} vs {name_b} {fmt(adj_b)}."
    )
    if unknown_both:
        reasons.append(f"Neither quote prices: {', '.join(unknown_both)}. These remain open costs for both.")
    if qa.transit_days and qb.transit_days and qa.transit_days != qb.transit_days:
        faster = name_a if qa.transit_days < qb.transit_days else name_b
        reasons.append(f"Transit: {name_a} {qa.transit_days} days vs {name_b} {qb.transit_days} days ({faster} is faster).")

    if comp.scope_mismatch or untrusted:
        confidence = "low"
    elif fills_a or fills_b or unknown_both or sa.total_mismatch or sb.total_mismatch:
        confidence = "medium"
    else:
        confidence = "high"

    diff = adj_a.amount - adj_b.amount
    if abs(diff) <= TOO_CLOSE_PCT * min(adj_a.amount, adj_b.amount):
        return result(
            "too_close",
            f"Too close to call on cost (difference {fmt(Money(amount=abs(diff), currency=target))}); decide on service, transit and terms.",
            "low" if confidence == "low" else "medium",
            adj_a, adj_b,
        )
    verdict, winner = ("a", name_a) if diff < 0 else ("b", name_b)
    saving = Money(amount=abs(diff), currency=target)
    return result(verdict, f"{winner} is likely cheaper by about {fmt(saving)} on a like-for-like basis.", confidence, adj_a, adj_b)


# --- LLM explanation ---------------------------------------------------------

EXPLAIN_PROMPT = """You explain a freight quote comparison to a logistics buyer.
Write 3 to 5 plain sentences based ONLY on the facts JSON the user sends.
Do not add numbers, charges, or claims that are not in the facts. Keep the verdict and confidence as given.
Return JSON: {"explanation": "<text>"}"""


class Explanation(BaseModel):
    explanation: str


def explain(rec: Recommendation, config: LLMConfig, client: OpenAI | None = None) -> tuple[str | None, str | None]:
    """Returns (text, problem). text is None if the model failed or introduced unsupported numbers."""
    client = client or config.client()
    facts = rec.model_dump_json()
    messages = [{"role": "system", "content": EXPLAIN_PROMPT}, {"role": "user", "content": facts}]
    try:
        content, _ = chat_json(client, config, messages, Explanation.model_json_schema(), "Explanation")
        text = Explanation.model_validate_json(json_payload(content)).explanation.strip()
    except Exception as e:
        return None, f"Explanation unavailable: {e}"
    allowed = numbers_in(facts)
    unsupported = sorted(str(n) for n in numbers_in(text) - allowed if n > 5)
    if unsupported:
        return None, f"Model explanation discarded: it mentioned numbers not in the facts ({', '.join(unsupported)})."
    return text, None
