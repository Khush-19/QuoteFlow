from datetime import date
from decimal import Decimal

from quoteflow.llm import LLMConfig
from quoteflow.recommend import explain, recommend
from quoteflow.schema import EvidenceCheck, ExtractedQuote

from .test_compare import charge, quote
from .test_llm import FakeClient

TODAY = date(2026, 10, 30)


def eq(q, name="x.pdf", checks=()):
    return ExtractedQuote(source_file=name, quote=q, evidence_checks=list(checks), source="ground_truth_fixture")


def test_samples_verdict(quote_a, quote_b):
    rec = recommend(eq(quote_a), eq(quote_b), today=TODAY).recommendation
    assert rec.verdict == "b"
    assert rec.confidence == "medium"  # gaps were filled with borrowed prices
    assert rec.adjusted_a.amount == Decimal("3085.00")  # 2455 + DTHC 230 + delivery 400
    assert rec.adjusted_b.amount == Decimal("2945.00")  # 2150 + est 400 + BAF 310 + export customs 85
    assert {f.label for f in rec.fill_ins_b} == {"Bunker / fuel surcharge", "Export customs clearance"}
    assert any("does not equal its priced lines" in c for c in rec.caveats)
    assert "Import duties & taxes" in rec.unknown_in_both


def test_identical_complete_quotes_high_confidence_too_close():
    q = quote(charge("Ocean freight", 1000), total=Decimal(1000))
    rec = recommend(eq(q), eq(q), today=TODAY).recommendation
    assert rec.verdict == "too_close"
    assert rec.confidence == "medium"  # too_close is never "high"


def test_clear_winner_high_confidence():
    a = quote(charge("Ocean freight", 1000), total=Decimal(1000), carrier_or_forwarder="A")
    b = quote(charge("Ocean freight", 1500), total=Decimal(1500), carrier_or_forwarder="B")
    rec = recommend(eq(a), eq(b), today=TODAY).recommendation
    assert rec.verdict == "a" and rec.confidence == "high"


def test_cannot_compare_without_fx():
    a = quote(charge("Ocean freight", 1000, "USD"))
    b = quote(charge("Ocean freight", 900, "EUR"), total=Decimal(900), total_ccy="EUR")
    rec = recommend(eq(a), eq(b), target="USD", today=TODAY).recommendation
    assert rec.verdict == "cannot_compare"
    assert rec.adjusted_a is None and rec.adjusted_b is None


def test_fx_rate_enables_compare():
    a = quote(charge("Ocean freight", 1000, "USD"))
    b = quote(charge("Ocean freight", 800, "EUR"))
    rec = recommend(eq(a), eq(b), target="USD", rates={"EUR": Decimal("1.10")}, today=TODAY).recommendation
    assert rec.verdict == "b" and rec.known_b.amount == Decimal("880.00")


def test_unverified_evidence_lowers_confidence():
    a = quote(charge("Ocean freight", 1000))
    b = quote(charge("Ocean freight", 1500))
    bad = EvidenceCheck(label="charge: Ocean freight", status="not_found")
    rec = recommend(eq(a, checks=[bad]), eq(b), today=TODAY).recommendation
    assert rec.confidence == "low"
    assert any("could not be verified" in c for c in rec.caveats)


def test_scope_mismatch_lowers_confidence():
    a = quote(charge("Ocean freight", 1000), equipment="1x40HC")
    b = quote(charge("Ocean freight", 1500), equipment="2x20GP")
    assert recommend(eq(a), eq(b), today=TODAY).recommendation.confidence == "low"


def test_expired_quote_caveat():
    a = quote(charge("Ocean freight", 1000), validity_until=date(2026, 10, 1))
    b = quote(charge("Ocean freight", 1500))
    assert any("expired" in c for c in recommend(eq(a), eq(b), today=TODAY).recommendation.caveats)


# --- explanation guard ----------------------------------------------------------


def _rec(quote_a, quote_b):
    return recommend(eq(quote_a), eq(quote_b), today=TODAY).recommendation


def test_explanation_accepted_when_numbers_match(quote_a, quote_b):
    client = FakeClient('{"explanation": "Northwind is cheaper by about USD 140.00 once gaps are filled (USD 2,945.00 vs USD 3,085.00)."}')
    text, problem = explain(_rec(quote_a, quote_b), LLMConfig("k", "m"), client)
    assert problem is None and "140" in text


def test_explanation_with_invented_number_discarded(quote_a, quote_b):
    client = FakeClient('{"explanation": "Northwind saves USD 999.00."}')
    text, problem = explain(_rec(quote_a, quote_b), LLMConfig("k", "m"), client)
    assert text is None and "999" in problem


def test_explanation_api_failure_is_soft(quote_a, quote_b):
    client = FakeClient("not json at all")
    text, problem = explain(_rec(quote_a, quote_b), LLMConfig("k", "m"), client)
    assert text is None and problem.startswith("Explanation unavailable")
