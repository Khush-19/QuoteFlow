from decimal import Decimal

import pytest

from quoteflow.compare import canonical_key, compare, convert, summarize
from quoteflow.schema import Charge, FreightQuote, Money


def charge(name, amount, currency="USD", status="included", category="other", quantity=None):
    return Charge(
        name=name, category=category, amount=amount, currency=currency, status=status,
        quantity=quantity, evidence={"page": 1, "quote_text": name},
    )


def quote(*charges, total=None, total_ccy="USD", **kw):
    base = dict(
        carrier_or_forwarder="X", quote_ref=None, origin="Shanghai", destination="Rotterdam", mode="sea_fcl",
        incoterm="FOB", equipment="1x40HC", validity_until=None, transit_days=None, charges=list(charges),
        quoted_total=total, quoted_total_currency=total_ccy if total is not None else None,
        quoted_total_evidence=None, exclusions=[],
    )
    return FreightQuote(**(base | kw))


# --- summarize ----------------------------------------------------------------


def test_missing_fees_are_unknown_not_zero():
    s = summarize(quote(charge("Ocean freight", 1000), charge("Customs", None, None, "at_cost")))
    assert s.calculated_totals == [Money(amount=Decimal("1000.00"), currency="USD")]
    assert [u.reason for u in s.unknown_items] == ["at_cost"]
    assert s.is_complete is False


def test_included_but_unpriced_is_unknown():
    s = summarize(quote(charge("Ocean freight", 1000), charge("THC", None)))
    assert [u.reason for u in s.unknown_items] == ["amount_missing"]


def test_included_without_currency_is_unknown():
    s = summarize(quote(charge("THC", 120, currency=None)))
    assert s.calculated_totals == []
    assert [u.reason for u in s.unknown_items] == ["currency_missing"]


def test_estimates_kept_separate_from_calculated():
    s = summarize(quote(charge("Ocean freight", 1000), charge("Delivery", 400, status="estimated")))
    assert s.calculated_totals[0].amount == Decimal("1000.00")
    assert s.estimated_extras[0].amount == Decimal("400.00")
    assert s.converted_total.amount == Decimal("1400.00")


def test_quoted_vs_calculated_mismatch():
    s = summarize(quote(charge("A", 100), charge("B", 50), total=Decimal(120)))
    assert s.total_mismatch == Money(amount=Decimal("-30.00"), currency="USD")


def test_matching_total_has_no_mismatch():
    assert summarize(quote(charge("A", 100), total=Decimal(100))).total_mismatch is None


def test_quantity_multiplies():
    s = summarize(quote(charge("THC", 145, quantity=2)))
    assert s.calculated_totals[0].amount == Decimal("290.00")


def test_mixed_currency_not_summed_without_rate():
    s = summarize(quote(charge("Freight", 1000, "USD"), charge("DTHC", 200, "EUR")), target="USD")
    assert {m.currency for m in s.calculated_totals} == {"USD", "EUR"}
    assert s.converted_total is None
    assert "No exchange rate given for EUR" in s.converted_note


def test_mixed_currency_converted_with_user_rate():
    s = summarize(quote(charge("Freight", 1000, "USD"), charge("DTHC", 200, "EUR")), target="USD", rates={"EUR": Decimal("1.10")})
    assert s.converted_total == Money(amount=Decimal("1220.00"), currency="USD")
    assert "user-supplied" in s.converted_note


def test_no_float_drift():
    s = summarize(quote(*[charge(f"c{i}", Decimal("0.10")) for i in range(3)]))
    assert s.calculated_totals[0].amount == Decimal("0.30")


def test_convert_reports_missing():
    total, missing = convert([Money(amount=Decimal(1), currency="CNY")], "USD", {})
    assert total is None and missing == ["CNY"]


# --- samples ------------------------------------------------------------------


def test_sample_a_summary(quote_a):
    s = summarize(quote_a)
    assert s.quoted_total.amount == Decimal("2455.00")
    assert s.calculated_totals[0].amount == Decimal("2455.00")
    assert s.total_mismatch is None
    assert len(s.unknown_items) == 3


def test_sample_b_summary(quote_b):
    s = summarize(quote_b)
    assert s.calculated_totals[0].amount == Decimal("2150.00")
    assert s.total_mismatch.amount == Decimal("-50.00")
    assert s.estimated_extras[0].amount == Decimal("400.00")
    assert {u.name for u in s.unknown_items} == {"Import customs clearance", "Peak season surcharge"}


# --- compare ------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,key",
    [
        ("Origin THC", "origin_thc"),
        ("Terminal handling at origin", "origin_thc"),
        ("DTHC", "dest_thc"),
        ("BAF (bunker adjustment)", "baf"),
        ("Ocean freight", "main_freight"),
        ("Documentation fee (B/L)", "documentation"),
        ("Import duties and taxes", "duties_taxes"),
        ("On-carriage to consignee", "delivery"),
    ],
)
def test_canonical_keys(name, key):
    assert canonical_key(charge(name, 1)) == key


def test_unrecognised_charge_gets_category_slug():
    assert canonical_key(charge("Seal Fee", 10, category="origin")) == "origin:seal_fee"


def test_sample_gaps(quote_a, quote_b):
    rows = {r.key: r for r in compare(quote_a, quote_b).rows}
    assert rows["main_freight"].flag == "priced_both"
    assert rows["baf"].flag == "gap_b"  # A prices BAF, B never mentions it
    assert rows["export_customs"].flag == "gap_b"
    assert rows["dest_thc"].flag == "gap_a"  # B prices it, A excludes destination charges
    assert rows["dest_thc"].a.note == "covered by 'Destination charges'"
    assert rows["delivery"].flag == "gap_a"
    assert rows["import_customs"].flag == "unknown_both"
    assert "destination_charges" not in rows  # umbrella folded into specific rows


def test_context_warnings():
    a = quote(charge("x", 1), incoterm="FOB", equipment="1x40HC")
    b = quote(charge("x", 1), incoterm="EXW", equipment="2x20GP", destination="Hamburg")
    warnings = " ".join(compare(a, b).context_warnings)
    assert "Incoterms differ" in warnings and "Equipment differs" in warnings and "Destination differs" in warnings


def test_similar_locations_not_flagged():
    a, b = quote(charge("x", 1), origin="Shanghai, CN"), quote(charge("x", 1), origin="Shanghai")
    assert compare(a, b).context_warnings == []
