from decimal import Decimal

import pytest
from pydantic import ValidationError

from quoteflow.schema import Charge, FreightQuote

from .conftest import SAMPLE_NAMES, load_truth


def _charge(**overrides):
    base = {
        "name": "BAF",
        "category": "surcharge",
        "amount": 310,
        "currency": "usd",
        "status": "included",
        "evidence": {"page": 1, "quote_text": "BAF USD 310"},
    }
    return Charge.model_validate(base | overrides)


@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_ground_truth_validates(name):
    assert isinstance(load_truth(name), FreightQuote)


def test_missing_amount_stays_none_not_zero():
    c = _charge(amount=None, status="at_cost")
    assert c.amount is None
    assert c.line_total is None


def test_amount_string_with_separators_parsed_exactly():
    assert _charge(amount="USD 1,850.00").amount == Decimal("1850.00")


def test_empty_amount_string_is_none():
    assert _charge(amount="").amount is None


def test_currency_uppercased():
    assert _charge().currency == "USD"


def test_line_total_uses_quantity():
    assert _charge(amount=145, quantity=2).line_total == Decimal(290)


def test_invalid_status_rejected():
    with pytest.raises(ValidationError):
        _charge(status="maybe")


def test_evidence_page_must_be_positive():
    with pytest.raises(ValidationError):
        _charge(evidence={"page": 0, "quote_text": "x"})
