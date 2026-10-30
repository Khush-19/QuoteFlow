"""Data models.

`FreightQuote` (and what it contains) is what the LLM must return. Everything else
is computed by deterministic Python and never produced by the model.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import BaseModel, Field, field_validator

ChargeStatus = Literal["included", "excluded", "not_mentioned", "estimated", "at_cost"]
ChargeCategory = Literal[
    "origin", "freight", "destination", "customs", "insurance", "surcharge", "documentation", "other"
]
TransportMode = Literal["sea_fcl", "sea_lcl", "air", "road", "unknown"]


def _parse_decimal(value):
    """Accept numbers or strings like '1,850.00' / 'USD 1850'. Empty -> None."""
    if value is None or isinstance(value, (int, float, Decimal)):
        return value
    if isinstance(value, str):
        cleaned = re.sub(r"[^\d.\-]", "", value.replace(",", ""))
        if not cleaned:
            return None
        try:
            return Decimal(cleaned)
        except InvalidOperation:
            raise ValueError(f"not a number: {value!r}")
    return value


def _normalize_currency(value):
    if value is None:
        return None
    v = str(value).strip().upper()
    return v or None


# --- LLM extraction output -------------------------------------------------


class Evidence(BaseModel):
    page: int = Field(ge=1, description="1-based page number where the text appears")
    quote_text: str = Field(min_length=1, description="Verbatim snippet copied from the document")


class Charge(BaseModel):
    name: str = Field(description="Charge name as written in the quote, e.g. 'THC Origin', 'BAF'")
    category: ChargeCategory
    amount: Decimal | None = Field(
        description="Amount as written for this charge. null if not stated. Never 0 unless the quote says 0."
    )
    currency: str | None = Field(description="ISO 4217 code, e.g. USD, EUR. null if not stated.")
    basis: str | None = Field(default=None, description="e.g. 'per container', 'per shipment', 'per kg'")
    quantity: Decimal | None = Field(
        default=None, description="Units the amount applies to, if stated (e.g. 2 containers). null = 1."
    )
    status: ChargeStatus
    evidence: Evidence

    _amount = field_validator("amount", "quantity", mode="before")(_parse_decimal)
    _currency = field_validator("currency", mode="before")(_normalize_currency)

    @property
    def line_total(self) -> Decimal | None:
        if self.amount is None:
            return None
        return self.amount * (self.quantity if self.quantity is not None else Decimal(1))


class FreightQuote(BaseModel):
    carrier_or_forwarder: str | None
    quote_ref: str | None
    origin: str | None
    destination: str | None
    mode: TransportMode
    incoterm: str | None
    equipment: str | None = Field(description="e.g. '1x40HC', '2x20GP', '850 kg / 4.2 cbm'")
    validity_until: date | None
    transit_days: int | None
    charges: list[Charge]
    quoted_total: Decimal | None = Field(description="Grand total exactly as stated in the quote, else null")
    quoted_total_currency: str | None
    quoted_total_evidence: Evidence | None
    exclusions: list[Evidence] = Field(description="Statements of what is excluded / not included")
    extraction_warnings: list[str] = Field(
        default_factory=list, description="Anything ambiguous or unreadable in the document"
    )

    _total = field_validator("quoted_total", mode="before")(_parse_decimal)
    _currency = field_validator("quoted_total_currency", mode="before")(_normalize_currency)


# --- Computed by Python ------------------------------------------------------

EvidenceStatus = Literal["verified", "wrong_page", "not_found"]


class EvidenceCheck(BaseModel):
    label: str  # e.g. "charge: BAF" or "quoted total"
    status: EvidenceStatus
    found_on_page: int | None = None
    amount_in_snippet: bool | None = None  # None when there is no amount to check


class ExtractedQuote(BaseModel):
    """A quote plus provenance: which file, which model, and how well evidence checked out."""

    source_file: str
    quote: FreightQuote
    evidence_checks: list[EvidenceCheck]
    source: Literal["llm", "ground_truth_fixture", "user_edited"]
    model: str | None = None
    warnings: list[str] = Field(default_factory=list)


class Money(BaseModel):
    amount: Decimal
    currency: str


class UnknownItem(BaseModel):
    name: str
    reason: Literal["excluded", "at_cost", "not_mentioned", "amount_missing", "currency_missing"]


class CostSummary(BaseModel):
    quoted_total: Money | None  # what the document says
    calculated_totals: list[Money]  # sum of priced included charges, one per currency
    estimated_extras: list[Money]  # charges the quote itself marks as estimates
    unknown_items: list[UnknownItem]
    total_mismatch: Money | None  # quoted - calculated, when both exist in the same currency
    converted_total: Money | None  # calculated + estimates in target currency, if rates allow
    converted_note: str | None
    is_complete: bool  # no unknowns and nothing left unconverted
