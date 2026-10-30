"""Check that LLM-cited evidence actually exists in the PDF text."""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation

from .pdf_text import DocumentText
from .schema import Evidence, EvidenceCheck, FreightQuote

_NUMBER = re.compile(r"\d[\d.,' ]*\d|\d")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[‐-―]", "-", text)  # dashes
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    return re.sub(r"\s+", " ", text).strip()


def numbers_in(text: str) -> set[Decimal]:
    """All numbers in text, reading both 1,850.00 and 1.850,00 styles."""
    found: set[Decimal] = set()
    for raw in _NUMBER.findall(text):
        raw = raw.replace(" ", "").replace("'", "")
        candidates = {raw.replace(",", "")}  # 1,850.00
        candidates.add(raw.replace(".", "").replace(",", "."))  # 1.850,00
        for c in candidates:
            try:
                found.add(Decimal(c))
            except InvalidOperation:
                pass
    return found


def check_evidence(label: str, evidence: Evidence, doc: DocumentText, amount: Decimal | None = None) -> EvidenceCheck:
    needle = normalize(evidence.quote_text)
    pages = {p.page: normalize(p.text) for p in doc.pages}
    amount_ok = None if amount is None else amount in numbers_in(evidence.quote_text)
    if needle and needle in pages.get(evidence.page, ""):
        return EvidenceCheck(label=label, status="verified", found_on_page=evidence.page, amount_in_snippet=amount_ok)
    for page_no, text in pages.items():
        if needle and needle in text:
            return EvidenceCheck(label=label, status="wrong_page", found_on_page=page_no, amount_in_snippet=amount_ok)
    return EvidenceCheck(label=label, status="not_found", amount_in_snippet=amount_ok)


def check_quote(quote: FreightQuote, doc: DocumentText) -> list[EvidenceCheck]:
    checks = [check_evidence(f"charge: {c.name}", c.evidence, doc, c.amount) for c in quote.charges]
    if quote.quoted_total_evidence is not None:
        checks.append(check_evidence("quoted total", quote.quoted_total_evidence, doc, quote.quoted_total))
    checks += [check_evidence(f"exclusion {i + 1}", e, doc) for i, e in enumerate(quote.exclusions)]
    return checks


def is_trusted(check: EvidenceCheck) -> bool:
    return check.status != "not_found" and check.amount_in_snippet is not False
