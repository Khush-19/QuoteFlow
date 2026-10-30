from decimal import Decimal

import pymupdf
import pytest

from quoteflow.evidence import check_evidence, check_quote, is_trusted, numbers_in
from quoteflow.pdf_text import PdfReadError, extract_text
from quoteflow.schema import Evidence

from .conftest import SAMPLE_NAMES, load_doc, load_truth


def _pdf(*page_texts: str) -> bytes:
    doc = pymupdf.open()
    for t in page_texts:
        page = doc.new_page()
        if t:
            page.insert_text((72, 72), t)
    return doc.tobytes()


def test_sample_text_extracted_with_pages():
    doc = load_doc("quote_a_blueharbor")
    assert doc.total_pages == 1
    assert doc.pages[0].page == 1
    assert "USD 2,455.00" in doc.pages[0].text
    assert doc.warnings == []


def test_ligatures_folded():
    doc = load_doc("quote_b_northwind")
    assert "offer" in doc.pages[0].text  # source PDF contains the "ﬀ" ligature


def test_blank_page_warns_and_has_no_text():
    doc = extract_text(_pdf(""), "scan.pdf")
    assert not doc.has_text
    assert "no extractable text" in doc.warnings[0]


def test_prompt_text_has_page_markers():
    doc = extract_text(_pdf("first page text here ok", "second page text here ok"), "two.pdf")
    prompt = doc.to_prompt_text()
    assert "[page 1]" in prompt and "[page 2]" in prompt


def test_not_a_pdf_raises():
    with pytest.raises(PdfReadError):
        extract_text(b"hello, not a pdf", "x.pdf")


@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_all_ground_truth_evidence_verifies(name):
    checks = check_quote(load_truth(name), load_doc(name))
    bad = [c for c in checks if c.status != "verified" or c.amount_in_snippet is False]
    assert bad == []


def test_fabricated_snippet_not_found():
    doc = load_doc("quote_a_blueharbor")
    check = check_evidence("x", Evidence(page=1, quote_text="Fuel surcharge USD 99.00"), doc)
    assert check.status == "not_found"
    assert not is_trusted(check)


def test_wrong_amount_flagged_even_if_snippet_real():
    doc = load_doc("quote_a_blueharbor")
    check = check_evidence("BAF", Evidence(page=1, quote_text="BAF (bunker adjustment) USD 310.00"), doc, Decimal("130"))
    assert check.status == "verified"
    assert check.amount_in_snippet is False
    assert not is_trusted(check)


def test_wrong_page_detected():
    doc = extract_text(_pdf("intro page with words", "Ocean freight USD 1,000.00"), "two.pdf")
    check = check_evidence("x", Evidence(page=1, quote_text="Ocean freight USD 1,000.00"), doc, Decimal(1000))
    assert check.status == "wrong_page" and check.found_on_page == 2


def test_numbers_in_handles_both_separators():
    assert Decimal("1850.00") in numbers_in("USD 1,850.00")
    assert Decimal("1850.00") in numbers_in("EUR 1.850,00")
