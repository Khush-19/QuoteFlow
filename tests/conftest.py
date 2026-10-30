import json
from pathlib import Path

import pytest

from quoteflow.pdf_text import extract_text
from quoteflow.schema import FreightQuote

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
SAMPLE_NAMES = ["quote_a_blueharbor", "quote_b_northwind"]


def load_truth(name: str) -> FreightQuote:
    return FreightQuote.model_validate_json((SAMPLES / "ground_truth" / f"{name}.json").read_text())


def load_doc(name: str):
    path = SAMPLES / f"{name}.pdf"
    return extract_text(path.read_bytes(), path.name)


@pytest.fixture
def quote_a() -> FreightQuote:
    return load_truth("quote_a_blueharbor")


@pytest.fixture
def quote_b() -> FreightQuote:
    return load_truth("quote_b_northwind")


@pytest.fixture
def truth_json():
    return lambda name: json.loads((SAMPLES / "ground_truth" / f"{name}.json").read_text())
