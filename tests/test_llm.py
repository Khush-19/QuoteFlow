"""LLM extraction tests.

Unit tests use a fake client whose replies are the hand-labelled ground truth JSON
(not recorded model output). The `live` test calls the real API: pytest -m live
"""

import json
import os
from types import SimpleNamespace

import httpx2
import openai
import pytest
from dotenv import load_dotenv

from quoteflow.llm import ExtractionError, LLMConfig, LLMConfigError, extract_quote, json_payload
from quoteflow.pdf_text import extract_text

from .conftest import SAMPLE_NAMES, load_doc, load_truth


def _bad_request(msg="unsupported response_format"):
    return openai.BadRequestError(msg, response=httpx2.Response(400, request=httpx2.Request("POST", "http://x")), body=None)


class FakeClient:
    """Mimics client.chat.completions.create; replies are strings or exceptions, in order."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


def _config(fmt="json_schema"):
    return LLMConfig(api_key="test", model="test-model", response_format=fmt)


def _truth_text(name, mutate=None):
    data = json.loads(load_truth(name).model_dump_json())
    if mutate:
        mutate(data)
    return json.dumps(data)


@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_extract_valid_response(name):
    client = FakeClient(_truth_text(name))
    result = extract_quote(load_doc(name), _config(), client)
    assert result.source == "llm" and result.model == "test-model"
    assert result.quote == load_truth(name)
    assert all(c.status == "verified" for c in result.evidence_checks)
    assert result.warnings == []
    sent = client.calls[0]["response_format"]
    assert sent["type"] == "json_schema" and sent["json_schema"]["name"] == "FreightQuote"


def test_think_block_and_fences_stripped():
    raw = "<think>let me see {not json}</think>\n```json\n" + _truth_text("quote_a_blueharbor") + "\n```"
    result = extract_quote(load_doc("quote_a_blueharbor"), _config(), FakeClient(raw))
    assert result.quote.quoted_total is not None


def test_retry_after_validation_error():
    client = FakeClient('{"charges": "oops"}', _truth_text("quote_a_blueharbor"))
    result = extract_quote(load_doc("quote_a_blueharbor"), _config(), client)
    assert len(client.calls) == 2
    assert "failed validation" in client.calls[1]["messages"][-1]["content"]
    assert result.quote == load_truth("quote_a_blueharbor")


def test_two_validation_failures_raise():
    with pytest.raises(ExtractionError, match="failed validation twice"):
        extract_quote(load_doc("quote_a_blueharbor"), _config(), FakeClient("nope", "{}"))


def test_falls_back_when_format_rejected():
    config = _config()
    client = FakeClient(_bad_request(), _truth_text("quote_a_blueharbor"))
    extract_quote(load_doc("quote_a_blueharbor"), config, client)
    assert client.calls[1]["response_format"]["json_schema"].get("title") == "FreightQuote"  # raw schema
    assert config.response_format == "json_schema_raw"


def test_all_formats_rejected_raises():
    client = FakeClient(_bad_request(), _bad_request(), _bad_request())
    with pytest.raises(ExtractionError, match="rejected every"):
        extract_quote(load_doc("quote_a_blueharbor"), _config(), client)


def test_fabricated_charge_is_flagged():
    def add_fake(d):
        d["charges"].append(
            {
                "name": "Fuel surcharge", "category": "surcharge", "amount": "99", "currency": "USD",
                "basis": None, "quantity": None, "status": "included",
                "evidence": {"page": 1, "quote_text": "Fuel surcharge USD 99.00"},
            }
        )

    result = extract_quote(load_doc("quote_a_blueharbor"), _config(), FakeClient(_truth_text("quote_a_blueharbor", add_fake)))
    assert any("Fuel surcharge" in w and "not found" in w for w in result.warnings)


def test_scanned_pdf_rejected_before_calling_api():
    import pymupdf

    doc = pymupdf.open()
    doc.new_page()
    client = FakeClient()
    with pytest.raises(ExtractionError, match="no extractable text"):
        extract_quote(extract_text(doc.tobytes(), "scan.pdf"), _config(), client)
    assert client.calls == []


def test_config_requires_key_and_model(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.setenv("NEBIUS_MODEL", "m")
    with pytest.raises(LLMConfigError, match="NEBIUS_API_KEY"):
        LLMConfig.from_env()
    monkeypatch.setenv("NEBIUS_API_KEY", "k")
    monkeypatch.delenv("NEBIUS_MODEL")
    with pytest.raises(LLMConfigError, match="NEBIUS_MODEL"):
        LLMConfig.from_env()


def test_json_payload_without_object_raises():
    with pytest.raises(ValueError):
        json_payload("I cannot help with that.")


@pytest.mark.live
@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_live_extraction_matches_ground_truth(name):
    load_dotenv()
    if not (os.getenv("NEBIUS_API_KEY") and os.getenv("NEBIUS_MODEL")):
        pytest.skip("NEBIUS_API_KEY / NEBIUS_MODEL not set")
    result = extract_quote(load_doc(name), LLMConfig.from_env())
    truth = load_truth(name)
    print(result.model_dump_json(indent=2))
    assert result.quote.quoted_total == truth.quoted_total
    priced = lambda q: sorted(c.amount for c in q.charges if c.amount is not None and c.status == "included")
    assert priced(result.quote) == priced(truth)
    assert not any(c.status == "not_found" for c in result.evidence_checks)
