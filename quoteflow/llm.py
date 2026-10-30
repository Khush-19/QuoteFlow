"""Nemotron via Nebius Token Factory (OpenAI-compatible API).

The model is only used to (1) turn PDF text into a FreightQuote and (2) phrase an
explanation of facts computed elsewhere. It never does arithmetic we rely on.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from openai import BadRequestError, OpenAI
from pydantic import ValidationError

from .evidence import check_quote
from .pdf_text import DocumentText
from .schema import ExtractedQuote, FreightQuote

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
# Request shapes for structured output, in the order we try them. Nebius docs show
# both the OpenAI-style wrapper and a raw schema; json_object is the last resort.
RESPONSE_FORMATS = ("json_schema", "json_schema_raw", "json_object")


class LLMConfigError(RuntimeError):
    pass


class ExtractionError(RuntimeError):
    pass


@dataclass
class LLMConfig:
    api_key: str
    model: str
    base_url: str = DEFAULT_BASE_URL
    response_format: str = "json_schema"

    @classmethod
    def from_env(cls) -> "LLMConfig":
        api_key = os.getenv("NEBIUS_API_KEY")
        model = os.getenv("NEBIUS_MODEL")
        if not api_key:
            raise LLMConfigError("NEBIUS_API_KEY is not set.")
        if not model:
            raise LLMConfigError("NEBIUS_MODEL is not set. Run scripts/check_nebius.py to pick one.")
        fmt = os.getenv("NEBIUS_RESPONSE_FORMAT", "json_schema")
        if fmt not in RESPONSE_FORMATS:
            raise LLMConfigError(f"NEBIUS_RESPONSE_FORMAT must be one of {RESPONSE_FORMATS}")
        return cls(api_key, model, os.getenv("NEBIUS_BASE_URL") or DEFAULT_BASE_URL, fmt)

    def client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=120)


EXTRACTION_PROMPT = """You extract data from international freight quotations.
Return ONLY a JSON object matching this JSON schema:
{schema}

Rules:
- Use only information in the document. Never invent charges, amounts, currencies or dates.
- If an amount is not stated, set amount to null. Never use 0 for an unknown amount.
- status: "included" = priced and part of the offer; "excluded" = the quote says it is not included;
  "at_cost" = will be charged at cost/actuals with no fixed price; "estimated" = the quote calls the
  amount an estimate; "not_mentioned" = only use if the document explicitly says it may apply without detail.
- Create a charge entry for each excluded or at-cost item too, with amount null.
- evidence.quote_text must be copied verbatim from the document (a short exact snippet that includes the
  amount when there is one). evidence.page is the number from the [page N] marker.
- quoted_total is the grand total exactly as printed; null if the document has no total.
- currency must be an ISO 4217 code (USD, EUR, CNY, ...).
- Put anything ambiguous in extraction_warnings.
"""


def _response_format(kind: str, schema: dict, name: str) -> dict:
    if kind == "json_schema":
        return {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}
    if kind == "json_schema_raw":
        return {"type": "json_schema", "json_schema": schema}
    return {"type": "json_object"}


def json_payload(content: str) -> str:
    """Strip reasoning blocks / code fences some models wrap around JSON."""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end < start:
        raise ValueError("no JSON object in model response")
    return content[start : end + 1]


def chat_json(client: OpenAI, config: LLMConfig, messages: list[dict], schema: dict, name: str) -> tuple[str, str]:
    """One completion with structured output. Falls back through request shapes the
    API rejects. Returns (content, format_used)."""
    formats = RESPONSE_FORMATS[RESPONSE_FORMATS.index(config.response_format) :]
    last_error: Exception | None = None
    for kind in formats:
        try:
            resp = client.chat.completions.create(
                model=config.model,
                messages=messages,
                response_format=_response_format(kind, schema, name),
                temperature=0,
                max_tokens=6000,
            )
        except BadRequestError as e:
            last_error = e
            continue
        config.response_format = kind  # remember what worked for later calls
        return resp.choices[0].message.content or "", kind
    raise ExtractionError(f"API rejected every structured-output format: {last_error}")


def extract_quote(doc: DocumentText, config: LLMConfig, client: OpenAI | None = None) -> ExtractedQuote:
    if not doc.has_text:
        raise ExtractionError(f"{doc.filename}: no extractable text (scanned PDF?). OCR is not supported.")
    client = client or config.client()
    schema = FreightQuote.model_json_schema()
    messages = [
        {"role": "system", "content": EXTRACTION_PROMPT.format(schema=json.dumps(schema))},
        {"role": "user", "content": f"Document: {doc.filename}\n\n{doc.to_prompt_text()}"},
    ]
    quote: FreightQuote | None = None
    error = ""
    for _ in range(2):  # one retry with the validation error fed back
        content, _fmt = chat_json(client, config, messages, schema, "FreightQuote")
        try:
            quote = FreightQuote.model_validate_json(json_payload(content))
            break
        except (ValidationError, ValueError) as e:
            error = str(e)
            messages += [
                {"role": "assistant", "content": content},
                {"role": "user", "content": f"That JSON failed validation:\n{error}\nReturn the corrected JSON only."},
            ]
    if quote is None:
        raise ExtractionError(f"{doc.filename}: model output failed validation twice: {error[:500]}")

    checks = check_quote(quote, doc)
    warnings = list(doc.warnings)
    for c in checks:
        if c.status == "not_found":
            warnings.append(f"Evidence for {c.label} was not found in the PDF; treat as unverified.")
        elif c.amount_in_snippet is False:
            warnings.append(f"Amount for {c.label} does not appear in its cited snippet; check it.")
        elif c.status == "wrong_page":
            warnings.append(f"Evidence for {c.label} was cited on the wrong page (found on page {c.found_on_page}).")
    warnings += quote.extraction_warnings
    return ExtractedQuote(
        source_file=doc.filename,
        quote=quote,
        evidence_checks=checks,
        source="llm",
        model=config.model,
        warnings=warnings,
    )

