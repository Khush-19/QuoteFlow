"""M0 smoke test: verify Nebius Token Factory access and Nemotron structured output.

Usage:
    .venv/bin/python scripts/check_nebius.py            # list Nemotron models, test NEBIUS_MODEL or first match
    .venv/bin/python scripts/check_nebius.py --model ID # test a specific model

Prints only what the live API returns. Exits non-zero on any failure.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quoteflow.llm import RESPONSE_FORMATS, _response_format  # noqa: E402

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

SAMPLE_TEXT = """[page 1]
ACME Logistics - Quotation Q-1001
Shanghai -> Rotterdam, 1x40HC, FOB
Ocean freight: USD 1,850.00 per container
THC origin: USD 120.00
Destination charges: at cost
Duties and taxes not included.
"""


class ProbeCharge(BaseModel):
    name: str
    amount: float | None
    currency: str | None
    status: str
    evidence: str


class ProbeQuote(BaseModel):
    forwarder: str | None
    charges: list[ProbeCharge]
    exclusions: list[str]


def list_nemotron_models(client: OpenAI) -> list[str]:
    ids = sorted(m.id for m in client.models.list())
    print(f"Models visible to this key: {len(ids)}")
    matches = [i for i in ids if "nemotron" in i.lower()]
    print("Nemotron models:")
    for i in matches:
        print(f"  - {i}")
    if not matches:
        print("  (none)")
    return matches


def probe(client: OpenAI, model: str, mode: str) -> bool:
    response_format = _response_format(mode, ProbeQuote.model_json_schema(), "ProbeQuote")

    system = (
        "Extract freight quote charges as JSON matching this schema: "
        + json.dumps(ProbeQuote.model_json_schema())
        + " Use null for unknown amounts. Never invent values. "
        "status is one of included, excluded, at_cost. evidence is a verbatim snippet."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": SAMPLE_TEXT}],
            response_format=response_format,
            temperature=0,
            max_tokens=1500,
        )
    except Exception as e:  # report the real API error verbatim
        print(f"[{mode}] request failed: {type(e).__name__}: {e}")
        return False

    content = resp.choices[0].message.content or ""
    print(f"[{mode}] usage: {resp.usage}")
    print(f"[{mode}] raw content:\n{content}\n")
    try:
        parsed = ProbeQuote.model_validate_json(content)
    except ValidationError as e:
        print(f"[{mode}] Pydantic validation FAILED:\n{e}")
        return False
    print(f"[{mode}] Pydantic validation OK: {len(parsed.charges)} charges, {len(parsed.exclusions)} exclusions")
    return True


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("NEBIUS_MODEL") or None)
    args = parser.parse_args()

    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        print("NEBIUS_API_KEY is not set (export it or put it in .env).")
        return 2
    base_url = os.getenv("NEBIUS_BASE_URL") or DEFAULT_BASE_URL
    print(f"Base URL: {base_url}")
    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        matches = list_nemotron_models(client)
    except Exception as e:
        print(f"Listing models failed: {type(e).__name__}: {e}")
        return 1

    model = args.model or (matches[0] if matches else None)
    if not model:
        print("No model to test. Pass --model.")
        return 1
    print(f"\nTesting model: {model}\n")

    results = {mode: probe(client, model, mode) for mode in RESPONSE_FORMATS}

    print("\nSummary:")
    print(f"  model: {model}")
    for mode, ok in results.items():
        print(f"  {mode:16s} {'OK' if ok else 'FAILED'}")
    working = [m for m, ok in results.items() if ok]
    if working:
        print(f"\nSuggested .env:\n  NEBIUS_MODEL={model}\n  NEBIUS_RESPONSE_FORMAT={working[0]}")
    return 0 if working else 1


if __name__ == "__main__":
    sys.exit(main())
