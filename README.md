# QuoteFlow Lite

Upload two international freight quotation PDFs. QuoteFlow extracts every charge with
NVIDIA Nemotron (via Nebius Token Factory), compares them with deterministic Python, flags
missing or excluded fees, and gives an evidence-backed recommendation.

**Design rules:** the LLM only extracts data and phrases explanations. Every total, gap and
verdict is computed in Python. Missing fees are never treated as zero. Every extracted charge
carries a verbatim snippet and page number, and those snippets are checked against the PDF text.

## Setup

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env          # then add NEBIUS_API_KEY
.venv/bin/python scripts/check_nebius.py   # lists Nemotron models, tests structured output
```

Copy the suggested `NEBIUS_MODEL` / `NEBIUS_RESPONSE_FORMAT` lines into `.env`.

## Run

```bash
.venv/bin/streamlit run app.py
```

Click **Load sample quotes** → **Extract & compare**. Without an API key, the sidebar's
*Offline demo* toggle loads hand-labelled ground truth for the bundled samples. That data
is labelled "not model output" everywhere in the UI.

## Tests

```bash
.venv/bin/pytest            # unit tests (LLM mocked)
.venv/bin/pytest -m live    # real Nebius calls against the sample PDFs
```

## Layout

| Path | Role |
|---|---|
| `quoteflow/schema.py` | Pydantic models (LLM output vs computed) |
| `quoteflow/pdf_text.py` | PyMuPDF text extraction, page-tagged |
| `quoteflow/evidence.py` | Checks snippets and amounts against PDF text |
| `quoteflow/llm.py` | Nebius client, structured extraction, retry, format fallback |
| `quoteflow/compare.py` | Totals, unknowns, currency handling, charge matching |
| `quoteflow/recommend.py` | Rule-based verdict and confidence; guarded LLM explanation |
| `app.py` | Streamlit UI |
| `samples/` | Synthetic PDFs (fictional companies) + generator + ground truth |

## How costs are reported

- **Stated total**: the grand total printed on the quote.
- **Sum of priced lines**: included charges with an amount and currency.
- **Estimates**: amounts the quote itself calls estimates, kept separate.
- **Unknown**: excluded, at cost, or missing amount/currency. Listed, never summed as 0.
- **Like-for-like estimate**: priced lines + estimates, with each quote's gaps filled at the
  other quote's price for the same charge. It is labelled as an estimate.
- **Currency**: no live FX. Totals in different currencies are only combined using rates you enter.

## Deploying (Streamlit Community Cloud)

Set `NEBIUS_API_KEY`, `NEBIUS_MODEL` and optionally `NEBIUS_RESPONSE_FORMAT` in the app's
secrets. `app.py` reads `st.secrets` when the env vars are absent.
