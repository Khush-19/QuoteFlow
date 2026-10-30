"""QuoteFlow Lite - compare two freight quotation PDFs.

Run: .venv/bin/streamlit run app.py
"""

from __future__ import annotations

import hashlib
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from quoteflow.evidence import check_quote, is_trusted
from quoteflow.llm import ExtractionError, LLMConfig, LLMConfigError, extract_quote
from quoteflow.pdf_text import PdfReadError, extract_text
from quoteflow.recommend import Analysis, explain, fmt, mismatch_text, recommend
from quoteflow.schema import Charge, ExtractedQuote, FreightQuote

SAMPLES = Path(__file__).parent / "samples"
SAMPLE_FILES = ["quote_a_blueharbor.pdf", "quote_b_northwind.pdf"]
STATUSES = ["included", "estimated", "excluded", "at_cost", "not_mentioned"]
CATEGORIES = ["freight", "origin", "destination", "customs", "surcharge", "documentation", "insurance", "other"]
FLAG_TEXT = {
    "priced_both": "Priced in both",
    "gap_a": "Missing in A",
    "gap_b": "Missing in B",
    "unknown_both": "Unknown in both",
}

st.set_page_config(page_title="QuoteFlow Lite", page_icon="🚢", layout="wide")
load_dotenv()
try:  # Streamlit Cloud secrets -> env, without overriding a local .env
    for k in ("NEBIUS_API_KEY", "NEBIUS_MODEL", "NEBIUS_BASE_URL", "NEBIUS_RESPONSE_FORMAT"):
        if k in st.secrets and not os.getenv(k):
            os.environ[k] = str(st.secrets[k])
except Exception:
    pass


def llm_config() -> tuple[LLMConfig | None, str | None]:
    if "llm_config" not in st.session_state:
        try:
            st.session_state.llm_config = LLMConfig.from_env()
        except LLMConfigError as e:
            return None, str(e)
    return st.session_state.llm_config, None


# --- sidebar -----------------------------------------------------------------

config, config_error = llm_config()
with st.sidebar:
    st.header("Settings")
    if config:
        st.success(f"Nemotron ready\n\n`{config.model}`")
    else:
        st.warning(f"LLM not configured: {config_error}")
    use_fixture = st.toggle(
        "Offline demo (samples only)",
        value=config is None,
        help="Skip the LLM and load the hand-labelled ground truth for the bundled sample PDFs. "
        "This is NOT model output and is labelled as such everywhere.",
    )
    target = st.text_input("Compare in currency", value="USD", max_chars=3).strip().upper() or "USD"
    st.caption("Exchange rates (user-supplied, no live rates). One line per currency, e.g. `EUR = 1.08` "
               f"meaning 1 EUR = 1.08 {target}.")
    rates_text = st.text_area("Rates", value="", label_visibility="collapsed", placeholder="EUR = 1.08")
    rates: dict[str, Decimal] = {}
    for line in rates_text.splitlines():
        if "=" in line:
            ccy, _, val = line.partition("=")
            try:
                rates[ccy.strip().upper()] = Decimal(val.strip())
            except InvalidOperation:
                st.error(f"Bad rate line: {line!r}")


# --- inputs ------------------------------------------------------------------

st.title("QuoteFlow Lite")
st.caption("Upload two freight quotations. Charges are extracted with NVIDIA Nemotron; every total is computed "
           "in plain Python, and missing fees are never treated as zero.")

if st.button("Load sample quotes"):
    st.session_state.files = {
        side: (name, (SAMPLES / name).read_bytes()) for side, name in zip("ab", SAMPLE_FILES)
    }

cols = st.columns(2)
for side, col in zip("ab", cols):
    up = col.file_uploader(f"Quote {side.upper()}", type="pdf", key=f"upload_{side}")
    if up is not None:
        st.session_state.setdefault("files", {})[side] = (up.name, up.getvalue())
    current = st.session_state.get("files", {}).get(side)
    if current:
        col.caption(f"📄 {current[0]}")

files = st.session_state.get("files", {})


def run_extraction(side: str, name: str, data: bytes) -> ExtractedQuote:
    doc = extract_text(data, name)
    st.session_state.setdefault("docs", {})[side] = doc
    key = (hashlib.sha256(data).hexdigest(), use_fixture)
    cache = st.session_state.setdefault("extractions", {})
    if key in cache:
        return cache[key]
    if use_fixture:
        truth = SAMPLES / "ground_truth" / f"{Path(name).stem}.json"
        if not truth.exists():
            raise ExtractionError(f"Offline demo only works with the bundled samples; {name} is not one.")
        quote = FreightQuote.model_validate_json(truth.read_text())
        result = ExtractedQuote(source_file=name, quote=quote, evidence_checks=check_quote(quote, doc),
                                source="ground_truth_fixture", warnings=doc.warnings)
    else:
        if config is None:
            raise ExtractionError(config_error or "LLM not configured")
        result = extract_quote(doc, config)
    cache[key] = result
    return result


if len(files) == 2 and st.button("Extract & compare", type="primary"):
    st.session_state.results = {}
    st.session_state.pop("explanation", None)
    for side in "ab":
        name, data = files[side]
        with st.spinner(f"Reading {name}..."):
            try:
                st.session_state.results[side] = run_extraction(side, name, data)
            except (PdfReadError, ExtractionError) as e:
                st.error(str(e))
            except Exception as e:  # surface real API errors verbatim
                st.error(f"{name}: extraction failed: {type(e).__name__}: {e}")

results: dict[str, ExtractedQuote] = st.session_state.get("results", {})
if len(results) < 2:
    st.stop()


# --- review ------------------------------------------------------------------

st.header("1. Review extracted charges")
st.caption("Edit any cell to correct the extraction; the comparison uses your edits.")

SOURCE_BADGE = {
    "llm": "🤖 Extracted by Nemotron",
    "ground_truth_fixture": "📋 Hand-labelled sample data (not model output)",
    "user_edited": "✏️ Edited by you",
}

edited: dict[str, ExtractedQuote] = {}
review_cols = st.columns(2)
for side, col in zip("ab", review_cols):
    r = results[side]
    q = r.quote
    with col:
        st.subheader(f"{side.upper()}: {q.carrier_or_forwarder or r.source_file}")
        st.caption(f"{SOURCE_BADGE[r.source]}" + (f" · `{r.model}`" if r.model else ""))
        st.markdown(
            f"**Ref** {q.quote_ref or '—'} · **Route** {q.origin or '?'} → {q.destination or '?'} · "
            f"**Equipment** {q.equipment or '—'} · **Incoterm** {q.incoterm or '—'}  \n"
            f"**Transit** {f'{q.transit_days} days' if q.transit_days else '—'} · "
            f"**Valid until** {q.validity_until or '—'} · **Stated total** "
            f"{fmt(None) if q.quoted_total is None else f'{q.quoted_total_currency} {q.quoted_total:,.2f}'}"
        )
        checks = {c.label: c for c in r.evidence_checks}
        rows = []
        for c in q.charges:
            chk = checks.get(f"charge: {c.name}")
            rows.append({
                "Charge": c.name,
                "Category": c.category,
                "Amount": float(c.amount) if c.amount is not None else None,
                "Currency": c.currency,
                "Qty": float(c.quantity) if c.quantity is not None else None,
                "Basis": c.basis,
                "Status": c.status,
                "Page": c.evidence.page,
                "Evidence": c.evidence.quote_text,
                "Check": "—" if chk is None else ("✅" if is_trusted(chk) and chk.status == "verified" else "⚠️" if is_trusted(chk) else "❌"),
            })
        table = st.data_editor(
            rows,
            key=f"editor_{side}",
            num_rows="dynamic",
            hide_index=True,
            disabled=["Check"],
            column_config={
                "Amount": st.column_config.NumberColumn(format="%.2f", help="Empty = unknown (never zero)"),
                "Status": st.column_config.SelectboxColumn(options=STATUSES, required=True),
                "Category": st.column_config.SelectboxColumn(options=CATEGORIES, required=True),
                "Evidence": st.column_config.TextColumn(width="large"),
                "Check": st.column_config.TextColumn(help="✅ snippet found in PDF · ⚠️ wrong page · ❌ not found / amount not in snippet"),
            },
        )
        for w in r.warnings:
            st.warning(w, icon="⚠️")

        def dec(x):
            return None if x is None or x != x else Decimal(str(round(x, 4)))  # x != x catches NaN

        new_charges = []
        for row in table:
            if not row.get("Charge"):
                continue
            new_charges.append({
                "name": row["Charge"], "category": row.get("Category") or "other", "amount": dec(row.get("Amount")),
                "currency": row.get("Currency") or None, "quantity": dec(row.get("Qty")), "basis": row.get("Basis") or None,
                "status": row.get("Status") or "included",
                "evidence": {"page": int(row.get("Page") or 1), "quote_text": row.get("Evidence") or "(added by user)"},
            })
        try:
            new_quote = q.model_copy(update={"charges": [Charge.model_validate(c) for c in new_charges]})
        except ValueError as e:
            st.error(f"Edit not applied: {e}")
            new_quote = q
        if new_quote.model_dump() != q.model_dump():
            doc = st.session_state.get("docs", {}).get(side)
            edited[side] = r.model_copy(update={
                "quote": new_quote,
                "source": "user_edited",
                "evidence_checks": check_quote(new_quote, doc) if doc else r.evidence_checks,
            })
            st.caption("✏️ Using your edits.")
        else:
            edited[side] = r


# --- comparison --------------------------------------------------------------

analysis: Analysis = recommend(edited["a"], edited["b"], target=target, rates=rates)
rec = analysis.recommendation
name = {s: edited[s].quote.carrier_or_forwarder or f"Quote {s.upper()}" for s in "ab"}

st.header("2. Cost comparison")
m = st.columns(2)
for side, col in zip("ab", m):
    s = getattr(analysis, f"summary_{side}")
    with col:
        st.subheader(name[side])
        st.markdown(
            "| Item | Amount |\n|---|---:|\n"
            f"| Stated total (from PDF) | **{fmt(s.quoted_total)}** |\n"
            f"| Sum of priced lines | {', '.join(fmt(x) for x in s.calculated_totals) or 'none'} |\n"
            f"| Estimates (per quote) | {', '.join(fmt(x) for x in s.estimated_extras) or 'none'} |\n"
            f"| Known cost in {target} | **{fmt(s.converted_total)}** |"
        )
        if s.total_mismatch:
            st.error(mismatch_text(s.total_mismatch))
        if s.converted_note:
            st.info(s.converted_note)
        if s.unknown_items:
            st.markdown("**Unknown costs (not counted as zero):** " + ", ".join(f"{u.name} *({u.reason.replace('_', ' ')})*" for u in s.unknown_items))

st.subheader("Charge-by-charge")


def cell_text(cell):
    if cell.status == "absent":
        return "— not in quote"
    txt = fmt(cell.amount) if cell.amount else cell.status.replace("_", " ")
    if cell.amount and cell.status == "estimated":
        txt += " (est.)"
    if cell.note:
        txt += f" — {cell.note}"
    elif cell.page:
        txt += f" (p.{cell.page})"
    return txt


st.dataframe(
    [{"Charge": r.label, f"A: {name['a']}": cell_text(r.a), f"B: {name['b']}": cell_text(r.b), "Flag": FLAG_TEXT[r.flag]}
     for r in analysis.comparison.rows],
    hide_index=True,
    use_container_width=True,
)


# --- recommendation ----------------------------------------------------------

st.header("3. Recommendation")
box = {"a": st.success, "b": st.success, "too_close": st.info, "cannot_compare": st.warning}[rec.verdict]
box(f"**{rec.headline}**  \nConfidence: **{rec.confidence}**")
if rec.adjusted_a and rec.adjusted_b:
    c1, c2 = st.columns(2)
    c1.metric(f"{name['a']}: like-for-like estimate", fmt(rec.adjusted_a), help="Priced lines + estimates + gaps filled at the other quote's price")
    c2.metric(f"{name['b']}: like-for-like estimate", fmt(rec.adjusted_b), help="Priced lines + estimates + gaps filled at the other quote's price")
st.markdown("**Why**\n" + "\n".join(f"- {x}" for x in rec.reasons))
if rec.caveats:
    st.markdown("**Check before booking**\n" + "\n".join(f"- {x}" for x in rec.caveats))

if config and st.button("Explain with Nemotron"):
    with st.spinner("Asking Nemotron..."):
        st.session_state.explanation = explain(rec, config)
if "explanation" in st.session_state:
    text, problem = st.session_state.explanation
    if text:
        st.markdown(f"> {text}")
        st.caption(f"Generated by `{config.model}` from the facts above; numbers checked against them.")
    else:
        st.warning(problem)

st.download_button(
    "Download comparison (JSON)",
    data=analysis.model_dump_json(indent=2),
    file_name="quoteflow_comparison.json",
    mime="application/json",
)
