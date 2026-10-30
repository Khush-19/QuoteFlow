"""Generate synthetic freight quote PDFs plus hand-labelled ground truth.

All companies, references and prices are fictional. Run:
    .venv/bin/python samples/make_samples.py

Writes samples/quote_a_blueharbor.pdf, samples/quote_b_northwind.pdf and
samples/ground_truth/<name>.json (FreightQuote JSON written by hand here, NOT model output).
"""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf

HERE = Path(__file__).parent
PAGE = pymupdf.Rect(0, 0, 595, 842)  # A4 in points
BODY = pymupdf.Rect(50, 50, 545, 800)

CSS = """
body { font-family: sans-serif; font-size: 10pt; }
h1 { font-size: 16pt; margin-bottom: 2pt; }
h2 { font-size: 12pt; margin-top: 10pt; }
table { border-collapse: collapse; width: 100%; }
td, th { border: 1px solid #888; padding: 3pt; }
.small { font-size: 8pt; color: #444; }
"""


def ev(page: int, text: str) -> dict:
    return {"page": page, "quote_text": text}


def charge(name, category, amount, currency, status, page, snippet, basis=None, quantity=None):
    return {
        "name": name,
        "category": category,
        "amount": amount,
        "currency": currency,
        "basis": basis,
        "quantity": quantity,
        "status": status,
        "evidence": ev(page, snippet),
    }


# --- Quote A: tabular layout, all USD, quoted total matches the lines -------

A_ROWS = [
    # (label, amount text, basis)
    ("Ocean freight Shanghai - Rotterdam", "USD 1,850.00", "per container"),
    ("BAF (bunker adjustment)", "USD 310.00", "per container"),
    ("Origin THC", "USD 145.00", "per container"),
    ("Export customs clearance", "USD 85.00", "per shipment"),
    ("Documentation fee (B/L)", "USD 65.00", "per shipment"),
]
A_EXCLUSION = "Destination charges, import duties and taxes are excluded and payable by the consignee."
A_INSURANCE = "Cargo insurance is not included unless requested in writing."

A_HTML = f"""
<h1>Blue Harbor Logistics</h1>
<p class="small">Fictional company for demo purposes</p>
<p><b>Quotation No.:</b> BHL-2026-0417<br>
<b>Route:</b> Shanghai, CN to Rotterdam, NL<br>
<b>Service:</b> Sea freight FCL, 1x40HC<br>
<b>Incoterm:</b> FOB Shanghai<br>
<b>Estimated transit:</b> 32 days<br>
<b>Valid until:</b> 2026-11-30</p>
<h2>Charges</h2>
<table>
<tr><th>Charge</th><th>Amount</th><th>Basis</th></tr>
{''.join(f'<tr><td>{n}</td><td>{a}</td><td>{b}</td></tr>' for n, a, b in A_ROWS)}
<tr><td><b>Total</b></td><td><b>USD 2,455.00</b></td><td></td></tr>
</table>
<h2>Terms</h2>
<p>{A_EXCLUSION}</p>
<p>{A_INSURANCE}</p>
<p class="small">Rates subject to equipment and space availability at time of booking.</p>
"""

A_TRUTH = {
    "carrier_or_forwarder": "Blue Harbor Logistics",
    "quote_ref": "BHL-2026-0417",
    "origin": "Shanghai, CN",
    "destination": "Rotterdam, NL",
    "mode": "sea_fcl",
    "incoterm": "FOB",
    "equipment": "1x40HC",
    "validity_until": "2026-11-30",
    "transit_days": 32,
    "charges": [
        charge("Ocean freight", "freight", 1850.00, "USD", "included", 1, "Ocean freight Shanghai - Rotterdam USD 1,850.00", "per container"),
        charge("BAF (bunker adjustment)", "surcharge", 310.00, "USD", "included", 1, "BAF (bunker adjustment) USD 310.00", "per container"),
        charge("Origin THC", "origin", 145.00, "USD", "included", 1, "Origin THC USD 145.00", "per container"),
        charge("Export customs clearance", "customs", 85.00, "USD", "included", 1, "Export customs clearance USD 85.00", "per shipment"),
        charge("Documentation fee (B/L)", "documentation", 65.00, "USD", "included", 1, "Documentation fee (B/L) USD 65.00", "per shipment"),
        charge("Destination charges", "destination", None, None, "excluded", 1, A_EXCLUSION),
        charge("Import duties and taxes", "customs", None, None, "excluded", 1, A_EXCLUSION),
        charge("Cargo insurance", "insurance", None, None, "excluded", 1, A_INSURANCE),
    ],
    "quoted_total": 2455.00,
    "quoted_total_currency": "USD",
    "quoted_total_evidence": ev(1, "Total USD 2,455.00"),
    "exclusions": [ev(1, A_EXCLUSION), ev(1, A_INSURANCE)],
    "extraction_warnings": [],
}

# --- Quote B: prose layout, USD, total omits an estimate, BAF not mentioned -

B_P1 = "Ocean freight for 1x40HC from Shanghai to Rotterdam is USD 1,720.00 per container."
B_P2 = "Terminal handling at origin is USD 150.00 and our documentation fee is USD 50.00."
B_P3 = "Destination terminal handling (DTHC) at Rotterdam is USD 230.00."
B_P4 = "On-carriage to the consignee's warehouse is estimated at USD 400.00 and will be invoiced at actuals."
B_P5 = "Import customs clearance at destination will be charged at cost."
B_P6 = "Total quoted: USD 2,100.00"
B_P7 = "Peak season surcharge may apply depending on sailing date."

B_HTML = f"""
<h1>Northwind Freight Partners</h1>
<p class="small">Fictional company for demo purposes</p>
<p>Dear customer,</p>
<p>Thank you for your enquiry (our ref. NFP/Q/88213). Please find our offer below for an
FCA Shanghai shipment. Expected transit time is 29 days. This offer is valid until 2026-11-15.</p>
<p>{B_P1} {B_P2}</p>
<p>{B_P3} {B_P4}</p>
<p>{B_P5}</p>
<p><b>{B_P6}</b></p>
<p class="small">{B_P7} Insurance available on request.</p>
<p>Kind regards,<br>Northwind Freight Partners</p>
"""

B_TRUTH = {
    "carrier_or_forwarder": "Northwind Freight Partners",
    "quote_ref": "NFP/Q/88213",
    "origin": "Shanghai",
    "destination": "Rotterdam",
    "mode": "sea_fcl",
    "incoterm": "FCA",
    "equipment": "1x40HC",
    "validity_until": "2026-11-15",
    "transit_days": 29,
    "charges": [
        charge("Ocean freight", "freight", 1720.00, "USD", "included", 1, B_P1, "per container"),
        charge("Origin terminal handling", "origin", 150.00, "USD", "included", 1, "Terminal handling at origin is USD 150.00"),
        charge("Documentation fee", "documentation", 50.00, "USD", "included", 1, "our documentation fee is USD 50.00"),
        charge("Destination terminal handling (DTHC)", "destination", 230.00, "USD", "included", 1, B_P3),
        charge("On-carriage to consignee", "destination", 400.00, "USD", "estimated", 1, B_P4),
        charge("Import customs clearance", "customs", None, None, "at_cost", 1, B_P5),
        charge("Peak season surcharge", "surcharge", None, None, "at_cost", 1, B_P7),
    ],
    "quoted_total": 2100.00,
    "quoted_total_currency": "USD",
    "quoted_total_evidence": ev(1, B_P6),
    "exclusions": [ev(1, B_P5), ev(1, B_P7)],
    "extraction_warnings": [],
}

SAMPLES = {
    "quote_a_blueharbor": (A_HTML, A_TRUTH),
    "quote_b_northwind": (B_HTML, B_TRUTH),
}


def render_pdf(html: str, path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE.width, height=PAGE.height)
    spare, _ = page.insert_htmlbox(BODY, html, css=CSS)
    if spare < 0:
        raise RuntimeError(f"{path.name}: content did not fit on one page")
    doc.set_metadata({"title": path.stem, "producer": "QuoteFlow synthetic sample"})
    doc.save(path)


def main() -> None:
    truth_dir = HERE / "ground_truth"
    truth_dir.mkdir(exist_ok=True)
    for name, (html, truth) in SAMPLES.items():
        render_pdf(html, HERE / f"{name}.pdf")
        (truth_dir / f"{name}.json").write_text(json.dumps(truth, indent=2) + "\n")
        print(f"wrote samples/{name}.pdf and samples/ground_truth/{name}.json")


if __name__ == "__main__":
    main()
