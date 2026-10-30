"""PDF -> page-tagged plain text using PyMuPDF."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

import pymupdf

MAX_PAGES = 15  # keeps the prompt within a sensible token budget
MIN_PAGE_CHARS = 20  # below this a page is probably scanned / image-only


class PdfReadError(ValueError):
    pass


@dataclass
class PageText:
    page: int  # 1-based
    text: str


@dataclass
class DocumentText:
    filename: str
    pages: list[PageText]
    total_pages: int
    warnings: list[str] = field(default_factory=list)

    @property
    def has_text(self) -> bool:
        return any(len(p.text.strip()) >= MIN_PAGE_CHARS for p in self.pages)

    def to_prompt_text(self) -> str:
        return "\n\n".join(f"[page {p.page}]\n{p.text}" for p in self.pages)


def extract_text(data: bytes, filename: str) -> DocumentText:
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:
        raise PdfReadError(f"{filename}: not a readable PDF ({e})") from e
    with doc:
        if doc.needs_pass:
            raise PdfReadError(f"{filename}: PDF is password-protected")
        warnings: list[str] = []
        pages: list[PageText] = []
        for i, page in enumerate(doc):
            if i >= MAX_PAGES:
                warnings.append(f"Only the first {MAX_PAGES} of {len(doc)} pages were read.")
                break
            # NFKC folds ligatures (ﬁ -> fi) and odd spaces so evidence matching works.
            text = unicodedata.normalize("NFKC", page.get_text(sort=True)).strip()
            if len(text) < MIN_PAGE_CHARS:
                warnings.append(f"Page {i + 1} has no extractable text (scanned image?). It was not analysed.")
            pages.append(PageText(page=i + 1, text=text))
        return DocumentText(filename=filename, pages=pages, total_pages=len(doc), warnings=warnings)
