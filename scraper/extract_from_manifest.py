from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

try:
    import fitz
except ImportError:  # pragma: no cover - handled at runtime
    fitz = None

try:
    import pdfplumber
except ImportError:  # pragma: no cover - handled at runtime
    pdfplumber = None

try:
    import pytesseract
except ImportError:  # pragma: no cover - handled at runtime
    pytesseract = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - handled at runtime
    Image = None

_ROOT_DIR = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _ROOT_DIR / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from kb_paths import raw_root
import scrape as scrape_utils
from site_profiles import SITE_PROFILES, SiteProfile, get_site_profile


def default_manifest_path(profile: SiteProfile) -> Path:
    return _ROOT_DIR / "data" / "runtime" / f"{profile.output_domain}_crawl_manifest.json"


def default_extracted_manifest_path(profile: SiteProfile) -> Path:
    return _ROOT_DIR / "data" / "runtime" / f"{profile.output_domain}_extracted_manifest.jsonl"


def extracted_html_dir(profile: SiteProfile) -> Path:
    return raw_root() / f"{profile.output_domain}-extracted-html"


def extracted_pdf_dir(profile: SiteProfile) -> Path:
    return raw_root() / f"{profile.output_domain}-extracted-pdf"


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text_output(
    *,
    output_dir: Path,
    url: str,
    title: str,
    discovered_from: str | None,
    text: str,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_slug = scrape_utils.unique_slug_for_url(url)
    filename = f"{base_slug[:140]}.txt"
    path = output_dir / filename

    try:
        with path.open("w", encoding="utf-8") as handle:
            handle.write(f"{title or url}\n")
            handle.write(f"URL: {url}\n")
            if discovered_from:
                handle.write(f"Source: {discovered_from}\n")
            handle.write("\n")
            handle.write(text)
            handle.write("\n")
    except OSError:
        fallback_name = f"{base_slug[-8:]}.txt"
        path = output_dir / fallback_name
        with path.open("w", encoding="utf-8") as handle:
            handle.write(f"{title or url}\n")
            handle.write(f"URL: {url}\n")
            if discovered_from:
                handle.write(f"Source: {discovered_from}\n")
            handle.write("\n")
            handle.write(text)
            handle.write("\n")

    return str(path)


def reset_output_targets(
    profile: SiteProfile,
    *,
    include_html: bool,
    include_pdf: bool,
) -> None:
    output_dirs: list[Path] = []
    if include_html:
        output_dirs.append(extracted_html_dir(profile))
    if include_pdf:
        output_dirs.append(extracted_pdf_dir(profile))

    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in output_dir.glob("*.txt"):
            path.unlink()

    manifest_path = default_extracted_manifest_path(profile)
    if manifest_path.exists():
        manifest_path.unlink()


def load_existing_extraction_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []

    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def successful_urls(rows: list[dict], kind: str) -> set[str]:
    urls: set[str] = set()
    for row in rows:
        if row.get("kind") != kind:
            continue
        if not str(row.get("extraction_status", "")).startswith("ok"):
            continue
        url = row.get("url")
        text_file = row.get("text_file")
        if url and text_file:
            urls.add(str(url))
    return urls


def html_entries_from_manifest(manifest: dict) -> list[dict]:
    return [entry for entry in manifest.get("entries", []) if entry.get("kind") == "html_page"]


def pdf_entries_from_manifest(manifest: dict) -> list[dict]:
    return [entry for entry in manifest.get("entries", []) if entry.get("kind") == "pdf"]


def extract_html_entry(
    *,
    session: requests.Session,
    profile: SiteProfile,
    entry: dict,
) -> dict:
    url = scrape_utils.normalize_url(entry["url"])
    title = entry.get("title") or url
    discovered_from = entry.get("discovered_from")

    try:
        response = session.get(url, timeout=profile.timeout, allow_redirects=True)
        response.raise_for_status()
    except requests.RequestException as exc:
        return {
            "kind": "html_page",
            "url": url,
            "title": title,
            "discovered_from": discovered_from,
            "extractor": "none",
            "text_file": None,
            "text_length": 0,
            "extraction_status": f"request_failed: {exc}",
        }

    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type.lower():
        return {
            "kind": "html_page",
            "url": url,
            "title": title,
            "discovered_from": discovered_from,
            "extractor": "none",
            "text_file": None,
            "text_length": 0,
            "extraction_status": f"non_html_content_type: {content_type}",
        }

    soup = BeautifulSoup(response.text, "lxml")
    resolved_title = soup.title.get_text(strip=True) if soup.title else title
    text, extractor_used, validation_reason = scrape_utils.extract_text(
        response.text,
        soup,
        profile,
        url,
        resolved_title,
    )

    if len(text) < profile.min_content_length:
        return {
            "kind": "html_page",
            "url": url,
            "title": resolved_title,
            "discovered_from": discovered_from,
            "extractor": extractor_used,
            "text_file": None,
            "text_length": len(text),
            "extraction_status": f"weak_html: {validation_reason}",
        }

    text_file = write_text_output(
        output_dir=extracted_html_dir(profile),
        url=url,
        title=resolved_title,
        discovered_from=discovered_from,
        text=text,
    )
    return {
        "kind": "html_page",
        "url": url,
        "title": resolved_title,
        "discovered_from": discovered_from,
        "extractor": extractor_used,
        "text_file": text_file,
        "text_length": len(text),
        "extraction_status": f"ok: {validation_reason}",
    }


def clean_pdf_text(profile: SiteProfile, url: str, title: str, text: str) -> str:
    cleaned = scrape_utils.normalized_text(text)
    if profile.name == "biek":
        cleaned = scrape_utils.clean_biek_text(url, title, cleaned)
    return scrape_utils.normalized_text(cleaned)


def tesseract_available() -> bool:
    return pytesseract is not None and Image is not None and shutil.which("tesseract") is not None


def extract_pdf_text_with_pymupdf(pdf_bytes: bytes) -> tuple[str, int]:
    if fitz is None:
        return "", 0

    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page_count = document.page_count
        text_parts = [page.get_text("text") for page in document]
        return "\n".join(text_parts), page_count
    finally:
        document.close()


def extract_pdf_text_with_pdfplumber(pdf_bytes: bytes) -> tuple[str, int]:
    if pdfplumber is None:
        return "", 0

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        page_count = len(pdf.pages)
        text_parts = [(page.extract_text() or "") for page in pdf.pages]
    return "\n".join(text_parts), page_count


def extract_pdf_text_with_ocr(pdf_bytes: bytes, ocr_lang: str) -> tuple[str, int]:
    if not tesseract_available() or fitz is None:
        return "", 0

    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page_count = document.page_count
        text_parts: list[str] = []
        for page in document:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            text_parts.append(pytesseract.image_to_string(image, lang=ocr_lang))
        return "\n".join(text_parts), page_count
    finally:
        document.close()


def extract_pdf_entry(
    *,
    session: requests.Session,
    profile: SiteProfile,
    entry: dict,
    ocr_lang: str,
) -> dict:
    url = scrape_utils.normalize_url(entry["url"])
    title = entry.get("title") or url
    discovered_from = entry.get("discovered_from")

    try:
        response = session.get(url, timeout=max(profile.timeout, 60), allow_redirects=True)
        response.raise_for_status()
    except requests.RequestException as exc:
        return {
            "kind": "pdf",
            "url": url,
            "title": title,
            "discovered_from": discovered_from,
            "extractor": "none",
            "text_file": None,
            "text_length": 0,
            "page_count": 0,
            "extraction_status": f"request_failed: {exc}",
        }

    content_type = response.headers.get("Content-Type", "")
    pdf_bytes = response.content

    extractors = (
        ("pymupdf", extract_pdf_text_with_pymupdf),
        ("pdfplumber", extract_pdf_text_with_pdfplumber),
    )

    last_error = None
    for extractor_name, extractor in extractors:
        try:
            raw_text, page_count = extractor(pdf_bytes)
        except Exception as exc:  # pragma: no cover - network/content variability
            last_error = f"{extractor_name}_failed: {exc}"
            continue

        cleaned = clean_pdf_text(profile, url, title, raw_text)
        if len(cleaned) < profile.min_content_length:
            last_error = f"{extractor_name}_weak_text"
            continue

        text_file = write_text_output(
            output_dir=extracted_pdf_dir(profile),
            url=url,
            title=title,
            discovered_from=discovered_from,
            text=cleaned,
        )
        return {
            "kind": "pdf",
            "url": url,
            "title": title,
            "discovered_from": discovered_from,
            "extractor": extractor_name,
            "text_file": text_file,
            "text_length": len(cleaned),
            "page_count": page_count,
            "content_type": content_type,
            "extraction_status": "ok",
        }

    if tesseract_available():
        try:
            raw_ocr_text, page_count = extract_pdf_text_with_ocr(pdf_bytes, ocr_lang)
        except Exception as exc:  # pragma: no cover - depends on local OCR setup/content
            last_error = f"ocr_failed: {exc}"
        else:
            cleaned_ocr = clean_pdf_text(profile, url, title, raw_ocr_text)
            if len(cleaned_ocr) >= profile.min_content_length:
                text_file = write_text_output(
                    output_dir=extracted_pdf_dir(profile),
                    url=url,
                    title=title,
                    discovered_from=discovered_from,
                    text=cleaned_ocr,
                )
                return {
                    "kind": "pdf",
                    "url": url,
                    "title": title,
                    "discovered_from": discovered_from,
                    "extractor": "tesseract_ocr",
                    "text_file": text_file,
                    "text_length": len(cleaned_ocr),
                    "page_count": page_count,
                    "content_type": content_type,
                    "extraction_status": "ok: ocr_fallback",
                }
            last_error = "ocr_weak_text"

    return {
        "kind": "pdf",
        "url": url,
        "title": title,
        "discovered_from": discovered_from,
        "extractor": "none",
        "text_file": None,
        "text_length": 0,
        "page_count": 0,
        "content_type": content_type,
        "extraction_status": last_error or "no_pdf_extractor_available",
    }


def append_manifest_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract HTML pages and PDFs from a crawl manifest."
    )
    parser.add_argument(
        "--site",
        choices=sorted(SITE_PROFILES),
        default="biek",
        help="Which configured site profile to extract for.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional explicit crawl manifest path.",
    )
    parser.add_argument(
        "--max-html",
        type=int,
        default=None,
        help="Optional limit on HTML entries to process.",
    )
    parser.add_argument(
        "--max-pdfs",
        type=int,
        default=None,
        help="Optional limit on PDF entries to process.",
    )
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Process only HTML entries.",
    )
    parser.add_argument(
        "--pdf-only",
        action="store_true",
        help="Process only PDF entries.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Clear previous extracted outputs and start from scratch.",
    )
    parser.add_argument(
        "--ocr-lang",
        default="eng",
        help="Tesseract OCR language(s) for scanned PDFs, e.g. eng or eng+urd.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.html_only and args.pdf_only:
        raise SystemExit("Choose either --html-only or --pdf-only, not both.")

    profile = get_site_profile(args.site)
    manifest_path = Path(args.manifest) if args.manifest else default_manifest_path(profile)
    manifest = load_manifest(manifest_path)
    session = scrape_utils.make_session(profile)
    extracted_manifest_path = default_extracted_manifest_path(profile)

    include_html = not args.pdf_only
    include_pdf = not args.html_only
    existing_rows = load_existing_extraction_rows(extracted_manifest_path)

    if args.fresh:
        reset_output_targets(profile, include_html=include_html, include_pdf=include_pdf)
        existing_rows = []

    html_entries = html_entries_from_manifest(manifest)
    pdf_entries = pdf_entries_from_manifest(manifest)

    if not args.fresh:
        completed_html_urls = successful_urls(existing_rows, "html_page")
        completed_pdf_urls = successful_urls(existing_rows, "pdf")
        if include_html:
            html_entries = [entry for entry in html_entries if entry.get("url") not in completed_html_urls]
        if include_pdf:
            pdf_entries = [entry for entry in pdf_entries if entry.get("url") not in completed_pdf_urls]

    if args.max_html is not None:
        html_entries = html_entries[: args.max_html]
    if args.max_pdfs is not None:
        pdf_entries = pdf_entries[: args.max_pdfs]

    rows: list[dict] = []

    print(f"Site: {profile.name}")
    print(f"Crawl manifest: {manifest_path}")
    print(f"HTML output dir: {extracted_html_dir(profile)}")
    print(f"PDF output dir: {extracted_pdf_dir(profile)}")
    print(f"Extracted manifest: {extracted_manifest_path}\n")
    print(f"OCR available: {'yes' if tesseract_available() else 'no'} | OCR lang: {args.ocr_lang}")
    if args.fresh:
        print("Mode: fresh run\n")
    else:
        completed_html = len(successful_urls(existing_rows, "html_page"))
        completed_pdf = len(successful_urls(existing_rows, "pdf"))
        print(f"Mode: resume if possible (completed HTML: {completed_html}, completed PDFs: {completed_pdf})\n")

    if include_html:
        print(f"Extracting HTML pages: {len(html_entries)}")
        for index, entry in enumerate(html_entries, start=1):
            print(f"  [HTML {index}/{len(html_entries)}] {entry['url']}")
            rows.append(extract_html_entry(session=session, profile=profile, entry=entry))

    if include_pdf:
        print(f"Extracting PDFs: {len(pdf_entries)}")
        for index, entry in enumerate(pdf_entries, start=1):
            print(f"  [PDF {index}/{len(pdf_entries)}] {entry['url']}")
            rows.append(
                extract_pdf_entry(
                    session=session,
                    profile=profile,
                    entry=entry,
                    ocr_lang=args.ocr_lang,
                )
            )

    append_manifest_rows(extracted_manifest_path, rows)

    success_count = sum(1 for row in rows if str(row.get("extraction_status", "")).startswith("ok"))
    failed_count = len(rows) - success_count
    print("\n" + "-" * 50)
    print(f"Processed: {len(rows)}")
    print(f"Successful extractions: {success_count}")
    print(f"Weak/failed extractions: {failed_count}")
    print(f"Metadata saved to: {extracted_manifest_path}")


if __name__ == "__main__":
    main()
