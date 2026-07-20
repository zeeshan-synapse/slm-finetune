#!/usr/bin/env python3
"""Map BIEK nav coverage rows to the best extracted HTML/PDF content source."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit, urlunsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COVERAGE_PATH = PROJECT_ROOT / "config" / "biek_nav_coverage.json"
DEFAULT_EXTRACTED_MANIFEST = PROJECT_ROOT / "data" / "runtime" / "biek_extracted_manifest.jsonl"
DEFAULT_ENRICHED_DOCS = PROJECT_ROOT / "data" / "runtime" / "biek_enriched_documents.jsonl"
DEFAULT_RAW_HTML_DIR = PROJECT_ROOT / "data" / "raw" / "biek-extracted-html"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "runtime" / "biek_nav_content_map.json"

DOCUMENT_EXTENSIONS = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".mp4", ".doc", ".docx", ".xls", ".xlsx")
SENSITIVE_LABEL_TOKENS = (
    "chairman",
    "board members",
    "history",
    "committees",
    "committee",
    "statistics",
    "contact",
    "authorized banks",
)


def normalize_url(value: str | None) -> str:
    url = str(value or "").strip()
    if not url:
        return ""
    url = url.replace("\\", "/")
    parts = urlsplit(url)
    scheme = "https"
    netloc = parts.netloc.lower()
    if netloc == "biek.edu.pk":
        netloc = "www.biek.edu.pk"
    path = unquote(parts.path or "")
    path = re.sub(r"/+", "/", path).rstrip("/")
    encoded_path = quote(path, safe="/:@&=+$,;~()*!'.-")
    query = parts.query
    normalized = urlunsplit((scheme, netloc, encoded_path, query, ""))
    return normalized.rstrip("/").lower()


def display_url(value: str | None) -> str:
    url = str(value or "").strip()
    if not url:
        return ""
    if url.startswith("http://"):
        url = "https://" + url[len("http://") :]
    return url.replace("https://biek.edu.pk", "https://www.biek.edu.pk", 1).rstrip("/")


def normalize_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def tokens(value: str) -> set[str]:
    return {token for token in re.sub(r"[^a-z0-9]+", " ", value.lower()).split() if len(token) > 1}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                rows.append(data)
    return rows


def source_url(row: dict[str, Any]) -> str:
    return normalize_url(str(row.get("source_url") or row.get("url") or ""))


def source_page(row: dict[str, Any]) -> str:
    return normalize_url(str(row.get("source_page") or ""))


def is_document_url(value: str | None) -> bool:
    path = unquote(urlsplit(str(value or "")).path).lower()
    return path.endswith(DOCUMENT_EXTENSIONS)


def is_container_row(nav_row: dict[str, Any]) -> bool:
    url = display_url(str(nav_row.get("url") or ""))
    return int(nav_row.get("level") or 0) == 0 or url.endswith("default.asp#")


def is_sensitive_row(nav_row: dict[str, Any]) -> bool:
    label = normalize_text(str(nav_row.get("label") or "")).lower()
    return any(token in label for token in SENSITIVE_LABEL_TOKENS)


def read_raw_html_pages(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not directory.exists():
        return rows
    for path in sorted(directory.glob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lines = [line.rstrip() for line in text.splitlines()]
        if len(lines) < 2:
            continue
        title = normalize_text(lines[0])
        url_line = normalize_text(lines[1])
        url = url_line.split("URL:", 1)[1].strip() if url_line.lower().startswith("url:") else url_line
        source_page_value = ""
        if len(lines) > 2 and normalize_text(lines[2]).lower().startswith("source:"):
            source_page_value = lines[2].split(":", 1)[1].strip()
        body_start = 3 if source_page_value else 2
        rows.append(
            {
                "kind": "html_page",
                "source_kind": "html_page",
                "source_url": display_url(url),
                "source_page": display_url(source_page_value),
                "title": title,
                "text_file": str(path),
                "text_length": len("\n".join(lines[body_start:]).strip()),
                "extraction_status": "ok",
            }
        )
    return rows


def resolve_exact_source(nav_row: dict[str, Any], sources: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str, float, str]:
    nav_url = normalize_url(str(nav_row.get("url") or ""))
    if not nav_url:
        return None, "unmapped", 0.0, "Nav row has no URL."

    if is_container_row(nav_row):
        exact = [
            source
            for source in sources
            if source_url(source) == nav_url
        ]
        if exact:
            return exact[0], "exact_url", 100.0, ""
        return None, "unmapped", 0.0, "Container row has no exact extracted container page."

    nav_is_doc = is_document_url(nav_url)
    for source in sources:
        if source_url(source) == nav_url:
            return source, "exact_url", 100.0, ""

    if nav_is_doc:
        return None, "unmapped", 0.0, "No extracted document matched the exact nav URL."
    if is_sensitive_row(nav_row):
        return None, "unmapped", 0.0, "No exact extracted source found; weak matching disabled for sensitive nav item."
    return None, "unmapped", 0.0, "No exact extracted source found; weak token matching disabled."


def build_source(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "url": display_url(str(row.get("source_url") or row.get("url") or "")),
        "source_page": display_url(str(row.get("source_page") or "")),
        "title": normalize_text(str(row.get("title") or "")),
        "kind": str(row.get("kind") or row.get("source_kind") or ""),
        "text_file": str(row.get("text_file") or ""),
        "text_length": row.get("text_length"),
        "extraction_status": row.get("extraction_status"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve BIEK nav rows to extracted content sources.")
    parser.add_argument("--coverage", default=str(DEFAULT_COVERAGE_PATH))
    parser.add_argument("--extracted", default=str(DEFAULT_EXTRACTED_MANIFEST))
    parser.add_argument("--enriched", default=str(DEFAULT_ENRICHED_DOCS))
    parser.add_argument("--raw-html-dir", default=str(DEFAULT_RAW_HTML_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    coverage_path = Path(args.coverage)
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    nav_rows = coverage.get("rows") or []
    sources = read_raw_html_pages(Path(args.raw_html_dir)) + read_jsonl(Path(args.enriched)) + read_jsonl(Path(args.extracted))

    rows: list[dict[str, Any]] = []
    suspicious_prevented = 0
    for nav_row in nav_rows:
        best, match_type, best_score, warning = resolve_exact_source(nav_row, sources)
        if best is None and not is_container_row(nav_row):
            suspicious_prevented += 1
        preferred = build_source(best) if best is not None else {}
        rows.append(
            {
                "id": nav_row.get("id"),
                "label": nav_row.get("label"),
                "url": nav_row.get("url"),
                "parent": nav_row.get("parent"),
                "handling_mode": nav_row.get("handling_mode"),
                "preferred_source": preferred,
                "match_type": match_type,
                "match_score": round(float(best_score), 4),
                "fallback_link": nav_row.get("url"),
                "warning": warning,
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "site": "biek",
        "source_coverage_file": str(coverage_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_item_count": len(rows),
        "mapped_count": sum(1 for row in rows if row.get("preferred_source")),
        "unmapped_count": sum(1 for row in rows if not row.get("preferred_source")),
        "suspicious_mappings_prevented": suspicious_prevented,
        "rows": rows,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Coverage file: {coverage_path.resolve()}")
    print(f"Output content map: {output.resolve()}")
    print(f"Total nav rows: {len(rows)}")
    print(f"Mapped rows: {payload['mapped_count']}")
    print(f"Unmapped rows: {payload['unmapped_count']}")
    print(f"Suspicious mappings prevented: {suspicious_prevented}")

    check_labels = [
        "History of Board",
        "Chairman",
        "Board Members",
        "Committees",
        "Contact us",
        "Authorized Banks",
        "Statistics",
        "Verification Certificate Form",
        "Verification Provisional Certificate",
        "Verification Marksheet Form",
        "Verification Migration Form",
    ]
    print("Key mapping checks:")
    for label in check_labels:
        row = next((item for item in rows if item.get("label") == label), None)
        if row is None:
            print(f"- {label}: missing coverage row")
            continue
        source = row.get("preferred_source") or {}
        source_label = source.get("url") or row.get("fallback_link") or ""
        print(f"- {label}: {row.get('match_type')} -> {source_label}")


if __name__ == "__main__":
    main()
