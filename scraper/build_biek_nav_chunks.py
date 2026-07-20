#!/usr/bin/env python3
"""Build KB-ready chunks from the BIEK nav content map.

These chunks are nav-contract backed: each row comes from the public nav tree
and its exact mapped HTML/PDF content source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = PROJECT_ROOT / "data" / "runtime"

DEFAULT_CONTENT_MAP_PATH = RUNTIME_DIR / "biek_nav_content_map.json"
DEFAULT_OUTPUT_PATH = RUNTIME_DIR / "biek_nav_chunks.jsonl"
DEFAULT_SUMMARY_PATH = RUNTIME_DIR / "biek_nav_chunks_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BIEK nav-backed KB chunks.")
    parser.add_argument("--input", default=str(DEFAULT_CONTENT_MAP_PATH), help="Path to BIEK nav content map JSON.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Path to write nav chunk JSONL.")
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY_PATH), help="Path to write summary JSON.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional debug limit on mapped nav rows.")
    parser.add_argument("--target-words", type=int, default=180, help="Preferred chunk size in words.")
    parser.add_argument("--max-words", type=int, default=260, help="Hard maximum chunk size in words.")
    parser.add_argument("--min-words", type=int, default=45, help="Minimum body words for non-link chunks.")
    parser.add_argument("--overlap-words", type=int, default=24, help="Approximate overlap between chunks.")
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Include nav rows marked handling_mode=exclude. By default these are skipped.",
    )
    return parser.parse_args()


def normalize_line(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def normalize_body_text(text: str) -> str:
    text = text.replace("\r", "\n")
    lines = [normalize_line(line) for line in text.splitlines()]
    cleaned: list[str] = []
    previous_blank = False
    seen_repeated: set[str] = set()
    for line in lines:
        if not line:
            if cleaned and not previous_blank:
                cleaned.append("")
            previous_blank = True
            continue
        previous_blank = False
        key = line.lower()
        if len(line) > 20 and key in seen_repeated:
            continue
        seen_repeated.add(key)
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def word_count(text: str) -> int:
    return len(text.split())


def stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", normalize_line(value).lower()).strip("-") or "nav"


def read_content_map(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text_source(path_value: str) -> tuple[str, str, str, str]:
    path = Path(path_value)
    raw = path.read_text(encoding="utf-8", errors="ignore")
    lines = [line.rstrip() for line in raw.splitlines()]
    title = normalize_line(lines[0]) if lines else ""
    source_url = ""
    source_page = ""
    body_start = 1
    if len(lines) > 1 and normalize_line(lines[1]).lower().startswith("url:"):
        source_url = lines[1].split(":", 1)[1].strip()
        body_start = 2
    if len(lines) > 2 and normalize_line(lines[2]).lower().startswith("source:"):
        source_page = lines[2].split(":", 1)[1].strip()
        body_start = 3
    body = normalize_body_text("\n".join(lines[body_start:]))
    return title, source_url, source_page, body


def alias_variants(label: str, parent: str, url: str) -> list[str]:
    aliases: list[str] = []

    def add(value: str) -> None:
        value = normalize_line(value)
        if value and value not in aliases:
            aliases.append(value)

    add(label)
    low = label.lower()
    if low.endswith(" form"):
        add(label[: -len(" form")])
    if low.endswith(" page"):
        add(label[: -len(" page")])
    if "verification" in low:
        add("verification form")
        add("verification forms")
        add("shared verification PDF")
    if "marksheet" in low:
        add("marksheet verification")
    if "model paper" in low:
        add("model papers")
    if "date sheet" in low:
        add("datesheet")
        add("exam schedule")
    if "authorized banks" in low:
        add("authorized bank")
        add("bank payment")
        add("UBL")
    if "contact" in low:
        add("contact")
        add("phone")
        add("email")
    if "board members" in low:
        add("board member")
        add("who appears on board members")
    if "chairman" in low:
        add("chairman of BIEK")
    if parent:
        add(f"{parent} {label}")
    if url:
        add(Path(url).name)
    return aliases


def structured_header(row: dict[str, Any], source: dict[str, Any], aliases: list[str]) -> str:
    fields = [
        ("Nav label", row.get("label")),
        ("Parent section", row.get("parent")),
        ("Handling mode", row.get("handling_mode")),
        ("Nav URL", row.get("url")),
        ("Source URL", source.get("url")),
        ("Source page", source.get("source_page")),
        ("Source kind", source.get("kind")),
        ("Source title", source.get("title")),
        ("Aliases", ", ".join(aliases)),
    ]
    return "\n".join(f"{label}: {normalize_line(str(value))}" for label, value in fields if normalize_line(str(value or "")))


def split_units(text: str) -> list[str]:
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = normalize_line(paragraph)
        if not paragraph:
            continue
        if word_count(paragraph) <= 90:
            units.append(paragraph)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        for sentence in sentences:
            sentence = normalize_line(sentence)
            if sentence:
                units.append(sentence)
    return units


def split_large_unit(unit: str, max_words: int) -> list[str]:
    words = unit.split()
    if len(words) <= max_words:
        return [unit]
    return [" ".join(words[index : index + max_words]) for index in range(0, len(words), max_words)]


def chunk_units(units: list[str], target_words: int, max_words: int, overlap_words: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    def overlap_tail(items: list[str]) -> list[str]:
        if overlap_words <= 0:
            return []
        selected: list[str] = []
        total = 0
        for item in reversed(items):
            selected.append(item)
            total += word_count(item)
            if total >= overlap_words:
                break
        return list(reversed(selected))

    for raw_unit in units:
        for unit in split_large_unit(raw_unit, max_words):
            unit_words = word_count(unit)
            if current and current_words + unit_words > max_words:
                chunks.append("\n\n".join(current).strip())
                current = overlap_tail(current)
                current_words = sum(word_count(item) for item in current)
            current.append(unit)
            current_words += unit_words
            if current_words >= target_words:
                chunks.append("\n\n".join(current).strip())
                current = overlap_tail(current)
                current_words = sum(word_count(item) for item in current)

    if current:
        chunks.append("\n\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def build_nav_doc(row: dict[str, Any]) -> dict[str, Any] | None:
    source = row.get("preferred_source") or {}
    text_file = str(source.get("text_file") or "").strip()
    if not text_file or not Path(text_file).exists():
        return None
    file_title, file_url, file_source_page, body = read_text_source(text_file)
    source = dict(source)
    source.setdefault("title", file_title)
    source.setdefault("url", file_url)
    source.setdefault("source_page", file_source_page)
    aliases = alias_variants(str(row.get("label") or ""), str(row.get("parent") or ""), str(row.get("url") or ""))
    return {
        "row": row,
        "source": source,
        "aliases": aliases,
        "body": body,
    }


def chunk_nav_doc(
    doc: dict[str, Any],
    *,
    target_words: int,
    max_words: int,
    min_words: int,
    overlap_words: int,
) -> list[dict[str, Any]]:
    row = doc["row"]
    source = doc["source"]
    body = doc["body"]
    aliases = doc["aliases"]
    if not body and not source.get("url"):
        return []

    header = structured_header(row, source, aliases)
    units = split_units(body)
    if not units:
        units = [f"The BIEK nav item {row.get('label')} is available at {row.get('url')}."]

    body_chunks = chunk_units(units, target_words=target_words, max_words=max_words, overlap_words=overlap_words)
    rows: list[dict[str, Any]] = []
    nav_id = str(row.get("id") or slugify(str(row.get("label") or "nav")))
    nav_url = str(row.get("url") or "")
    source_url = str(source.get("url") or nav_url)
    base_hash = stable_hash(f"{nav_id}|{source_url}")
    for index, body_chunk in enumerate(body_chunks, start=1):
        if word_count(body_chunk) < min_words and index > 1:
            continue
        text = f"{header}\n\nBody:\n{body_chunk}".strip()
        rows.append(
            {
                "chunk_id": f"biek::nav::{slugify(str(row.get('label') or 'nav'))}::{index:03d}::{base_hash}",
                "source_url": source_url,
                "source_page": source.get("source_page") or "",
                "title": row.get("label") or source.get("title") or "",
                "document_type": "nav_page" if str(source.get("kind") or "") == "html_page" else "nav_document",
                "section": row.get("parent") or row.get("label") or "BIEK",
                "topic": row.get("label") or "",
                "student_relevance": "nav_coverage",
                "handling_mode": row.get("handling_mode"),
                "nav_id": nav_id,
                "nav_label": row.get("label"),
                "nav_parent": row.get("parent"),
                "nav_url": nav_url,
                "aliases": aliases,
                "source_kind": source.get("kind"),
                "source_text_file": source.get("text_file"),
                "match_type": row.get("match_type"),
                "chunk_index": index,
                "word_count": word_count(text),
                "clean_body_text": body_chunk,
                "text": text,
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path: Path, content_map: dict[str, Any], docs: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> None:
    summary = {
        "source_content_map": str(DEFAULT_CONTENT_MAP_PATH),
        "nav_rows": content_map.get("total_item_count"),
        "mapped_rows": content_map.get("mapped_count"),
        "nav_docs_used": len(docs),
        "chunk_count": len(chunks),
        "chunks_by_parent": dict(Counter(str(chunk.get("nav_parent") or "") for chunk in chunks)),
        "chunks_by_label": dict(Counter(str(chunk.get("nav_label") or "") for chunk in chunks)),
        "chunks_by_document_type": dict(Counter(str(chunk.get("document_type") or "") for chunk in chunks)),
        "unmapped_rows": [
            {
                "label": row.get("label"),
                "url": row.get("url"),
                "warning": row.get("warning"),
            }
            for row in content_map.get("rows", [])
            if not row.get("preferred_source")
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.min_words > args.max_words:
        raise ValueError("--min-words cannot be greater than --max-words.")
    if args.target_words > args.max_words:
        raise ValueError("--target-words cannot be greater than --max-words.")

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)

    content_map = read_content_map(input_path)
    mapped_rows = [
        row
        for row in content_map.get("rows", [])
        if row.get("preferred_source")
        and (args.include_excluded or str(row.get("handling_mode") or "").strip().lower() != "exclude")
    ]
    if args.max_rows is not None:
        mapped_rows = mapped_rows[: args.max_rows]

    docs = [doc for row in mapped_rows if (doc := build_nav_doc(row)) is not None]
    chunks: list[dict[str, Any]] = []
    for doc in docs:
        chunks.extend(
            chunk_nav_doc(
                doc,
                target_words=args.target_words,
                max_words=args.max_words,
                min_words=args.min_words,
                overlap_words=args.overlap_words,
            )
        )

    write_jsonl(output_path, chunks)
    write_summary(summary_path, content_map, docs, chunks)

    print(f"Input content map: {input_path}")
    print(f"Mapped nav rows: {len(mapped_rows)}")
    print(f"Nav docs used: {len(docs)}")
    print(f"Output chunks: {output_path}")
    print(f"Chunk count: {len(chunks)}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
