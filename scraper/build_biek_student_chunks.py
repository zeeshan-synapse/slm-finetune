from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = PROJECT_ROOT / "data" / "runtime"

DEFAULT_INPUT_PATH = RUNTIME_DIR / "biek_enriched_documents.jsonl"
DEFAULT_OUTPUT_PATH = RUNTIME_DIR / "biek_student_chunks.jsonl"
DEFAULT_SUMMARY_PATH = RUNTIME_DIR / "biek_student_chunks_summary.json"

KEEP_RELEVANCE = {"high_relevance", "medium_relevance"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build student-focused KB chunks from enriched BIEK records."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Path to enriched BIEK JSONL.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Path to write chunk JSONL.")
    parser.add_argument(
        "--summary-output",
        default=str(DEFAULT_SUMMARY_PATH),
        help="Path to write chunk summary JSON.",
    )
    parser.add_argument("--max-docs", type=int, default=None, help="Optional debug limit on docs.")
    parser.add_argument("--target-words", type=int, default=170, help="Preferred chunk size in words.")
    parser.add_argument("--max-words", type=int, default=230, help="Hard maximum chunk size in words.")
    parser.add_argument("--min-words", type=int, default=70, help="Preferred minimum chunk size in words.")
    parser.add_argument("--overlap-words", type=int, default=24, help="Approximate overlap between chunks.")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def normalize_line(text: str) -> str:
    return " ".join(text.split()).strip()


def normalize_body_text(text: str) -> str:
    text = text.replace("\r", "\n")
    lines = [normalize_line(line) for line in text.splitlines()]
    cleaned: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank and cleaned:
                cleaned.append("")
            previous_blank = True
            continue
        cleaned.append(line)
        previous_blank = False
    return "\n".join(cleaned).strip()


def word_count(text: str) -> int:
    return len(text.split())


def token_count_matching(text: str, pattern: str) -> int:
    regex = re.compile(pattern)
    return sum(1 for token in text.split() if regex.search(token))


def select_docs(rows: list[dict[str, Any]], max_docs: int | None) -> list[dict[str, Any]]:
    selected = [row for row in rows if row.get("student_relevance") in KEEP_RELEVANCE]
    if max_docs is not None:
        selected = selected[:max_docs]
    return selected


def join_list(values: list[str]) -> str:
    cleaned = [normalize_line(str(value)) for value in values if normalize_line(str(value))]
    return ", ".join(cleaned)


def metadata_lines(doc: dict[str, Any]) -> list[str]:
    lines: list[str] = []

    field_specs = [
        ("Title", doc.get("title")),
        ("Document type", doc.get("document_type")),
        ("Section", doc.get("section")),
        ("Topic", doc.get("topic")),
        ("Issue date", doc.get("issue_date")),
        ("Academic year", doc.get("academic_year")),
        ("Audience", join_list(doc.get("audience", []))),
        ("Group/faculty", join_list(doc.get("group_or_faculty", []))),
        ("Exam part", doc.get("exam_part")),
        ("Session type", join_list(doc.get("session_type", []))),
        ("Important dates", join_list(doc.get("important_dates", []))),
        ("Fee details", join_list(doc.get("fee_details", []))),
        ("Portal links", join_list(doc.get("portal_links", []))),
        ("Download links", join_list(doc.get("download_links", []))),
    ]

    contact_info = doc.get("contact_info", {}) or {}
    emails = join_list(contact_info.get("emails", []))
    phones = join_list(contact_info.get("phones", []))
    if emails:
        field_specs.append(("Contact emails", emails))
    if phones:
        field_specs.append(("Contact phones", phones))

    for label, value in field_specs:
        normalized = normalize_line(str(value or ""))
        if normalized:
            lines.append(f"{label}: {normalized}")

    return lines


def compose_chunk_source_text(doc: dict[str, Any]) -> str:
    parts: list[str] = []
    meta = metadata_lines(doc)
    if meta:
        parts.append("\n".join(meta))

    body = normalize_body_text(str(doc.get("clean_body_text", "")))
    if body:
        parts.append(f"Body:\n{body}")

    return "\n\n".join(parts).strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [normalize_line(part) for part in parts if normalize_line(part)]


def split_oversized_unit(unit: str, max_words: int) -> list[str]:
    if word_count(unit) <= max_words:
        return [unit]

    pieces = split_sentences(unit)
    if len(pieces) <= 1:
        words = unit.split()
        return [" ".join(words[index : index + max_words]) for index in range(0, len(words), max_words)]

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for piece in pieces:
        piece_words = word_count(piece)
        if piece_words > max_words:
            if current:
                chunks.append(" ".join(current).strip())
                current = []
                current_words = 0
            chunks.extend(split_oversized_unit(piece, max_words))
            continue
        if current and current_words + piece_words > max_words:
            chunks.append(" ".join(current).strip())
            current = []
            current_words = 0
        current.append(piece)
        current_words += piece_words
    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def choose_doc_chunk_params(doc: dict[str, Any], default_target: int, default_max: int) -> tuple[int, int]:
    doc_type = str(doc.get("document_type") or "")
    if doc_type == "notification":
        return 150, 210
    if doc_type == "form":
        return 210, 280
    if doc_type == "datesheet":
        return 160, 220
    if doc_type == "results_document":
        return 220, 300
    if doc_type in {"model_paper", "scheme_of_studies"}:
        return 190, 250
    return default_target, default_max


def split_doc_into_units(doc: dict[str, Any], text: str) -> list[str]:
    doc_type = str(doc.get("document_type") or "")
    meta, _, body = text.partition("\n\nBody:\n")
    meta_units = [normalize_line(line) for line in meta.splitlines() if normalize_line(line)]
    body_units: list[str]

    if doc_type == "form":
        raw_lines = [normalize_line(line) for line in body.splitlines() if normalize_line(line)]
        body_units = raw_lines or split_sentences(body)
    elif doc_type == "notification":
        body_units = split_sentences(body) or [normalize_line(body)] if body else []
    elif doc_type == "datesheet":
        raw_lines = [normalize_line(line) for line in body.splitlines() if normalize_line(line)]
        body_units = raw_lines or split_sentences(body)
    else:
        body_units = split_sentences(body) or ([normalize_line(body)] if body else [])

    max_unit_words = 120 if doc_type in {"form", "results_document"} else 100
    normalized_units: list[str] = []
    for unit in meta_units + body_units:
        normalized_units.extend(split_oversized_unit(unit, max_unit_words))
    return normalized_units


def overlap_units(units: list[str], overlap_words: int) -> list[str]:
    if overlap_words <= 0:
        return []
    selected: list[str] = []
    total_words = 0
    for unit in reversed(units):
        selected.append(unit)
        total_words += word_count(unit)
        if total_words >= overlap_words:
            break
    return list(reversed(selected))


def should_emit_chunk(current_units: list[str], next_unit: str, target_words: int, max_words: int, min_words: int) -> bool:
    if not current_units:
        return False
    current_words = sum(word_count(unit) for unit in current_units)
    projected_words = current_words + word_count(next_unit)
    if projected_words > max_words and current_words >= min_words:
        return True
    return current_words >= target_words and projected_words > target_words + 20


def is_low_signal_chunk(doc: dict[str, Any], chunk_text: str) -> bool:
    doc_type = str(doc.get("document_type") or "")
    words = max(word_count(chunk_text), 1)
    alpha_tokens = token_count_matching(chunk_text, r"[A-Za-z]")
    digit_tokens = token_count_matching(chunk_text, r"\d")
    alpha_ratio = alpha_tokens / words
    digit_ratio = digit_tokens / words

    if doc_type == "results_document":
        if digit_ratio >= 0.55 and alpha_ratio <= 0.45:
            return True
        if alpha_tokens < 40 and digit_tokens > alpha_tokens * 2:
            return True

    return False


def chunk_doc(
    doc: dict[str, Any],
    target_words: int,
    max_words: int,
    min_words: int,
    overlap_words: int,
) -> list[dict[str, Any]]:
    source_text = compose_chunk_source_text(doc)
    if not source_text:
        return []

    target_words, max_words = choose_doc_chunk_params(doc, target_words, max_words)
    units = split_doc_into_units(doc, source_text)
    units = [unit for unit in units if unit]
    if not units:
        return []

    chunks: list[list[str]] = []
    current_units: list[str] = []
    for unit in units:
        if should_emit_chunk(current_units, unit, target_words, max_words, min_words):
            chunks.append(current_units[:])
            current_units = overlap_units(current_units, overlap_words)
        current_units.append(unit)

    if current_units:
        chunks.append(current_units[:])

    rows: list[dict[str, Any]] = []
    for chunk_index, chunk_units in enumerate(chunks, start=1):
        chunk_text = "\n\n".join(chunk_units).strip()
        if is_low_signal_chunk(doc, chunk_text):
            continue
        rows.append(
            {
                "chunk_id": f"biek::{doc['document_type']}::{chunk_index:03d}::{abs(hash(doc['source_url'])) % 10**8}",
                "source_url": doc["source_url"],
                "source_page": doc.get("source_page"),
                "title": doc.get("title"),
                "document_type": doc.get("document_type"),
                "section": doc.get("section"),
                "topic": doc.get("topic"),
                "student_relevance": doc.get("student_relevance"),
                "action_type": doc.get("action_type"),
                "issue_date": doc.get("issue_date"),
                "academic_year": doc.get("academic_year"),
                "chunk_index": chunk_index,
                "word_count": word_count(chunk_text),
                "text": chunk_text,
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path: Path, docs: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> None:
    summary = {
        "eligible_document_count": len(docs),
        "chunk_count": len(chunks),
        "documents_by_type": dict(Counter(doc["document_type"] for doc in docs)),
        "documents_by_section": dict(Counter(doc["section"] for doc in docs)),
        "documents_by_relevance": dict(Counter(doc["student_relevance"] for doc in docs)),
        "chunks_by_type": dict(Counter(chunk["document_type"] for chunk in chunks)),
        "chunks_by_section": dict(Counter(chunk["section"] for chunk in chunks)),
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.min_words > args.max_words:
        raise ValueError("--min-words cannot be greater than --max-words.")
    if args.target_words > args.max_words:
        raise ValueError("--target-words cannot be greater than --max-words.")

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)

    docs = select_docs(load_rows(input_path), args.max_docs)
    chunks: list[dict[str, Any]] = []
    for doc in docs:
        chunks.extend(
            chunk_doc(
                doc,
                target_words=args.target_words,
                max_words=args.max_words,
                min_words=args.min_words,
                overlap_words=args.overlap_words,
            )
        )

    write_jsonl(output_path, chunks)
    write_summary(summary_path, docs, chunks)

    print(f"Input docs: {input_path}")
    print(f"Eligible docs: {len(docs)}")
    print(f"Output chunks: {output_path}")
    print(f"Chunk count: {len(chunks)}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
