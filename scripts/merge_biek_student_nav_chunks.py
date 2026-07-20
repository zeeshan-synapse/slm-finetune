#!/usr/bin/env python3
"""Merge BIEK student chunks with nav-backed coverage chunks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_STUDENT_CHUNKS = Path("data/runtime/biek_student_chunks.jsonl")
DEFAULT_NAV_CHUNKS = Path("data/runtime/biek_nav_chunks.jsonl")
DEFAULT_OUTPUT = Path("data/runtime/biek_student_plus_nav_chunks.jsonl")
DEFAULT_SUMMARY = Path("data/runtime/biek_student_plus_nav_chunks_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge BIEK student-focused chunks with nav-backed chunks."
    )
    parser.add_argument("--student-input", default=str(DEFAULT_STUDENT_CHUNKS))
    parser.add_argument("--nav-input", default=str(DEFAULT_NAV_CHUNKS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY))
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object.")
            if not row.get("text"):
                raise ValueError(f"{path}:{line_number} is missing text.")
            rows.append(row)
    return rows


def stable_chunk_id(row: dict[str, Any], source_name: str, index: int) -> str:
    chunk_id = str(row.get("chunk_id") or "").strip()
    if chunk_id:
        return chunk_id
    return f"biek::{source_name}::{index:06d}"


def merge_rows(
    student_rows: list[dict[str, Any]],
    nav_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    merged: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    duplicate_counts: Counter[str] = Counter()

    for source_name, rows in (("student", student_rows), ("nav", nav_rows)):
        for index, row in enumerate(rows, start=1):
            merged_row = dict(row)
            chunk_id = stable_chunk_id(merged_row, source_name, index)
            if chunk_id in seen_ids:
                duplicate_counts[source_name] += 1
                chunk_id = f"{chunk_id}::{source_name}::{index:06d}"
            merged_row["chunk_id"] = chunk_id
            merged_row["kb_chunk_source"] = source_name
            seen_ids.add(chunk_id)
            merged.append(merged_row)

    return merged, duplicate_counts


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(
    path: Path,
    *,
    student_input: Path,
    nav_input: Path,
    output: Path,
    student_rows: list[dict[str, Any]],
    nav_rows: list[dict[str, Any]],
    merged_rows: list[dict[str, Any]],
    duplicate_counts: Counter[str],
) -> None:
    by_source = Counter(row.get("kb_chunk_source", "unknown") for row in merged_rows)
    by_document_type = Counter(str(row.get("document_type") or "unknown") for row in merged_rows)
    by_nav_label = Counter(
        str(row.get("nav_label"))
        for row in merged_rows
        if row.get("kb_chunk_source") == "nav" and row.get("nav_label")
    )
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "student_input": str(student_input),
        "nav_input": str(nav_input),
        "output": str(output),
        "student_chunks": len(student_rows),
        "nav_chunks": len(nav_rows),
        "merged_chunks": len(merged_rows),
        "chunks_by_source": dict(sorted(by_source.items())),
        "chunks_by_document_type": dict(sorted(by_document_type.items())),
        "top_nav_labels": dict(by_nav_label.most_common(20)),
        "duplicate_chunk_ids_rewritten": dict(sorted(duplicate_counts.items())),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    student_input = Path(args.student_input)
    nav_input = Path(args.nav_input)
    output = Path(args.output)
    summary_output = Path(args.summary_output)

    student_rows = load_jsonl(student_input)
    nav_rows = load_jsonl(nav_input)
    merged_rows, duplicate_counts = merge_rows(student_rows, nav_rows)

    write_jsonl(output, merged_rows)
    write_summary(
        summary_output,
        student_input=student_input,
        nav_input=nav_input,
        output=output,
        student_rows=student_rows,
        nav_rows=nav_rows,
        merged_rows=merged_rows,
        duplicate_counts=duplicate_counts,
    )

    print(f"Student chunks: {len(student_rows)}")
    print(f"Nav chunks: {len(nav_rows)}")
    print(f"Merged chunks: {len(merged_rows)}")
    print(f"Output: {output.resolve()}")
    print(f"Summary: {summary_output.resolve()}")
    if duplicate_counts:
        print(f"Duplicate chunk ids rewritten: {dict(duplicate_counts)}")


if __name__ == "__main__":
    main()
