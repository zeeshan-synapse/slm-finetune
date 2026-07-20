#!/usr/bin/env python3
"""Deterministic BIEK roll-number result lookup from extracted gazette text."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTED_MANIFEST = PROJECT_ROOT / "data" / "runtime" / "biek_extracted_manifest.jsonl"
ROLL_MARKS_RE = re.compile(r"(?P<roll>\d{5,8})\s*\((?P<marks>\d+)(?:\+\s*(?P<bonus>\d+))?\)")
ROLL_RE = re.compile(r"\b(\d{5,8})\b")
YEAR_RE = re.compile(r"\b(20\d{2})\b")

GROUP_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Pre Engineering", re.compile(r"pre[-\s]?engineering|/SE/", re.I)),
    ("Pre Medical", re.compile(r"pre[-\s]?medical|/SM/", re.I)),
    ("Science General", re.compile(r"science\s+general|/SG/", re.I)),
    ("Commerce Regular", re.compile(r"commerce\s+regular|/CMR/", re.I)),
    ("Commerce Private", re.compile(r"commerce\s+private|/CMP/", re.I)),
    ("Humanities Regular", re.compile(r"humanities\s+regular|/HTR/", re.I)),
    ("Humanities Private", re.compile(r"humanities\s+private|/HTP/", re.I)),
    ("Home Economics", re.compile(r"home\s+econom|/HE/", re.I)),
    ("Special Candidate", re.compile(r"special\s+candidate|/SPC/", re.I)),
]


@dataclass
class ResultQuery:
    roll_number: str
    year: int | None = None
    session: str | None = None
    group: str | None = None
    exam_part: str | None = None


def normalize_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def title_case_label(value: str | None) -> str:
    text = normalize_text(value)
    return " ".join(part if part.isupper() else part.capitalize() for part in text.split())


def infer_year(*values: str) -> int | None:
    for value in values:
        match = YEAR_RE.search(value or "")
        if match:
            return int(match.group(1))
    return None


def infer_published_date(*values: str) -> str | None:
    for value in values:
        match = re.search(r"\((\d{2}-\d{2}-\d{4})\)", value or "")
        if match:
            day, month, year = match.group(1).split("-")
            return f"{year}-{month}-{day}"
    return None


def infer_session(*values: str) -> str | None:
    haystack = " ".join(values).lower()
    if "supplementary" in haystack or "supply" in haystack:
        return "Supplementary"
    if "annual" in haystack:
        return "Annual"
    return None


def infer_group(*values: str) -> str | None:
    haystack = " ".join(values)
    for label, pattern in GROUP_PATTERNS:
        if pattern.search(haystack):
            return label
    return None


def infer_exam_part(*values: str) -> str | None:
    haystack = " ".join(values).lower()
    if re.search(r"\bpart[\s\-]*ii\b|\bclass\s*xii\b", haystack):
        return "Part II"
    if re.search(r"\bpart[\s\-]*i\b|\bclass\s*xi\b", haystack):
        return "Part I"
    return None


def parse_query(question: str) -> ResultQuery | None:
    roll_match = ROLL_RE.search(question)
    if not roll_match:
        return None
    return ResultQuery(
        roll_number=roll_match.group(1),
        year=infer_year(question),
        session=infer_session(question),
        group=infer_group(question),
        exam_part=infer_exam_part(question),
    )


def read_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not DEFAULT_EXTRACTED_MANIFEST.exists():
        return rows
    with DEFAULT_EXTRACTED_MANIFEST.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            url = str(row.get("url") or "")
            title = str(row.get("title") or "")
            if "result" in f"{url} {title}".lower() and row.get("text_file"):
                rows.append(row)
    return rows


def find_roll_entries(row: dict[str, Any], roll_number: str) -> list[dict[str, Any]]:
    text_file = Path(str(row.get("text_file") or ""))
    if not text_file.exists():
        return []
    try:
        text = text_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    entries: list[dict[str, Any]] = []
    for match in ROLL_MARKS_RE.finditer(text):
        if match.group("roll") != roll_number:
            continue
        start = max(0, match.start() - 80)
        end = min(len(text), match.end() + 80)
        context = normalize_text(text[start:end])
        entries.append(
            {
                "roll_number": roll_number,
                "marks": match.group("marks"),
                "bonus": match.group("bonus"),
                "context": context,
            }
        )
    return entries


def document_metadata(row: dict[str, Any]) -> dict[str, Any]:
    title = normalize_text(str(row.get("title") or ""))
    url = str(row.get("url") or "")
    blob = f"{title} {url}"
    url_year = re.search(r"Result-(20\d{2})", url, re.I)
    return {
        "year": int(url_year.group(1)) if url_year else infer_year(blob),
        "published": infer_published_date(blob),
        "session": infer_session(blob),
        "group": infer_group(blob),
        "exam_part": infer_exam_part(blob),
        "source": title or url,
        "result_pdf": url,
    }


def score_match(query: ResultQuery, result: dict[str, Any]) -> int:
    score = 0
    for key in ("year", "session", "group", "exam_part"):
        requested = getattr(query, key)
        if not requested:
            continue
        actual = result.get(key)
        if str(actual or "").lower() == str(requested).lower():
            score += 3
        else:
            score -= 5
    return score


def lookup_results(query: ResultQuery) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for row in read_manifest():
        meta = document_metadata(row)
        for entry in find_roll_entries(row, query.roll_number):
            result = {**meta, **entry}
            result["_score"] = score_match(query, result)
            matches.append(result)
    matches.sort(key=lambda item: (item.get("_score", 0), item.get("year") or 0), reverse=True)
    if any([query.year, query.session, query.group, query.exam_part]):
        best = [item for item in matches if item.get("_score", 0) >= 0]
        return best[:5]
    return matches[:5]


def format_single_result(result: dict[str, Any]) -> str:
    parts = [
        f"I found a BIEK result for roll number {result.get('roll_number')}",
        f"for {result.get('year')}" if result.get("year") else "",
        result.get("session") or "",
        result.get("group") or "",
        result.get("exam_part") or "",
    ]
    lead = ", ".join(part for part in parts if part)
    marks = result.get("marks") or ""
    bonus = result.get("bonus") or ""
    mark_text = f"{marks}+ {bonus}" if bonus else marks
    details = []
    if mark_text:
        details.append(f"marks shown: {mark_text}")
    if result.get("published"):
        details.append(f"published on {result['published']}")
    sentence = lead + "."
    if details:
        sentence += " The record shows " + ", ".join(details) + "."
    if result.get("result_pdf"):
        sentence += f" Result PDF: {result['result_pdf']}"
    return sentence


def format_multiple_results(results: list[dict[str, Any]]) -> str:
    roll = results[0].get("roll_number")
    lines = [f"I found multiple BIEK result records for roll number {roll}."]
    for result in results:
        bits = [
            str(result.get("year") or ""),
            str(result.get("session") or ""),
            str(result.get("group") or ""),
            str(result.get("exam_part") or ""),
        ]
        bits = [bit for bit in bits if bit]
        marks = result.get("marks")
        bonus = result.get("bonus")
        mark_text = f"{marks}+ {bonus}" if bonus else str(marks or "")
        if mark_text:
            bits.append(f"marks {mark_text}")
        line = "- " + " | ".join(bits)
        if result.get("result_pdf"):
            line += f". Result PDF: {result['result_pdf']}"
        lines.append(line)
    lines.append("If you want, add the year, session, group, or part and I can narrow it to one exact result.")
    return "\n".join(lines)


def result_lookup_answer(question: str) -> str | None:
    query = parse_query(question)
    if query is None:
        return None
    results = lookup_results(query)
    if not results:
        return "I could not find a matching BIEK result record for that roll number in the available result data."
    if len(results) == 1 or any([query.year, query.session, query.group, query.exam_part]):
        return format_single_result(results[0])
    return format_multiple_results(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Look up BIEK result rows by roll number.")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()
    print(result_lookup_answer(args.question) or "No result lookup intent detected.")


if __name__ == "__main__":
    main()
