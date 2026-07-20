#!/usr/bin/env python3
"""Bootstrap BIEK nav coverage rows from data/runtime/nav_tree.json."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NAV_TREE_PATH = PROJECT_ROOT / "data" / "runtime" / "nav_tree.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "config" / "biek_nav_coverage.json"


def normalize_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", normalize_text(value).lower()).strip("-")


def normalize_url(value: str | None) -> str:
    url = str(value or "").strip()
    if url.startswith("https://biek.edu.pk"):
        url = url.replace("https://biek.edu.pk", "https://www.biek.edu.pk", 1)
    if url.startswith("http://"):
        url = "https://" + url[len("http://") :]
    return url


def item_id(label: str, parent_chain: list[str]) -> str:
    parts = [slugify(part) for part in parent_chain + [label] if slugify(part)]
    return "biek-nav::" + "::".join(parts)


def infer_handling_mode(label: str, url: str, level: int, has_children: bool) -> tuple[str, str, bool, str]:
    low = f"{label} {url}".lower()
    if any(token in low for token in ("tender", "tax payer", "ftn no", "active tax payer")):
        return "exclude", "low", False, "Admin or non-student nav item; excluded from student-facing coverage."
    if any(token in low for token in ("result declaration", "last results", "results.asp", "roll number")):
        return "special_lookup", "high", False, "Results-style item likely needs a dedicated lookup or routing path."
    if has_children or level == 0 or url.lower().endswith("default.asp#"):
        return "container_summary", "medium", True, "Section-style nav item; summarize child items."
    if any(token in low for token in (".pdf", ".jpg", ".jpeg", ".png", ".mp4")):
        return "document_summary", "high", True, "Direct downloadable/static resource; summarize content and provide the link."
    if any(
        token in low
        for token in (
            "authorized banks",
            "circular",
            "notification",
            "contact",
            "affiliated colleges",
            "chairman",
            "board members",
            "committees",
            "history",
            "press release",
            "statistics",
            "mcqs",
            "forms",
            "date sheet",
            "datesheet",
        )
    ):
        return "info_page", "high", True, "Public informational page that should answer from extracted page content."
    return "info_page", "medium", True, "Defaulted from public nav structure for content-backed answers."


def flatten_nodes(nodes: list[dict[str, Any]], *, parent: str | None = None, level: int = 0, chain: list[str] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    chain = chain or []
    for node in nodes:
        label = normalize_text(node.get("label"))
        if not label:
            continue
        url = normalize_url(node.get("url"))
        children = [child for child in node.get("children", []) if isinstance(child, dict)]
        handling_mode, relevance, should_index, notes = infer_handling_mode(label, url, level, bool(children))
        row = {
            "id": item_id(label, chain),
            "label": label,
            "url": url,
            "parent": parent,
            "level": level,
            "handling_mode": handling_mode,
            "student_relevance": relevance,
            "should_index": should_index,
            "notes": notes,
        }
        rows.append(row)
        rows.extend(flatten_nodes(children, parent=label, level=level + 1, chain=chain + [label]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap a BIEK nav coverage map from data/runtime/nav_tree.json.")
    parser.add_argument("--input", default=str(DEFAULT_NAV_TREE_PATH), help="Path to nav tree JSON.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Path to write coverage JSON.")
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output)
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = flatten_nodes(payload.get("items") or [])
    result = {
        "site": payload.get("site") or "biek",
        "source_nav_file": str(source),
        "source_url": payload.get("url"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_item_count": len(rows),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Source nav tree: {source.resolve()}")
    print(f"Output coverage file: {output.resolve()}")
    print(f"Total coverage rows: {len(rows)}")


if __name__ == "__main__":
    main()
