#!/usr/bin/env python3
import json
import re
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_DIR / "data" / "raw" / "cleaned-data"
OUTPUT_DIR = PROJECT_DIR / "data" / "knowledge-base"
OUTPUT_PATH = OUTPUT_DIR / "source_docs.jsonl"

SKIP_EXACT = {
    "skip to content",
    "home",
    "about us",
    "product",
    "our services",
    "industries",
    "blogs",
    "contact us",
    "get in touch",
    "get started",
    "get service now",
    "request a demo",
    "request a live demo",
    "let’s talk",
    "let's talk",
    "talk to the team",
    "translate »",
    "read more »",
    "contact us",
    "frequently asked questions",
    "have questions?",
    "general inquiries",
    "ready to get started?",
    "why choose opira.io™",
    "why choose agentic bot™",
    "the automation power-stack",
    "the voice agent power-stack",
    "capability",
    "capabilities",
    "feature",
    "features",
    "platform",
    "layer",
    "business value",
    "key elements",
    "what it handles",
    "area",
    "before",
    "with synapse tech inc.",
    "traditional chatbots",
    "traditional ats",
    "traditional approach",
    "traditional automation (zapier/legacy)",
}

SKIP_PREFIXES = (
    "url:",
    "request ",
    "explore ",
    "build ",
    "start ",
    "talk ",
    "translate",
    "read more",
    "what we deliver",
    "explore our ",
)

NAV_OR_CTA_RE = re.compile(
    r"^(request|explore|build|start|talk|read more|translate)\b", re.IGNORECASE
)
METRIC_FRAGMENT_RE = re.compile(r"^(?:[0-9]+|[0-9]+x|[0-9]+%|/[0-9]+|[%x]|✅|❌|⚠️|✔.*)$")
ONLY_SYMBOLS_RE = re.compile(r"^[^A-Za-z0-9]+$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
BREAK_MARKERS = (
    "don't take our word for it",
    "don’t take our word for it",
    "take theirs",
    "have questions?",
    "frequently asked questions",
    "faqs about ",
    "faqs:",
    "q1:",
)
SKIP_CONTAINS = (
    "lorem ipsum",
    "aisev",
)


def classify_page_type(stem: str) -> str:
    if stem.startswith("product-"):
        return "product"
    if stem.startswith("services-"):
        return "service"
    if stem.startswith("industries-"):
        return "industry"
    if stem == "about-us":
        return "about"
    if stem == "contact-us":
        return "contact"
    if stem == "home":
        return "home"
    if stem in {"blogs", "product", "services", "industries"}:
        return "index"
    return "article"


def derive_title(title_line: str, stem: str) -> str:
    if not title_line:
        return stem.replace("-", " ").title()
    title = title_line.strip()
    title = re.sub(r"\s*[-|]\s*Synapse Tech Inc\s*$", "", title, flags=re.IGNORECASE)
    return title.strip() or stem.replace("-", " ").title()


def normalize_line(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("React Flow=", "React Flow")
    text = text.replace("AI agents having", "AI agents with")
    return text


def should_skip_line(text: str) -> bool:
    low = text.lower().strip()
    if not low:
        return True
    if any(marker in low for marker in SKIP_CONTAINS):
        return True
    if low in SKIP_EXACT:
        return True
    if any(low.startswith(prefix) for prefix in SKIP_PREFIXES):
        return True
    if NAV_OR_CTA_RE.match(low) and len(low.split()) <= 6:
        return True
    if METRIC_FRAGMENT_RE.match(text.strip()):
        return True
    if ONLY_SYMBOLS_RE.match(text.strip()):
        return True
    if low in {"yes", "no"}:
        return True
    return False


def is_substantial_line(text: str) -> bool:
    if EMAIL_RE.match(text):
        return True
    words = text.split()
    if len(words) >= 5:
        return True
    # Keep shorter lines when they clearly define a concept.
    return ":" in text and len(words) >= 3


def clean_doc(path: Path) -> dict | None:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    if not raw_lines:
        return None

    title_line = raw_lines[0].strip() if raw_lines else ""
    url_line = raw_lines[1].strip() if len(raw_lines) > 1 else ""
    url = ""
    if url_line.lower().startswith("url:"):
        url = url_line.split(":", 1)[1].strip()

    kept_lines: list[str] = []
    seen_lines: set[str] = set()

    for raw in raw_lines[2:]:
        line = normalize_line(raw)
        low = line.lower().strip()
        if any(marker in low for marker in BREAK_MARKERS):
            break
        if should_skip_line(line):
            continue
        if not is_substantial_line(line):
            continue
        if line in seen_lines:
            continue
        seen_lines.add(line)
        kept_lines.append(line)

    clean_text = "\n".join(kept_lines).strip()
    if len(clean_text) < 200:
        return None

    title = derive_title(title_line, path.stem)
    return {
        "doc_id": path.stem,
        "source_file": path.name,
        "title": title,
        "url": url,
        "page_type": classify_page_type(path.stem),
        "clean_text": clean_text,
    }


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Missing input directory: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    docs: list[dict] = []
    skipped: list[str] = []

    for path in sorted(INPUT_DIR.glob("*.txt")):
        doc = clean_doc(path)
        if doc is None:
            skipped.append(path.name)
            continue
        docs.append(doc)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Input files: {len(list(INPUT_DIR.glob('*.txt')))}")
    print(f"Written docs: {len(docs)}")
    print(f"Skipped files: {len(skipped)}")
    if skipped:
        print("Skipped:", ", ".join(skipped))
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
