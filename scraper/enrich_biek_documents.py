from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = PROJECT_ROOT / "data" / "runtime"

DEFAULT_INPUT_MANIFEST = RUNTIME_DIR / "biek_extracted_manifest.jsonl"
DEFAULT_OUTPUT_JSONL = RUNTIME_DIR / "biek_enriched_documents.jsonl"
DEFAULT_OUTPUT_SUMMARY = RUNTIME_DIR / "biek_enriched_summary.json"

DATE_RANGE_RE = re.compile(
    r"\b(?:MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY)?\s*"
    r"(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\s*"
    r"(?:TO|-)\s*"
    r"(?:MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY)?\s*"
    r"(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b",
    re.IGNORECASE,
)
SINGLE_DATE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
YEAR_SPAN_RE = re.compile(r"\b(20\d{2})\s*[-/]\s*(20\d{2})\b")
YEAR_RE = re.compile(r"\b20\d{2}\b")
FEE_RE = re.compile(r"(?:@?\s*)?Rs\.?\s*\d[\d,]*(?:/-)?(?:\s*\+\s*Rs\.?\s*\d[\d,]*(?:/-)?)?", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s)]+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\b(?:\+?\d[\d\s-]{6,}\d)\b")

GROUP_PATTERNS = {
    "science": ("science", "pre-medical", "pre engineering", "pre-engineering", "science general"),
    "commerce": ("commerce",),
    "humanities": ("humanities", "arts"),
    "medical technology": ("medical technology",),
    "home economics": ("home economics",),
    "computer science": ("computer science",),
}
AUDIENCE_PATTERNS = {
    "private_colleges": ("private colleges", "private recognized", "private institution", "private candidates"),
    "government_colleges": ("govt. colleges", "government colleges", "government institutions"),
    "affiliated_colleges": ("affiliated", "recognized / affiliated", "recognized/affiliated"),
    "students": ("instruction for students", "students", "candidate", "candidates"),
    "principals": ("the principals", "principal"),
}


def load_rows(path: Path) -> list[dict]:
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


def latest_successful_rows(rows: list[dict]) -> list[dict]:
    by_url: dict[str, dict] = {}
    for row in rows:
        url = str(row.get("url") or "").strip()
        if not url:
            continue
        if not str(row.get("extraction_status", "")).startswith("ok"):
            continue
        by_url[url] = row
    return list(by_url.values())


def read_body_text(path_str: str) -> str:
    text = Path(path_str).read_text(encoding="utf-8")
    parts = text.split("\n\n", 1)
    return parts[1].strip() if len(parts) == 2 else text.strip()


def normalize_spaces(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def infer_section(url: str, discovered_from: str | None) -> str:
    combined = f"{url} {discovered_from or ''}".lower()
    if "/notifications/" in combined or "notifications.asp" in combined:
        return "notifications"
    if "/datesheet/" in combined or "datesheet" in combined:
        return "datesheet"
    if "affcoll" in combined or "affiliation" in combined:
        return "affiliation"
    if "contact" in combined:
        return "contact"
    if "result" in combined:
        return "results"
    if "tender" in combined:
        return "tenders"
    if "/online_forms/" in combined or "/allforms.asp" in combined or re.search(r"(?:^|[/_\-\s])forms?(?:$|[./_\-\s])", combined):
        return "forms"
    if "press" in combined:
        return "press_release"
    if "modelpaper" in combined or "model paper" in combined:
        return "model_paper"
    return "general"


def infer_document_type(url: str, title: str, body_text: str, section: str) -> str:
    searchable = f"{url} {title} {body_text[:2500]}".lower()
    if section == "contact":
        return "contact_info"
    if section == "affiliation":
        return "affiliation_list"
    if section == "datesheet":
        return "datesheet"
    if section == "tenders":
        return "tender"
    if section == "results":
        return "results_document"
    if section == "model_paper":
        return "model_paper"
    if "e-sheet" in searchable or "omr" in searchable or "answer sheet" in searchable:
        return "exam_material"
    if "scheme of studies" in searchable:
        return "scheme_of_studies"
    if "notification" in searchable or section == "notifications":
        return "notification"
    if "form" in searchable or section == "forms":
        return "form"
    if section == "press_release":
        return "press_release"
    return "general_page"


def infer_action_type(title: str, body_text: str, url: str) -> str | None:
    searchable = f"{title} {body_text[:3000]} {url}".lower()
    checks = [
        ("exam_form_submission", ("submission of examination forms", "examination forms", "exam form")),
        ("enrolment", ("enrolment", "enrollment", "enrolment forms", "enrollment forms")),
        ("fee_payment", ("fee", "fees", "kuick pay", "voucher")),
        ("verification", ("verification", "verify")),
        ("certificate", ("certificate form", "migration certificate", "provisional certificate")),
        ("scrutiny", ("scrutiny",)),
        ("duplicate_document", ("duplicate", "cancelation", "cancellation")),
        ("affiliation", ("affiliation", "affiliated colleges")),
        ("result_publication", ("result gazette", "result", "with-held", "ufm")),
        ("datesheet_release", ("date sheet", "datesheet")),
    ]
    for label, tokens in checks:
        if any(token in searchable for token in tokens):
            return label
    return None


def infer_form_action_type(title: str, body_text: str, url: str) -> str | None:
    searchable = f"{title} {body_text[:2000]} {url}".lower()
    checks = [
        ("scrutiny", ("scrutiny",)),
        ("verification", ("verification",)),
        ("migration_certificate", ("migration certificate", "migration form")),
        ("provisional_certificate", ("provisional certificate", "provisional certification")),
        ("certificate_issue", ("certificate form", "original certificate")),
        ("registration", ("registration form",)),
        ("enrolment", ("enrolment form", "enrollment form")),
        ("duplicate_document", ("duplicate", "cancelation", "cancellation")),
        ("exam_form_submission", ("examination form", "exam form")),
        ("fee_voucher", ("voucher", "challan")),
    ]
    for label, tokens in checks:
        if any(token in searchable for token in tokens):
            return label
    return None


def normalize_fee_value(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip()
    cleaned = cleaned.replace("@ ", "").replace("@", "")
    cleaned = re.sub(r"rs\.?", "Rs.", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"Rs\.\s*", "Rs. ", cleaned)
    cleaned = cleaned.replace(" /-", "/-")
    return cleaned.strip()


def extract_issue_date(title: str, body_text: str) -> str | None:
    searchable = f"{title}\n{body_text[:2000]}"
    match = re.search(r"date[:\s]+(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", searchable, re.IGNORECASE)
    if match:
        return match.group(1)
    match = SINGLE_DATE_RE.search(searchable)
    return match.group(0) if match else None


def extract_academic_year(title: str, body_text: str, url: str) -> str | None:
    searchable = f"{title} {body_text[:1000]} {url}"
    span = YEAR_SPAN_RE.search(searchable)
    if span:
        return f"{span.group(1)}-{span.group(2)}"
    years = YEAR_RE.findall(searchable)
    if years:
        return years[0]
    return None


def extract_group_or_faculty(title: str, body_text: str) -> list[str]:
    searchable = f"{title} {body_text[:4000]}".lower()
    found: list[str] = []
    for label, tokens in GROUP_PATTERNS.items():
        if any(token in searchable for token in tokens):
            found.append(label)
    return found


def extract_audience(title: str, body_text: str) -> list[str]:
    searchable = f"{title} {body_text[:3000]}".lower()
    found: list[str] = []
    for label, tokens in AUDIENCE_PATTERNS.items():
        if any(token in searchable for token in tokens):
            found.append(label)
    return found


def extract_exam_part(title: str, body_text: str) -> str | None:
    searchable = f"{title} {body_text[:3000]}".lower()
    if "part-i" in searchable or "part i" in searchable:
        return "part_i"
    if "part-ii" in searchable or "part ii" in searchable:
        return "part_ii"
    return None


def extract_session_type(title: str, body_text: str) -> list[str]:
    searchable = f"{title} {body_text[:3000]}".lower()
    found: list[str] = []
    for label in ("annual", "supplementary", "fresh", "regular", "private"):
        if label in searchable:
            found.append(label)
    return found


def extract_important_dates(body_text: str) -> list[str]:
    results: list[str] = []
    seen: set[str] = set()
    for start, end in DATE_RANGE_RE.findall(body_text):
        label = f"{start} to {end}"
        if label not in seen:
            seen.add(label)
            results.append(label)
    for date in SINGLE_DATE_RE.findall(body_text):
        if date not in seen:
            seen.add(date)
            results.append(date)
    return results[:20]


def extract_fee_details(body_text: str) -> list[str]:
    seen: set[str] = set()
    results: list[str] = []
    for match in FEE_RE.findall(body_text):
        cleaned = re.sub(r"\s+", " ", match).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            results.append(cleaned)
    return results[:20]


def filter_fee_details(section: str, document_type: str, values: list[str]) -> list[str]:
    if document_type in {"model_paper", "scheme_of_studies", "general_page"}:
        return []
    if section == "forms":
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = normalize_fee_value(value)
            key = cleaned.lower().replace(" ", "")
            if not cleaned or key in seen:
                continue
            seen.add(key)
            normalized.append(cleaned)
        return normalized[:10]
    if section == "notifications":
        return values
    return values[:5]


def extract_links(body_text: str, source_url: str) -> tuple[list[str], list[str]]:
    seen: set[str] = set()
    portal_links: list[str] = []
    download_links: list[str] = []

    for match in URL_RE.findall(body_text):
        cleaned = match.rstrip(".,);")
        if cleaned in seen:
            continue
        seen.add(cleaned)
        lowered = cleaned.lower()
        if ".pdf" in lowered:
            download_links.append(cleaned)
        elif "portal" in lowered or "institute" in lowered or "biek.edu.pk" in lowered:
            portal_links.append(cleaned)

    if source_url.lower().endswith(".pdf"):
        download_links.insert(0, source_url)

    return portal_links[:20], download_links[:20]


def extract_contact_info(body_text: str) -> dict[str, list[str]]:
    emails = list(dict.fromkeys(EMAIL_RE.findall(body_text)))[:10]
    first_chunk = body_text[:1200].lower()
    phones: list[str] = []
    if any(token in first_chunk for token in ("phone", "phones", "contact", "tel", "fax")):
        cleaned_candidates = []
        for match in PHONE_RE.findall(body_text[:2000]):
            cleaned = re.sub(r"\s+", " ", match).strip()
            digits_only = "".join(ch for ch in cleaned if ch.isdigit())
            if len(digits_only) < 7:
                continue
            if digits_only in {"0123456789", "1234567891011"}:
                continue
            cleaned_candidates.append(cleaned)
        phones = list(dict.fromkeys(cleaned_candidates))[:10]
    return {
        "emails": emails,
        "phones": phones,
    }


def extract_form_group_or_faculty(title: str, body_text: str) -> list[str]:
    title_lower = title.lower()
    first_chunk = body_text[:1200].lower()
    explicit_zone = f"{title_lower} {first_chunk}"
    found: list[str] = []
    for label, tokens in GROUP_PATTERNS.items():
        if any(token in explicit_zone for token in tokens):
            found.append(label)
    return found


def extract_form_exam_part(title: str, body_text: str) -> str | None:
    title_lower = title.lower()
    first_chunk = body_text[:1200].lower()
    explicit_zone = f"{title_lower} {first_chunk}"
    if any(
        token in explicit_zone
        for token in (
            "hsc part-ii",
            "hsc part ii",
            "intermediate part-ii",
            "intermediate part ii",
            "part-ii annual examinations",
            "part ii annual examinations",
        )
    ):
        return "part_ii"
    if any(
        token in explicit_zone
        for token in (
            "hsc part-i",
            "hsc part i",
            "intermediate part-i",
            "intermediate part i",
            "part-i annual examinations",
            "part i annual examinations",
        )
    ):
        return "part_i"
    return None


def extract_form_audience(title: str, body_text: str) -> list[str]:
    title_lower = title.lower()
    first_chunk = body_text[:1600].lower()
    explicit_zone = f"{title_lower} {first_chunk}"
    found: list[str] = []
    for label, tokens in AUDIENCE_PATTERNS.items():
        if any(token in explicit_zone for token in tokens):
            found.append(label)
    return found


def infer_topic(title: str, document_type: str, action_type: str | None) -> str:
    if action_type:
        return action_type.replace("_", " ")
    if document_type == "datesheet":
        return "exam schedule"
    if document_type == "contact_info":
        return "contact information"
    if document_type == "affiliation_list":
        return "affiliated colleges"
    return normalize_spaces(title)


def infer_student_relevance(
    *,
    section: str,
    document_type: str,
    action_type: str | None,
    title: str,
    body_text: str,
) -> str:
    searchable = f"{title} {body_text[:2500]}".lower()

    if section == "tenders" or document_type == "tender":
        return "exclude"

    if document_type in {"notification", "datesheet", "affiliation_list", "contact_info"}:
        return "high_relevance"

    if document_type == "results_document":
        return "high_relevance"

    if document_type == "exam_material":
        return "medium_relevance"

    if document_type == "form":
        if action_type in {
            "exam_form_submission",
            "enrolment",
            "registration",
            "fee_voucher",
            "verification",
            "certificate_issue",
            "scrutiny",
            "migration_certificate",
            "provisional_certificate",
        }:
            return "high_relevance"
        return "medium_relevance"

    if document_type in {"model_paper", "scheme_of_studies", "press_release"}:
        return "medium_relevance"

    if document_type == "general_page":
        return "low_relevance"

    if any(token in searchable for token in ("fee schedule", "last date", "date sheet", "enrolment", "registration form")):
        return "high_relevance"

    return "low_relevance"


def strip_non_applicable_fields(section: str, document_type: str, data: dict) -> dict:
    cleaned = dict(data)

    if document_type in {"model_paper", "scheme_of_studies", "general_page", "exam_material"}:
        cleaned["fee_details"] = []
        cleaned["contact_info"] = {"emails": [], "phones": []}

    if document_type == "model_paper":
        cleaned["action_type"] = None
        cleaned["important_dates"] = []
        cleaned["portal_links"] = []
        cleaned["session_type"] = []
        cleaned["exam_part"] = None
        cleaned["topic"] = normalize_spaces(cleaned["title"])

    if document_type == "scheme_of_studies":
        cleaned["action_type"] = None
        cleaned["important_dates"] = []
        cleaned["topic"] = "scheme of studies"

    if document_type == "exam_material":
        cleaned["action_type"] = None
        cleaned["important_dates"] = []
        cleaned["portal_links"] = []
        cleaned["session_type"] = []
        cleaned["exam_part"] = None
        cleaned["topic"] = normalize_spaces(cleaned["title"])

    if section == "forms":
        cleaned["important_dates"] = []
        cleaned["contact_info"] = {
            "emails": cleaned["contact_info"].get("emails", []),
            "phones": [],
        }

    return cleaned


def enrich_row(row: dict) -> dict:
    text_file = row["text_file"]
    body_text = normalize_spaces(read_body_text(text_file))
    source_url = str(row.get("url") or "")
    source_page = row.get("discovered_from")
    title = normalize_spaces(str(row.get("title") or source_url))

    section = infer_section(source_url, source_page)
    document_type = infer_document_type(source_url, title, body_text, section)
    if section == "forms":
        action_type = infer_form_action_type(title, body_text, source_url)
    elif section == "notifications":
        action_type = infer_action_type(title, body_text, source_url)
    else:
        action_type = None
    issue_date = extract_issue_date(title, body_text)
    academic_year = extract_academic_year(title, body_text, source_url)
    audience = extract_audience(title, body_text)
    groups = extract_group_or_faculty(title, body_text)
    exam_part = extract_exam_part(title, body_text)
    session_type = extract_session_type(title, body_text)
    important_dates = extract_important_dates(body_text)
    fee_details = filter_fee_details(section, document_type, extract_fee_details(body_text))
    portal_links, download_links = extract_links(body_text, source_url)
    contact_info = extract_contact_info(body_text)
    topic = infer_topic(title, document_type, action_type)
    student_relevance = infer_student_relevance(
        section=section,
        document_type=document_type,
        action_type=action_type,
        title=title,
        body_text=body_text,
    )

    if section == "forms":
        audience = extract_form_audience(title, body_text)
        groups = extract_form_group_or_faculty(title, body_text)
        exam_part = extract_form_exam_part(title, body_text)
        portal_links = []

    parsed = urlparse(source_url)
    enriched = {
        "source_url": source_url,
        "source_page": source_page,
        "source_kind": row.get("kind"),
        "extractor": row.get("extractor"),
        "text_file": text_file,
        "text_length": len(body_text),
        "page_count": row.get("page_count"),
        "title": title,
        "document_type": document_type,
        "section": section,
        "issue_date": issue_date,
        "academic_year": academic_year,
        "audience": audience,
        "group_or_faculty": groups,
        "exam_part": exam_part,
        "session_type": session_type,
        "topic": topic,
        "student_relevance": student_relevance,
        "action_type": action_type,
        "important_dates": important_dates,
        "fee_details": fee_details,
        "portal_links": portal_links,
        "download_links": download_links,
        "contact_info": contact_info,
        "source_path": parsed.path,
        "clean_body_text": body_text,
    }
    return strip_non_applicable_fields(section, document_type, enriched)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path: Path, rows: list[dict]) -> None:
    summary = {
        "document_count": len(rows),
        "document_types": dict(Counter(row["document_type"] for row in rows)),
        "sections": dict(Counter(row["section"] for row in rows)),
        "extractors": dict(Counter(row["extractor"] for row in rows)),
        "student_relevance": dict(Counter(row["student_relevance"] for row in rows)),
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich extracted BIEK documents into structured records.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_MANIFEST), help="Path to extracted manifest JSONL.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_JSONL), help="Path to enriched JSONL output.")
    parser.add_argument("--summary-output", default=str(DEFAULT_OUTPUT_SUMMARY), help="Path to summary JSON output.")
    parser.add_argument("--max-docs", type=int, default=None, help="Optional limit for debugging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)

    rows = load_rows(input_path)
    successful = latest_successful_rows(rows)
    if args.max_docs is not None:
        successful = successful[: args.max_docs]

    enriched = [enrich_row(row) for row in successful]
    write_jsonl(output_path, enriched)
    write_summary(summary_path, enriched)

    print(f"Input manifest: {input_path}")
    print(f"Successful unique docs: {len(successful)}")
    print(f"Enriched output: {output_path}")
    print(f"Summary output: {summary_path}")


if __name__ == "__main__":
    main()
