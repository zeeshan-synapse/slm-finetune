from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from site_profiles import BIEK_PROFILE, DEFAULT_HEADERS


DEFAULT_URLS = [
    "https://www.biek.edu.pk/default.asp",
    "https://www.biek.edu.pk/notifications.asp",
    "https://www.biek.edu.pk/Allforms.asp",
    "https://www.biek.edu.pk/Datesheet.asp",
    "https://www.biek.edu.pk/affcoll.asp",
    "https://www.biek.edu.pk/LastResults.asp",
    "https://www.biek.edu.pk/Permissionforms.asp",
]
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "runtime"
    / "biek_static_crawlability_audit.json"
)


def normalize_link(current_url: str, href: str) -> str:
    return urljoin(current_url, href.strip())


def short_sample(values: list[str], limit: int = 5) -> list[str]:
    seen: list[str] = []
    for value in values:
        clean = " ".join(value.split()).strip()
        if not clean or clean in seen:
            continue
        seen.append(clean)
        if len(seen) >= limit:
            break
    return seen


def extract_page_report(url: str, timeout: int) -> dict:
    response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    response.raise_for_status()

    html = response.text
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else ""

    normal_links: list[str] = []
    pdf_links: list[str] = []
    hash_links: list[str] = []
    javascript_links: list[str] = []
    suspicious_elements: list[str] = []

    for tag in soup.find_all(["a", "button"]):
        href = tag.get("href", "") if tag.name == "a" else ""
        onclick = tag.get("onclick", "")
        text = " ".join(tag.get_text(" ", strip=True).split())

        if href:
            absolute = normalize_link(response.url, href)
            lower_href = href.lower().strip()
            lower_absolute = absolute.lower()

            if lower_href == "#":
                hash_links.append(f"text={text or '[no text]'} href=#")
            elif lower_href.startswith("javascript:"):
                javascript_links.append(f"text={text or '[no text]'} href={href}")
            else:
                normal_links.append(absolute)
                if ".pdf" in lower_absolute:
                    pdf_links.append(absolute)

        if onclick:
            suspicious_elements.append(
                f"{tag.name} text={text or '[no text]'} onclick={onclick[:120]}"
            )

    has_postback = "__doPostBack" in html
    if has_postback:
        suspicious_elements.append("__doPostBack detected in raw HTML")

    assessment = classify_page(
        pdf_links=pdf_links,
        hash_links=hash_links,
        javascript_links=javascript_links,
        has_postback=has_postback,
        suspicious_elements=suspicious_elements,
    )

    return {
        "url": response.url,
        "status_code": response.status_code,
        "title": title,
        "normal_href_count": len(normal_links),
        "pdf_link_count": len(pdf_links),
        "hash_href_count": len(hash_links),
        "javascript_href_count": len(javascript_links),
        "has_postback": has_postback,
        "has_onclick_handlers": bool(suspicious_elements),
        "pdf_link_samples": short_sample(pdf_links),
        "suspicious_samples": short_sample(hash_links + javascript_links + suspicious_elements),
        "assessment": assessment,
    }


def classify_page(
    *,
    pdf_links: list[str],
    hash_links: list[str],
    javascript_links: list[str],
    has_postback: bool,
    suspicious_elements: list[str],
) -> str:
    if has_postback:
        return "postback_detected"
    if javascript_links:
        return "suspicious_js"
    if hash_links and not pdf_links:
        return "needs_manual_review"
    if suspicious_elements and not pdf_links:
        return "needs_manual_review"
    return "static_ok"


def summarize_report(report: dict) -> str:
    return (
        f"[{report['assessment']}] {report['url']} | "
        f"status={report['status_code']} | "
        f"hrefs={report['normal_href_count']} | "
        f"pdfs={report['pdf_link_count']} | "
        f"hash={report['hash_href_count']} | "
        f"js={report['javascript_href_count']} | "
        f"postback={report['has_postback']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether important BIEK pages are statically crawlable."
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="Optional URL to audit. Repeat to inspect multiple pages.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to write the JSON audit report.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=BIEK_PROFILE.timeout,
        help="HTTP timeout in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    urls = args.url or DEFAULT_URLS
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reports: list[dict] = []
    for url in urls:
        try:
            report = extract_page_report(url, timeout=args.timeout)
        except Exception as exc:
            report = {
                "url": url,
                "status_code": None,
                "title": "",
                "normal_href_count": 0,
                "pdf_link_count": 0,
                "hash_href_count": 0,
                "javascript_href_count": 0,
                "has_postback": False,
                "has_onclick_handlers": False,
                "pdf_link_samples": [],
                "suspicious_samples": [str(exc)],
                "assessment": "needs_manual_review",
            }
        reports.append(report)
        print(summarize_report(report))

    payload = {
        "site": "biek",
        "audited_urls": urls,
        "reports": reports,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved JSON audit report to: {output_path}")


if __name__ == "__main__":
    main()
