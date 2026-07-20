#!/usr/bin/env python3
"""Extract the public navigation tree for a configured site.

For BIEK we keep a checked fallback snapshot because the site is old and the
navigation is the coverage contract for the KB. The live extraction path is
used when possible; the fallback lets local recovery/regeneration work without
network access.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scraper.site_profiles import get_site_profile  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "runtime" / "nav_tree.json"


BIEK_FALLBACK_ITEMS: list[dict[str, Any]] = [
    {"label": "Home", "url": "https://www.biek.edu.pk/default.asp"},
    {
        "label": "About Us",
        "url": "https://www.biek.edu.pk/default.asp#",
        "children": [
            {"label": "History of Board", "url": "https://www.biek.edu.pk/history.asp"},
            {"label": "Chairman", "url": "https://www.biek.edu.pk/chairman.asp"},
            {"label": "Board Members", "url": "https://www.biek.edu.pk/Bmembers.asp"},
            {"label": "Committees", "url": "https://www.biek.edu.pk/Committees.asp"},
            {"label": "Last Results", "url": "https://www.biek.edu.pk/LastResults.asp"},
            {"label": "Contact us", "url": "https://www.biek.edu.pk/ContactUs.asp"},
        ],
    },
    {
        "label": "Misc",
        "url": "https://www.biek.edu.pk/default.asp#",
        "children": [
            {"label": "Tenders", "url": "https://www.biek.edu.pk/Tenders.asp"},
            {"label": "Circulars & Notificaitons", "url": "https://www.biek.edu.pk/notifications.asp"},
            {"label": "Press Release", "url": "https://www.biek.edu.pk/press_release.asp"},
            {"label": "FTN No. (Active Tax Payer)", "url": "https://www.biek.edu.pk/FBR/FBR.jpg"},
        ],
    },
    {
        "label": "Examinations",
        "url": "https://www.biek.edu.pk/default.asp#",
        "children": [
            {
                "label": "Tutorial Video about online documents issuance",
                "url": "https://www.biek.edu.pk/online%20videos/tutorial_video_of_online_documents.mp4",
            },
            {"label": "Recent Result Declaration", "url": "https://www.biek.edu.pk/results.asp"},
            {"label": "Date Sheet", "url": "https://www.biek.edu.pk/Datesheet.asp"},
            {"label": "Authorized Banks", "url": "https://www.biek.edu.pk/authbanks.asp"},
            {"label": "Statistics", "url": "https://www.biek.edu.pk/Stats.asp"},
            {
                "label": "Sample E-Sheet of 30 Pages for Annual 2026 E-Marking",
                "url": "https://www.biek.edu.pk/E-Sheet-Sample/E-Sheet%2030%20pages.pdf",
            },
            {
                "label": "Sample E-Sheet of 22 Pages for Annual 2026 E-Marking",
                "url": "https://www.biek.edu.pk/E-Sheet-Sample/E-Sheet%2022%20Pages.pdf",
            },
            {
                "label": "Scheme of Studies for Higher Secondary Certificates",
                "url": "https://www.biek.edu.pk/Sc_Std/SCHEME-OF-STUDIES-FOR-HSC.pdf",
            },
            {"label": "Scheme & Model Paper 2023", "url": "https://www.biek.edu.pk/PapersScheme-2023.asp"},
            {
                "label": "Model Paper 2026",
                "url": "https://www.biek.edu.pk/ModelPaper/2026/Model%20Paper%202026.pdf",
            },
            {
                "label": "Science General Mathematics Part-II Model Paper",
                "url": "https://www.biek.edu.pk/ModelPaper/2026/SCI%20GENERAL%20Mathematics-II-NEW-BOOK%20MODEL%20PAPER%202026.pdf",
            },
            {"label": "MCQs", "url": "https://www.biek.edu.pk/mcq.asp"},
        ],
    },
    {
        "label": "Download Forms",
        "url": "https://www.biek.edu.pk/Allforms.asp",
        "children": [
            {"label": "Certificate Form", "url": "https://www.biek.edu.pk/Online_Forms/HSC-CERTIFICATE/Certificate-Form-Final.pdf"},
            {"label": "Scrutiny Form", "url": "https://www.biek.edu.pk/Online_Forms/Scrutiny%20Forms/Scruitny-Form.pdf"},
            {
                "label": "Provisional Certification Form",
                "url": "https://www.biek.edu.pk/Online_Forms/HSC-CERTIFICATE/Provisional-Certificate-Form.pdf",
            },
            {"label": "Migration Form", "url": "https://www.biek.edu.pk/Online_Forms/HSC-CERTIFICATE/MIGRATION-CERTIFICATE.pdf"},
            {
                "label": "Registration Form Commerce",
                "url": "https://www.biek.edu.pk/Online_Forms/Enrolment-Registration-forms/Registration-Form-Commerce.pdf",
            },
            {
                "label": "Registration Form Humanities",
                "url": "https://www.biek.edu.pk/Online_Forms/Enrolment-Registration-forms/Registration-Form-Humanities.pdf",
            },
            {"label": "Verification Certificate Form", "url": "https://biek.edu.pk/Online_Forms/Verification%20forms/Verification-Certificate-Form.pdf"},
            {"label": "Verification Provisional Certificate", "url": "https://biek.edu.pk/Online_Forms/Verification%20forms/Verification-Forms-All.pdf"},
            {"label": "Verification Marksheet Form", "url": "https://biek.edu.pk/Online_Forms/Verification%20forms/Verification-Forms-All.pdf"},
            {"label": "Verification Migration Form", "url": "https://biek.edu.pk/Online_Forms/Verification%20forms/Verification-Forms-All.pdf"},
            {
                "label": "Cancellation of Enrolment",
                "url": "https://biek.edu.pk/Online_Forms/Enrolment-Registration-forms/CANCELATION-OF-ENROLMENT.pdf",
            },
            {
                "label": "Cancellation of Registration",
                "url": "https://biek.edu.pk/Online_Forms/Enrolment-Registration-forms/CANCELATION-OF-REGISTRATION.pdf",
            },
            {"label": "Duplicate Marksheet", "url": "https://biek.edu.pk/Online_Forms/Duplicate%20Testimonials/BIEK-Duplicate-Forms.pdf"},
            {"label": "Duplicate Enrolment Card", "url": "https://biek.edu.pk/Online_Forms/Duplicate%20Testimonials/BIEK-Duplicate-Forms.pdf"},
            {
                "label": "Duplicate Computerized Admit Card",
                "url": "https://biek.edu.pk/Online_Forms/Duplicate%20Testimonials/BIEK-Duplicate-Forms.pdf",
            },
            {"label": "Duplicate Registration Card", "url": "https://biek.edu.pk/Online_Forms/Duplicate%20Testimonials/BIEK-Duplicate-Forms.pdf"},
            {"label": "Duplicate Manual Admit Card", "url": "https://biek.edu.pk/Online_Forms/Duplicate%20Testimonials/BIEK-Duplicate-Forms.pdf"},
            {"label": "Proforma of Special Chance", "url": "https://www.biek.edu.pk/Online_Forms/EXAMINATION-FORMS/BIEK-Duplicate-Forms.pdf"},
            {
                "label": "Improvement of Division",
                "url": "https://www.biek.edu.pk/Online_Forms/EXAMINATION-FORMS/Improvement-of-Division-Form-Final.pdf",
            },
            {
                "label": "Registration of All Groups",
                "url": "https://www.biek.edu.pk/Online_Forms/Enrolment-Registration-forms/Registration-Form-All-Groups.pdf",
            },
            {"label": "Examination Forms", "url": "https://www.biek.edu.pk/examinationforms.asp"},
            {"label": "Permission Forms", "url": "https://www.biek.edu.pk/Permissionforms.asp"},
        ],
    },
    {
        "label": "Recognition",
        "url": "https://www.biek.edu.pk/default.asp#",
        "children": [
            {"label": "Affiliated Colleges", "url": "https://www.biek.edu.pk/affcoll.asp"},
            {"label": "IOC Proforma for Affiliation", "url": "https://www.biek.edu.pk/IOC-Performa.asp"},
        ],
    },
]


def normalize_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def visible_text(tag: Tag) -> str:
    return normalize_text(tag.get_text(" ", strip=True))


def resolve_href(base_url: str, href: str | None) -> str:
    href = str(href or "").strip()
    if not href or href.lower().startswith("javascript:"):
        return ""
    return urljoin(base_url, href)


def is_noise_label(label: str) -> bool:
    return normalize_text(label).lower() in {"menu", "more", "login", "search", "register"}


def extract_from_html(html: str, base_url: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    nav = soup.find("nav") or soup.find("ul", class_=re.compile(r"(nav|menu|main)", re.I))
    if nav is None:
        return []

    roots: list[dict[str, Any]] = []
    for li in nav.find_all("li", recursive=False):
        node = extract_li(li, base_url)
        if node:
            roots.append(node)
    if roots:
        return roots

    for anchor in nav.find_all("a"):
        label = visible_text(anchor)
        if label and not is_noise_label(label):
            roots.append({"label": label, "url": resolve_href(base_url, anchor.get("href"))})
    return roots


def extract_li(li: Tag, base_url: str) -> dict[str, Any] | None:
    anchor = li.find("a", recursive=False) or li.find("a")
    label = visible_text(anchor or li)
    if not label or is_noise_label(label):
        return None
    node: dict[str, Any] = {"label": label, "url": resolve_href(base_url, anchor.get("href") if anchor else "")}
    children: list[dict[str, Any]] = []
    child_ul = li.find(["ul", "ol"], recursive=False)
    if child_ul:
        for child_li in child_ul.find_all("li", recursive=False):
            child = extract_li(child_li, base_url)
            if child:
                children.append(child)
    if children:
        node["children"] = children
    return node


def render_tree_lines(items: list[dict[str, Any]], depth: int = 0) -> list[str]:
    lines: list[str] = []
    prefix = "  " * depth
    for item in items:
        label = item.get("label") or ""
        url = item.get("url") or ""
        lines.append(f"{prefix}- {label} -> {url}")
        lines.extend(render_tree_lines(item.get("children") or [], depth + 1))
    return lines


def build_payload(site: str, url: str, title: str, items: list[dict[str, Any]], source: str) -> dict[str, Any]:
    return {
        "site": site,
        "url": url,
        "title": title,
        "source": source,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "top_level_count": len(items),
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a site's public navigation tree.")
    parser.add_argument("--site", default="biek")
    parser.add_argument("--url", default="")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--fallback", action="store_true", help="Use the built-in BIEK fallback nav snapshot.")
    args = parser.parse_args()

    profile = get_site_profile(args.site)
    url = args.url or getattr(profile, "base_url", "") or "https://www.biek.edu.pk/default.asp"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    title = "Welcome to Board of Intermediate Education Karachi"
    items = BIEK_FALLBACK_ITEMS
    source = "fallback"
    if not args.fallback:
        try:
            response = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            title = normalize_text(soup.title.get_text(" ", strip=True) if soup.title else title)
            extracted = extract_from_html(response.text, url)
            if extracted:
                items = extracted
                source = "live"
        except Exception:
            source = "fallback"

    payload = build_payload(args.site, url, title, items, source)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Site: {args.site}")
    print(f"URL: {url}")
    print(f"Title: {title}")
    print(f"Top-level items: {len(items)}")
    print(f"Saved JSON: {output.resolve()}")
    print()
    for line in render_tree_lines(items):
        print(line)


if __name__ == "__main__":
    main()
