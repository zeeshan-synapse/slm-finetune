from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
try:
    import trafilatura
except ImportError:  # pragma: no cover - handled at runtime when dependency is missing
    trafilatura = None

_ROOT_DIR = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _ROOT_DIR / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from kb_paths import cleaned_data_dir
from site_profiles import SITE_PROFILES, SiteProfile, get_site_profile


BIEK_DROP_LINE_PATTERNS = (
    "© 2026 board of intermediate education karachi",
    "all rights reserved",
    "every year conducting about 375,000 candidates examination",
    "every year conducting about 375,000 candidates examination!, assessments & results",
    "mega events, students results declaration, medal award ceremony",
)
BIEK_HOMEPAGE_BLEED_PATTERNS = (
    "the board has announced to get any type of form online",
    "board of interemdiate education karachi has introduced a unique and advanced vertificaiton system",
    "board of intermediate education karachi is now developed an one pager form",
    "board of intermediate education karachi has introduced a online banking system",
    "board of intermediate education it department has devised the international pattern of marksheet",
)
BIEK_EXPECTED_SIGNALS = {
    "homepage": ("student", "board", "form", "model paper", "fee", "karachi"),
    "contact": ("contact", "phone", "email", "address", "board office", "fax"),
    "history": ("history", "ordinance", "university", "education", "board", "established"),
    "affiliation": ("affiliation", "affiliated", "college", "recognition", "institution"),
    "forms": ("form", "forms", "download", "voucher", "verification", "registration", "fee"),
    "datesheet": ("date sheet", "datesheet", "exam", "examination", "schedule", "hsc"),
    "committee": ("committee", "member", "chairman", "secretary", "board"),
}
BIEK_MISMATCH_SIGNALS = {
    "contact": ("ordinance", "vice-chancellor", "sharif commission"),
    "affiliation": ("ordinance", "vice-chancellor", "sharif commission"),
    "committee": ("ordinance", "vice-chancellor", "sharif commission"),
}
BIEK_MOJIBAKE_REPLACEMENTS = {
    "âWest": "West",
    "â€™": "'",
    "â€œ": '"',
    "â€": '"',
    "â€": '"',
    "â": "",
}


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc.lower()
    if netloc in {"biek.edu.pk", "www.biek.edu.pk"}:
        scheme = "https"
        netloc = "www.biek.edu.pk"
    path = parsed.path.rstrip("/") or "/"
    return f"{scheme}://{netloc}{path}"


def should_skip(url: str, profile: SiteProfile) -> bool:
    lowered = url.lower()
    return any(pattern.lower() in lowered for pattern in profile.skip_patterns)


def is_same_domain(url: str, profile: SiteProfile) -> bool:
    parsed = urlparse(url)
    return parsed.netloc in profile.allowed_domains or parsed.netloc == ""


def make_session(profile: SiteProfile) -> requests.Session:
    session = requests.Session()
    session.headers.update(profile.headers)
    return session


def slugify_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/")

    if not path:
        return "home"

    pieces = [piece for piece in path.split("/") if piece]
    slug = "-".join(pieces)
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", slug).strip("-").lower()
    return slug or "page"


def unique_slug_for_url(url: str) -> str:
    slug = slugify_url(url)
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}"


def clean_lines(lines: list[str]) -> list[str]:
    cleaned = []
    seen = set()

    for raw_line in lines:
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue

        lowered = line.lower()
        if lowered in seen:
            continue

        seen.add(lowered)
        cleaned.append(line)

    return cleaned


def normalized_text(text: str) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    cleaned = clean_lines(lines)
    return "\n".join(cleaned)


def normalize_mojibake(text: str) -> str:
    fixed = text
    for bad, good in BIEK_MOJIBAKE_REPLACEMENTS.items():
        fixed = fixed.replace(bad, good)
    return fixed


def biek_page_kind(url: str) -> str:
    path = urlparse(url).path.lower()
    if path.endswith("/default.asp") or path == "/" or path.endswith("/home.asp"):
        return "homepage"
    if "contact" in path:
        return "contact"
    if "history" in path:
        return "history"
    if "affcoll" in path or "affiliation" in path or "recognition" in path:
        return "affiliation"
    if "form" in path or "voucher" in path:
        return "forms"
    if "datesheet" in path:
        return "datesheet"
    if "committee" in path or "member" in path or "chairman" in path:
        return "committee"
    return "generic"


def clean_biek_text(url: str, title: str, text: str) -> str:
    page_kind = biek_page_kind(url)
    working = normalize_mojibake(text)
    kept_lines: list[str] = []

    for raw_line in working.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue

        lowered = line.lower()
        if any(pattern in lowered for pattern in BIEK_DROP_LINE_PATTERNS):
            continue
        if page_kind != "generic" and any(pattern in lowered for pattern in BIEK_HOMEPAGE_BLEED_PATTERNS):
            continue
        if page_kind == "homepage" and (
            "ordinance number iii of 1962" in lowered
            or "vice-chancellor" in lowered
            or "on 8 january 1962" in lowered
        ):
            continue
        if page_kind in {"affiliation", "committee"} and (
            "ordinance number iii of 1962" in lowered
            or "vice-chancellor" in lowered
            or "sharif commission" in lowered
            or "on 8 january 1962" in lowered
        ):
            continue
        if page_kind == "contact" and ("ordinance number iii of 1962" in lowered or "vice-chancellor" in lowered):
            continue

        kept_lines.append(line)

    cleaned = normalized_text("\n".join(kept_lines))

    if page_kind == "homepage":
        homepage_lines: list[str] = []
        for line in cleaned.splitlines():
            lowered = line.lower()
            if "board of intermediate education karachi on 8 january 1962" in lowered:
                continue
            homepage_lines.append(line)
        cleaned = normalized_text("\n".join(homepage_lines))

    return cleaned


def validate_biek_text(url: str, text: str, profile: SiteProfile) -> tuple[bool, str]:
    if len(text) < profile.min_content_length:
        return False, "below_min_content_length"

    page_kind = biek_page_kind(url)
    lowered = text.lower()

    if page_kind == "generic":
        return True, "generic_ok"

    expected_hits = sum(1 for token in BIEK_EXPECTED_SIGNALS.get(page_kind, ()) if token in lowered)
    mismatch_hits = sum(1 for token in BIEK_MISMATCH_SIGNALS.get(page_kind, ()) if token in lowered)

    if page_kind == "homepage":
        if expected_hits < 2:
            return False, "missing_homepage_signals"
        return True, "homepage_ok"

    if page_kind == "contact":
        if expected_hits == 0:
            return False, "missing_contact_signals"
        if mismatch_hits >= 2:
            return False, "history_bleed_on_contact_page"
        return True, "contact_ok"

    if page_kind in {"affiliation", "committee"}:
        if expected_hits == 0:
            return False, f"missing_{page_kind}_signals"
        if mismatch_hits >= 1:
            return False, f"history_bleed_on_{page_kind}_page"
        return True, f"{page_kind}_ok"

    if page_kind in {"history", "forms", "datesheet"}:
        if expected_hits == 0:
            return False, f"missing_{page_kind}_signals"
        return True, f"{page_kind}_ok"

    return True, "ok"


def finalize_extracted_text(
    *,
    profile: SiteProfile,
    url: str,
    title: str,
    text: str,
) -> tuple[str, bool, str]:
    cleaned = normalized_text(text)
    if profile.name != "biek":
        return cleaned, len(cleaned) >= profile.min_content_length, "generic"

    cleaned = clean_biek_text(url, title, cleaned)
    is_valid, reason = validate_biek_text(url, cleaned, profile)
    return cleaned, is_valid, reason


def extract_text_fallback(soup: BeautifulSoup, profile: SiteProfile) -> str:
    working_soup = BeautifulSoup(str(soup), "lxml")

    for selector in profile.noise_selectors:
        for tag in working_soup.select(selector):
            tag.decompose()

    main = None
    for selector in profile.main_selectors:
        main = working_soup.select_one(selector)
        if main:
            break

    if main is None:
        main = working_soup.body

    if main is None:
        return ""

    lines = main.get_text(separator="\n").splitlines()
    return "\n".join(clean_lines(lines))


def extract_text_with_trafilatura(html: str, url: str) -> str:
    if trafilatura is None:
        return ""

    extracted = trafilatura.extract(
        html,
        url=url,
        favor_precision=True,
        include_links=False,
        include_tables=False,
        include_images=False,
    )
    return normalized_text(extracted)


def extract_text(
    html: str,
    soup: BeautifulSoup,
    profile: SiteProfile,
    url: str,
    title: str,
) -> tuple[str, str, str]:
    primary_text = extract_text_with_trafilatura(html, url)
    primary_text, primary_valid, primary_reason = finalize_extracted_text(
        profile=profile,
        url=url,
        title=title,
        text=primary_text,
    )
    if primary_valid:
        return primary_text, "trafilatura", primary_reason

    fallback_text, fallback_valid, fallback_reason = finalize_extracted_text(
        profile=profile,
        url=url,
        title=title,
        text=extract_text_fallback(soup, profile),
    )
    if fallback_valid:
        return fallback_text, "beautifulsoup_fallback", fallback_reason

    final_reason = f"trafilatura:{primary_reason};fallback:{fallback_reason}"
    return "", "empty", final_reason


def extract_links(soup: BeautifulSoup, current_url: str, profile: SiteProfile) -> list[str]:
    links = []

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        if not href or href == "#":
            continue

        full_url = normalize_url(urljoin(current_url, href).split("#")[0])

        if is_same_domain(full_url, profile) and not should_skip(full_url, profile):
            links.append(full_url)

    return sorted(set(links))


def write_text_file(url: str, title: str, text: str, output_dir: Path) -> str:
    filename = f"{unique_slug_for_url(url)}.txt"
    path = output_dir / filename

    with path.open("w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write(f"URL: {url}\n\n")
        f.write(text)
        f.write("\n")

    return str(path)


def reset_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*.txt"):
        path.unlink()
    json_path = output_dir / "scraped.json"
    if json_path.exists():
        json_path.unlink()


def fetch_page(session: requests.Session, url: str, profile: SiteProfile) -> requests.Response:
    return session.get(url, timeout=profile.timeout, allow_redirects=True)


def crawl(profile: SiteProfile, max_pages: int | None = None) -> list[dict]:
    output_dir = cleaned_data_dir(profile.output_domain)
    json_output_path = output_dir / "scraped.json"
    page_limit = max_pages or profile.max_pages

    os.makedirs(output_dir, exist_ok=True)

    session = make_session(profile)
    visited = set()
    queue = [normalize_url(url) for url in profile.seed_urls]
    scraped_pages = []

    print(f"Starting crawl for site profile: {profile.name}")
    print(f"Base URL: {profile.base_url}")
    print(f"Output dir: {output_dir}")
    print(f"Max pages: {page_limit}\n")

    while queue and len(visited) < page_limit:
        url = queue.pop(0)

        if url in visited:
            continue

        visited.add(url)
        print(f"[{len(visited)}/{page_limit}] Scraping: {url}")

        try:
            response = fetch_page(session, url, profile)
        except requests.exceptions.RequestException as e:
            print(f"  Request failed: {e}")
            continue

        if response.status_code != 200:
            print(f"  Skipping (status {response.status_code})")
            continue

        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            print("  Skipping (non-HTML response)")
            continue

        soup = BeautifulSoup(response.text, "lxml")
        title = soup.title.get_text(strip=True) if soup.title else url
        text, extractor_used, validation_reason = extract_text(
            response.text,
            soup,
            profile,
            url,
            title,
        )

        if len(text) < profile.min_content_length:
            print(
                f"  Skipping (too little/invalid content: {len(text)} chars, "
                f"reason={validation_reason})"
            )
            continue

        saved_path = write_text_file(url, title, text, output_dir)
        page_data = {
            "url": url,
            "title": title,
            "content": text,
            "text_file": saved_path,
            "extractor": extractor_used,
            "validation": validation_reason,
        }
        scraped_pages.append(page_data)

        new_links = extract_links(soup, url, profile)
        for link in new_links:
            if link not in visited and link not in queue:
                queue.append(link)

        print(
            f"  Saved {saved_path} ({len(text)} chars, extractor={extractor_used}, "
            f"validation={validation_reason}) | "
            f"Queue: {len(queue)}"
        )
        time.sleep(profile.delay)

    with json_output_path.open("w", encoding="utf-8") as f:
        json.dump(scraped_pages, f, indent=2, ensure_ascii=False)

    print(f"\n{'-' * 50}")
    print(f"Done. Scraped {len(scraped_pages)} pages")
    print(f"Text files saved in: {output_dir}")
    print(f"JSON index saved to: {json_output_path}")
    return scraped_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape a configured site profile.")
    parser.add_argument(
        "--site",
        choices=sorted(SITE_PROFILES),
        default="synapse",
        help="Which configured site profile to scrape.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional page limit override.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    crawl(get_site_profile(args.site), max_pages=args.max_pages)
