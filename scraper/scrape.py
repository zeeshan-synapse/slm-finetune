import json
import os
import re
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ── Config ──────────────────────────────────────────────────────────────────
BASE_URL = "https://synapsetechinc.com"
ALLOWED_DOMAINS = {"synapsetechinc.com", "www.synapsetechinc.com"}

SEED_URLS = [
    "https://synapsetechinc.com/",
    "https://synapsetechinc.com/about-us/",
    "https://synapsetechinc.com/product/",
    "https://synapsetechinc.com/product/irecruit-one/",
    "https://synapsetechinc.com/product/agentic-bot/",
    "https://synapsetechinc.com/product/coversaction-ai/",
    "https://synapsetechinc.com/product/opira-ai/",
    "https://synapsetechinc.com/product/cyber-security-automation/",
    "https://synapsetechinc.com/services/",
    "https://synapsetechinc.com/services/automation/",
    "https://synapsetechinc.com/services/cloud-ai/",
    "https://synapsetechinc.com/services/custom-development/",
    "https://synapsetechinc.com/services/voice-agent/",
    "https://synapsetechinc.com/industries/",
    "https://synapsetechinc.com/industries/banking-and-financial/",
    "https://synapsetechinc.com/industries/cyber-security/",
    "https://synapsetechinc.com/industries/bpo-contact-centers/",
    "https://synapsetechinc.com/industries/retail-e-commerce/",
    "https://synapsetechinc.com/blogs/",
    "https://synapsetechinc.com/contact-us/",
]

RAW_DIR = "data/raw"
OUTPUT_DIR = os.path.join(RAW_DIR, "cleaned-data")
JSON_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "scraped.json")
MAX_PAGES = 100
DELAY = 1.0
TIMEOUT = 20
MIN_CONTENT_LENGTH = 80

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Referer": BASE_URL + "/",
}

SKIP_PATTERNS = [
    "/cdn-cgi/",
    "/wp-",
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".svg",
    ".webp",
    "/careers/",
    "/jobs/",
    "/login",
    "/logout",
    "javascript:",
    "mailto:",
    "#",
    "/zh-cn",
    "/pt-br",
    "/es-",
    "/de-",
    "/fr-",
]

NOISE_SELECTORS = [
    "script",
    "style",
    "noscript",
    "iframe",
    "svg",
    "form",
    "button",
    "nav",
    "footer",
    "header",
    ".menu",
    ".navigation",
    ".nav",
    ".footer",
    ".header",
    ".breadcrumbs",
    ".breadcrumb",
    ".cookie",
    ".popup",
    ".modal",
    ".newsletter",
]

MAIN_SELECTORS = [
    "main",
    "article",
    "[role='main']",
    ".elementor-location-single",
    ".elementor-location-archive",
    ".elementor-widget-theme-post-content",
    ".site-main",
    "#content",
    ".content",
    ".page-content",
    ".entry-content",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def should_skip(url: str) -> bool:
    return any(pattern in url for pattern in SKIP_PATTERNS)


def is_same_domain(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc in ALLOWED_DOMAINS or parsed.netloc == ""


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
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


def clean_lines(lines: list[str]) -> list[str]:
    cleaned = []
    seen = set()

    for raw_line in lines:
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue

        # Drop exact duplicates that often come from repeated UI blocks.
        lowered = line.lower()
        if lowered in seen:
            continue

        seen.add(lowered)
        cleaned.append(line)

    return cleaned


def extract_text(soup: BeautifulSoup) -> str:
    for selector in NOISE_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()

    main = None
    for selector in MAIN_SELECTORS:
        main = soup.select_one(selector)
        if main:
            break

    if main is None:
        main = soup.body

    if main is None:
        return ""

    lines = main.get_text(separator="\n").splitlines()
    cleaned = clean_lines(lines)
    return "\n".join(cleaned)


def extract_links(soup: BeautifulSoup, current_url: str) -> list[str]:
    links = []

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        full_url = normalize_url(urljoin(current_url, href).split("#")[0])

        if is_same_domain(full_url) and not should_skip(full_url):
            links.append(full_url)

    return sorted(set(links))


def write_text_file(url: str, title: str, text: str) -> str:
    filename = f"{slugify_url(url)}.txt"
    path = os.path.join(OUTPUT_DIR, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write(f"URL: {url}\n\n")
        f.write(text)
        f.write("\n")

    return path


def fetch_page(session: requests.Session, url: str) -> requests.Response:
    return session.get(url, timeout=TIMEOUT, allow_redirects=True)


# ── Main Crawler ─────────────────────────────────────────────────────────────

def crawl(start_url: str, seed_urls: list[str] | None = None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    session = make_session()
    visited = set()
    queue = [normalize_url(url) for url in (seed_urls or [start_url])]
    scraped_pages = []

    print(f"Starting crawl from: {start_url}")
    print(f"Max pages: {MAX_PAGES}\n")

    while queue and len(visited) < MAX_PAGES:
        url = queue.pop(0)

        if url in visited:
            continue

        visited.add(url)
        print(f"[{len(visited)}/{MAX_PAGES}] Scraping: {url}")

        try:
            response = fetch_page(session, url)
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Request failed: {e}")
            continue

        if response.status_code != 200:
            print(f"  ⚠️  Skipping (status {response.status_code})")
            continue

        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            print("  ⚠️  Skipping (non-HTML response)")
            continue

        soup = BeautifulSoup(response.text, "lxml")
        title = soup.title.get_text(strip=True) if soup.title else url
        text = extract_text(soup)

        if len(text) < MIN_CONTENT_LENGTH:
            print(f"  ⚠️  Skipping (too little content: {len(text)} chars)")
            continue

        saved_path = write_text_file(url, title, text)
        page_data = {
            "url": url,
            "title": title,
            "content": text,
            "text_file": saved_path,
        }
        scraped_pages.append(page_data)

        new_links = extract_links(soup, url)
        for link in new_links:
            if link not in visited and link not in queue:
                queue.append(link)

        print(f"  ✅ Saved {saved_path} ({len(text)} chars) | Queue: {len(queue)}")
        time.sleep(DELAY)

    with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(scraped_pages, f, indent=2, ensure_ascii=False)

    print(f"\n{'─' * 50}")
    print(f"✅ Done! Scraped {len(scraped_pages)} pages")
    print(f"📁 Text files saved in: {OUTPUT_DIR}")
    print(f"📁 JSON index saved to: {JSON_OUTPUT_PATH}")
    return scraped_pages


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    crawl(BASE_URL, seed_urls=SEED_URLS)
