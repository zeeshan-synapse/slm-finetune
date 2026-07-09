from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urljoin

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.exceptions import CloseSpider

import scrape as scrape_utils
from site_profiles import SITE_PROFILES, SiteProfile, get_site_profile

DISCOVERY_ASSET_EXTENSIONS = (
    ".pdf",
    ".bmp",
    ".gif",
    ".jpg",
    ".jpeg",
    ".png",
    ".svg",
    ".webp",
    ".mp4",
    ".avi",
    ".mov",
    ".wmv",
    ".mp3",
    ".wav",
    ".zip",
    ".rar",
)


def default_manifest_path(profile: SiteProfile) -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "data"
        / "runtime"
        / f"{profile.output_domain}_crawl_manifest.json"
    )


def short_sample(values: list[str], limit: int = 5) -> list[str]:
    seen: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.append(value)
        if len(seen) >= limit:
            break
    return seen


def summarize_manifest(entries: list[dict], seed_urls: list[str], page_limit: int) -> dict:
    kind_counts = Counter(entry["kind"] for entry in entries)
    html_entries = [entry for entry in entries if entry["kind"] == "html_page"]
    pdf_entries = [entry for entry in entries if entry["kind"] == "pdf"]
    status_counts = Counter(str(entry.get("status_code", "unknown")) for entry in html_entries)
    crawled_urls = {entry["url"] for entry in html_entries}
    missing_seeds = [url for url in seed_urls if url not in crawled_urls]

    return {
        "page_limit": page_limit,
        "seed_url_count": len(seed_urls),
        "seed_urls": seed_urls,
        "discovered_total": len(entries),
        "html_page_count": len(html_entries),
        "pdf_count": len(pdf_entries),
        "kind_counts": dict(kind_counts),
        "status_counts": dict(status_counts),
        "missing_seed_urls": missing_seeds,
        "sample_html_urls": short_sample([entry["url"] for entry in html_entries]),
        "sample_pdf_urls": short_sample([entry["url"] for entry in pdf_entries]),
    }


class DiscoverySpider(scrapy.Spider):
    name = "discovery_spider"

    def __init__(self, *, profile_name: str, max_pages: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.profile: SiteProfile = get_site_profile(profile_name)
        self.page_limit = max_pages or self.profile.max_pages
        self.start_urls = [scrape_utils.normalize_url(url) for url in self.profile.seed_urls]
        self.allowed_domains = sorted(self.profile.allowed_domains)
        self.seen_html_urls: set[str] = set()
        self.scheduled_html_urls: set[str] = set(self.start_urls)
        self.seen_pdf_urls: set[str] = set()
        self.entries: list[dict] = []
        self.page_counter = 0

    def parse(self, response: scrapy.http.Response):
        normalized_url = scrape_utils.normalize_url(response.url)
        if normalized_url in self.seen_html_urls:
            return

        content_type = response.headers.get("Content-Type", b"").decode("latin-1")
        depth = response.meta.get("depth", 0)
        referrer = response.meta.get("referrer")
        if "text/html" not in content_type.lower():
            self.logger.info(
                "Skipping non-HTML response %s (content-type=%s)",
                normalized_url,
                content_type or "unknown",
            )
            return

        self.seen_html_urls.add(normalized_url)
        self.page_counter += 1

        title = response.css("title::text").get(default="").strip()

        self.entries.append(
            {
                "kind": "html_page",
                "url": normalized_url,
                "status_code": response.status,
                "content_type": content_type,
                "title": title,
                "depth": depth,
                "discovered_from": referrer,
            }
        )

        self.logger.info(
            "Discovered HTML %s (status=%s, depth=%s)",
            normalized_url,
            response.status,
            depth,
        )

        if self.page_counter >= self.page_limit:
            raise CloseSpider("page_limit_reached")

        for href, anchor_text in self.extract_links(response, normalized_url):
            lower_href = href.lower()
            if lower_href.endswith(".pdf"):
                if href in self.seen_pdf_urls:
                    continue
                self.seen_pdf_urls.add(href)
                self.entries.append(
                    {
                        "kind": "pdf",
                        "url": href,
                        "status_code": None,
                        "content_type": "application/pdf",
                        "title": anchor_text,
                        "depth": depth + 1,
                        "discovered_from": normalized_url,
                    }
                )
                continue

            if href in self.seen_html_urls or href in self.scheduled_html_urls:
                continue

            self.scheduled_html_urls.add(href)
            yield scrapy.Request(
                href,
                callback=self.parse,
                dont_filter=True,
                meta={"referrer": normalized_url, "depth": depth + 1},
            )

    def extract_links(self, response: scrapy.http.Response, current_url: str) -> list[tuple[str, str]]:
        discovered: list[tuple[str, str]] = []
        seen: set[str] = set()

        for link in response.css("a[href]"):
            raw_href = (link.attrib.get("href") or "").strip()
            if not raw_href or raw_href == "#":
                continue

            absolute = urljoin(current_url, raw_href).split("#")[0]
            normalized = scrape_utils.normalize_url(absolute)
            if normalized in seen:
                continue

            if not scrape_utils.is_same_domain(normalized, self.profile):
                continue
            if should_skip_discovery_link(normalized, self.profile):
                continue

            seen.add(normalized)
            discovered.append((normalized, " ".join(link.css("::text").getall()).strip()))

        return discovered

    def closed(self, reason: str) -> None:
        manifest_path = default_manifest_path(self.profile)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        summary = summarize_manifest(self.entries, self.start_urls, self.page_limit)
        payload = {
            "site": self.profile.name,
            "base_url": self.profile.base_url,
            "close_reason": reason,
            "summary": summary,
            "entries": self.entries,
        }
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        self.logger.info("-" * 50)
        self.logger.info("Done. Processed %s HTML pages", self.page_counter)
        self.logger.info("Discovered %s entries total", len(self.entries))
        self.logger.info("Manifest saved to: %s", manifest_path)


def build_settings(profile: SiteProfile, max_pages: int) -> dict:
    default_headers = {k: v for k, v in profile.headers.items() if k.lower() != "user-agent"}
    return {
        "LOG_LEVEL": "INFO",
        "ROBOTSTXT_OBEY": False,
        "USER_AGENT": profile.headers.get("User-Agent", ""),
        "DEFAULT_REQUEST_HEADERS": default_headers,
        "DOWNLOAD_DELAY": profile.delay,
        "DOWNLOAD_TIMEOUT": profile.timeout,
        "CONCURRENT_REQUESTS": 4,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": profile.delay,
        "AUTOTHROTTLE_MAX_DELAY": max(profile.delay * 4, 2.0),
        "TELNETCONSOLE_ENABLED": False,
        "CLOSESPIDER_PAGECOUNT": max_pages,
    }


def should_skip_discovery_link(url: str, profile: SiteProfile) -> bool:
    lowered = url.lower()
    if any(lowered.endswith(ext) for ext in DISCOVERY_ASSET_EXTENSIONS if ext != ".pdf"):
        return True
    return scrape_utils.should_skip(url, profile) and not lowered.endswith(".pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrapy-based discovery crawl for HTML pages and PDF links."
    )
    parser.add_argument(
        "--site",
        choices=sorted(SITE_PROFILES),
        default="biek",
        help="Which configured site profile to crawl.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional HTML page limit override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = get_site_profile(args.site)
    max_pages = args.max_pages or profile.max_pages
    manifest_path = default_manifest_path(profile)

    print(f"Starting discovery crawl for site profile: {profile.name}")
    print(f"Base URL: {profile.base_url}")
    print(f"Manifest path: {manifest_path}")
    print(f"Max HTML pages: {max_pages}\n")

    process = CrawlerProcess(settings=build_settings(profile, max_pages))
    process.crawl(DiscoverySpider, profile_name=profile.name, max_pages=max_pages)
    process.start()


if __name__ == "__main__":
    main()
