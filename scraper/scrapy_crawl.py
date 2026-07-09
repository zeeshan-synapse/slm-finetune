from __future__ import annotations

import argparse
import json
from pathlib import Path

from bs4 import BeautifulSoup
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.exceptions import CloseSpider

import scrape as scrape_utils
from kb_paths import cleaned_data_dir
from site_profiles import SITE_PROFILES, SiteProfile, get_site_profile


class ProfileSpider(scrapy.Spider):
    name = "profile_spider"

    def __init__(self, *, profile_name: str, max_pages: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.profile: SiteProfile = get_site_profile(profile_name)
        self.page_limit = max_pages or self.profile.max_pages
        self.output_dir = cleaned_data_dir(self.profile.output_domain)
        self.start_urls = [scrape_utils.normalize_url(url) for url in self.profile.seed_urls]
        self.allowed_domains = sorted(self.profile.allowed_domains)
        self.scraped_pages: list[dict] = []
        self.seen_urls: set[str] = set()
        self.scheduled_urls: set[str] = set(self.start_urls)
        self.page_counter = 0

    def parse(self, response: scrapy.http.Response):
        normalized_url = scrape_utils.normalize_url(response.url)
        if normalized_url in self.seen_urls:
            return

        self.seen_urls.add(normalized_url)
        self.page_counter += 1

        content_type = response.headers.get("Content-Type", b"").decode("latin-1")
        if "text/html" not in content_type:
            self.logger.info("Skipping %s (non-HTML response)", normalized_url)
            return

        html = response.text
        soup = BeautifulSoup(html, "lxml")
        title = soup.title.get_text(strip=True) if soup.title else normalized_url
        text, extractor_used, validation_reason = scrape_utils.extract_text(
            html,
            soup,
            self.profile,
            normalized_url,
            title,
        )

        if len(text) < self.profile.min_content_length:
            self.logger.info(
                "Skipping %s (too little/invalid content: %s chars, reason=%s)",
                normalized_url,
                len(text),
                validation_reason,
            )
        else:
            saved_path = scrape_utils.write_text_file(normalized_url, title, text, self.output_dir)
            self.scraped_pages.append(
                {
                    "url": normalized_url,
                    "title": title,
                    "content": text,
                    "text_file": saved_path,
                    "extractor": extractor_used,
                    "validation": validation_reason,
                }
            )
            self.logger.info(
                "Saved %s (%s chars, extractor=%s, validation=%s)",
                saved_path,
                len(text),
                extractor_used,
                validation_reason,
            )

        if self.page_counter >= self.page_limit:
            raise CloseSpider("page_limit_reached")

        for link in scrape_utils.extract_links(soup, normalized_url, self.profile):
            if link in self.seen_urls or link in self.scheduled_urls:
                continue
            self.scheduled_urls.add(link)
            yield scrapy.Request(link, callback=self.parse, dont_filter=True)

    def closed(self, reason: str) -> None:
        json_output_path = self.output_dir / "scraped.json"
        with json_output_path.open("w", encoding="utf-8") as f:
            json.dump(self.scraped_pages, f, indent=2, ensure_ascii=False)

        self.logger.info("-" * 50)
        self.logger.info("Done. Processed %s responses", self.page_counter)
        self.logger.info("Saved %s pages", len(self.scraped_pages))
        self.logger.info("Text files saved in: %s", self.output_dir)
        self.logger.info("JSON index saved to: %s", json_output_path)
        self.logger.info("Close reason: %s", reason)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrapy-based crawl for a configured site profile.")
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
        help="Optional page limit override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = get_site_profile(args.site)
    max_pages = args.max_pages or profile.max_pages
    output_dir = cleaned_data_dir(profile.output_domain)
    scrape_utils.reset_output_dir(output_dir)

    print(f"Starting Scrapy crawl for site profile: {profile.name}")
    print(f"Base URL: {profile.base_url}")
    print(f"Output dir: {output_dir}")
    print(f"Max pages: {max_pages}\n")

    process = CrawlerProcess(settings=build_settings(profile, max_pages))
    process.crawl(ProfileSpider, profile_name=profile.name, max_pages=max_pages)
    process.start()


if __name__ == "__main__":
    main()
