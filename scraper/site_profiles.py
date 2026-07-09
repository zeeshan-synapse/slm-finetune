from __future__ import annotations

from dataclasses import dataclass


DEFAULT_HEADERS = {
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
}


COMMON_SKIP_PATTERNS = [
    "javascript:",
    "mailto:",
    "tel:",
    "#",
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".webp",
    ".zip",
    ".rar",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
]


COMMON_NOISE_SELECTORS = [
    "script",
    "style",
    "noscript",
    "iframe",
    "svg",
    "form",
    "button",
]


@dataclass(frozen=True)
class SiteProfile:
    name: str
    base_url: str
    allowed_domains: set[str]
    seed_urls: list[str]
    output_domain: str
    headers: dict[str, str]
    skip_patterns: list[str]
    noise_selectors: list[str]
    main_selectors: list[str]
    max_pages: int = 100
    delay: float = 1.0
    timeout: int = 20
    min_content_length: int = 80


SYNAPSE_PROFILE = SiteProfile(
    name="synapse",
    base_url="https://synapsetechinc.com",
    allowed_domains={"synapsetechinc.com", "www.synapsetechinc.com"},
    seed_urls=[
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
    ],
    output_domain="synapse",
    headers={**DEFAULT_HEADERS, "Referer": "https://synapsetechinc.com/"},
    skip_patterns=COMMON_SKIP_PATTERNS
    + [
        "/cdn-cgi/",
        "/wp-",
        "/careers/",
        "/jobs/",
        "/login",
        "/logout",
        "/zh-cn",
        "/pt-br",
        "/es-",
        "/de-",
        "/fr-",
    ],
    noise_selectors=COMMON_NOISE_SELECTORS
    + [
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
    ],
    main_selectors=[
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
    ],
)


BIEK_PROFILE = SiteProfile(
    name="biek",
    base_url="https://www.biek.edu.pk/default.asp",
    allowed_domains={"www.biek.edu.pk", "biek.edu.pk"},
    seed_urls=[
        "https://www.biek.edu.pk/default.asp",
        "https://www.biek.edu.pk/history.asp",
        "https://www.biek.edu.pk/notifications.asp",
        "https://www.biek.edu.pk/Allforms.asp",
        "https://www.biek.edu.pk/affcoll.asp",
    ],
    output_domain="biek",
    headers={**DEFAULT_HEADERS, "Referer": "https://www.biek.edu.pk/default.asp"},
    skip_patterns=COMMON_SKIP_PATTERNS
    + [
        "online.biek.edu.pk",
        "institute.biek.edu.pk",
        "play.google.com",
        "tcs",
    ],
    noise_selectors=COMMON_NOISE_SELECTORS
    + [
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
        ".slider",
        ".carousel",
        ".owl-carousel",
    ],
    main_selectors=[
        "main",
        "article",
        "[role='main']",
        "#content",
        ".content",
        ".page-content",
        ".container",
        "body",
    ],
    max_pages=150,
)


SITE_PROFILES = {
    SYNAPSE_PROFILE.name: SYNAPSE_PROFILE,
    BIEK_PROFILE.name: BIEK_PROFILE,
}


def get_site_profile(name: str) -> SiteProfile:
    key = (name or "").strip().lower()
    if key not in SITE_PROFILES:
        available = ", ".join(sorted(SITE_PROFILES))
        raise ValueError(f"Unknown site profile '{name}'. Available: {available}")
    return SITE_PROFILES[key]
