from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from site_profiles import SITE_PROFILES, get_site_profile


def default_manifest_path(site: str) -> Path:
    profile = get_site_profile(site)
    return (
        Path(__file__).resolve().parents[1]
        / "data"
        / "runtime"
        / f"{profile.output_domain}_crawl_manifest.json"
    )


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifest(manifest: dict) -> list[str]:
    entries = manifest.get("entries", [])
    html_entries = [entry for entry in entries if entry.get("kind") == "html_page"]
    pdf_entries = [entry for entry in entries if entry.get("kind") == "pdf"]
    missing_seeds = manifest.get("summary", {}).get("missing_seed_urls", [])
    depths = Counter(str(entry.get("depth", 0)) for entry in html_entries)

    lines = [
        f"Site: {manifest.get('site', 'unknown')}",
        f"Close reason: {manifest.get('close_reason', 'unknown')}",
        f"HTML pages discovered: {len(html_entries)}",
        f"PDF links discovered: {len(pdf_entries)}",
        f"Missing seed URLs: {len(missing_seeds)}",
        f"Max HTML depth seen: {max((entry.get('depth', 0) for entry in html_entries), default=0)}",
        f"Depth distribution: {dict(sorted(depths.items(), key=lambda item: int(item[0])))}",
    ]

    status_counts = Counter(str(entry.get("status_code", "unknown")) for entry in html_entries)
    lines.append(f"HTML status counts: {dict(status_counts)}")

    if missing_seeds:
        lines.append("Missing seed URL samples:")
        lines.extend(f"  - {url}" for url in missing_seeds[:10])

    if not pdf_entries:
        lines.append("Coverage note: no PDF links were discovered.")

    if len(html_entries) < len(manifest.get("summary", {}).get("seed_urls", [])):
        lines.append("Coverage note: fewer HTML pages than seed URLs were captured.")

    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a crawl manifest and summarize coverage.")
    parser.add_argument(
        "--site",
        choices=sorted(SITE_PROFILES),
        default="biek",
        help="Which configured site profile manifest to verify.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional explicit crawl manifest path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest) if args.manifest else default_manifest_path(args.site)
    manifest = load_manifest(manifest_path)

    print(f"Manifest: {manifest_path}")
    for line in verify_manifest(manifest):
        print(line)


if __name__ == "__main__":
    main()
