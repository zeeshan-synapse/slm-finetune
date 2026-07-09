from __future__ import annotations

import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DOMAIN = os.environ.get("KB_DOMAIN", "synapse").strip() or "synapse"


def raw_root() -> Path:
    return PROJECT_DIR / "data" / "raw"


def knowledge_base_root() -> Path:
    return PROJECT_DIR / "data" / "knowledge-base"


def cleaned_data_dir(domain: str = DEFAULT_DOMAIN) -> Path:
    return raw_root() / f"{domain}-cleaned-data"


def knowledge_base_dir(domain: str = DEFAULT_DOMAIN) -> Path:
    return knowledge_base_root() / domain
