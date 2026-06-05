#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import requests


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_PATH = PROJECT_DIR / "data" / "knowledge-base" / "faiss.index"
DEFAULT_META_PATH = PROJECT_DIR / "data" / "knowledge-base" / "index_meta.jsonl"
DEFAULT_MANIFEST_PATH = PROJECT_DIR / "data" / "knowledge-base" / "index_manifest.json"
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
TOKEN_RE = re.compile(r"[a-z0-9]+")
SPACING_RE = re.compile(r"\s+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "which",
    "who",
}
GENERIC_QUERY_TERMS = {
    "ai",
    "bot",
    "business",
    "company",
    "enterprise",
    "inc",
    "platform",
    "product",
    "products",
    "service",
    "services",
    "solution",
    "solutions",
    "synapse",
    "tech",
}
ENTITY_FOCUSED_PAGE_TYPE_BONUS = {
    "product": 0.09,
    "service": 0.05,
    "industry": 0.04,
    "index": -0.03,
    "about": -0.05,
    "contact": -0.02,
    "article": -0.01,
}
INTENT_PAGE_TYPE_BONUS = {
    "service": {"service": 0.14, "index": 0.08},
    "product": {"product": 0.14, "index": 0.08},
    "industry": {"industry": 0.14, "index": 0.06},
    "contact": {"contact": 0.16},
    "about": {"about": 0.14},
    "pricing": {"product": 0.06, "service": 0.06, "about": 0.05, "index": 0.04},
}

PRODUCT_ALIAS_PATTERNS = [
    ("coversaction ai", re.compile(r"\b(?:coversaction|conversaction)\s+ai\b|\bconversational\s+ai\b")),
    ("opira ai", re.compile(r"\bopira(?:\.io)?(?:\s+ai)?\b|\bopairo\b")),
    ("agentic bot", re.compile(r"\bagentic\s+bot\b")),
    ("irecruit one", re.compile(r"\birecruit(?:\s+one)?\b")),
    ("cyber security automation", re.compile(r"\bcyber\s+security\s+automation\b")),
]

CASUAL_REPLACEMENTS = {
    " u ": " you ",
    " ur ": " your ",
    " pls ": " please ",
    " plz ": " please ",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the local KB FAISS index with an embedded search question."
    )
    parser.add_argument(
        "query",
        nargs="+",
        help="Question or search query to run against the KB.",
    )
    parser.add_argument(
        "--index",
        default=str(DEFAULT_INDEX_PATH),
        help="Path to the FAISS index file.",
    )
    parser.add_argument(
        "--metadata",
        default=str(DEFAULT_META_PATH),
        help="Path to the metadata JSONL file.",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to the index manifest JSON file.",
    )
    parser.add_argument(
        "--model",
        help="Override the embedding model stored in the manifest.",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help="Base URL for the local Ollama server.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many results to show.",
    )
    parser.add_argument(
        "--page-type",
        action="append",
        default=[],
        help="Optional page_type filter. Repeat to allow multiple types.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON instead of a human-readable summary.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def check_ollama(ollama_url: str) -> None:
    response = requests.get(ollama_url, timeout=5)
    response.raise_for_status()


def embed_query_via_api_embed(ollama_url: str, model: str, query: str) -> list[float]:
    response = requests.post(
        f"{ollama_url}/api/embed",
        json={
            "model": model,
            "input": [query],
        },
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    embeddings = data.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != 1:
        raise ValueError("Unexpected /api/embed response shape from Ollama.")
    return embeddings[0]


def embed_query_via_legacy_api(ollama_url: str, model: str, query: str) -> list[float]:
    response = requests.post(
        f"{ollama_url}/api/embeddings",
        json={
            "model": model,
            "prompt": query,
        },
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    embedding = data.get("embedding")
    if not isinstance(embedding, list):
        raise ValueError("Unexpected /api/embeddings response shape from Ollama.")
    return embedding


def embed_query(ollama_url: str, model: str, query: str) -> np.ndarray:
    try:
        vector = embed_query_via_api_embed(ollama_url, model, query)
    except Exception as exc:
        print(f"/api/embed failed ({exc}); falling back to legacy /api/embeddings.")
        vector = embed_query_via_legacy_api(ollama_url, model, query)

    array = np.asarray([vector], dtype="float32")
    if array.ndim != 2 or array.shape[0] != 1:
        raise ValueError("Unexpected query embedding shape.")
    faiss.normalize_L2(array)
    return array


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def normalize_query_text(query: str) -> str:
    text = f" {query.lower()} "
    for src, dst in CASUAL_REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = SPACING_RE.sub(" ", text).strip()
    return text


def detect_product_aliases(normalized_query: str) -> list[str]:
    aliases: list[str] = []
    for label, pattern in PRODUCT_ALIAS_PATTERNS:
        if pattern.search(normalized_query):
            aliases.append(label)
    return aliases


def build_search_query(normalized_query: str, product_aliases: list[str]) -> str:
    if not product_aliases:
        return normalized_query
    alias_text = " ".join(product_aliases)
    return f"{normalized_query} {alias_text}"


def infer_intent(normalized_query: str, tokens: list[str], product_aliases: list[str]) -> str | None:
    token_set = set(tokens)

    if {"contact", "email", "phone", "reach"} & token_set:
        return "contact"
    if {"pricing", "price", "prices", "cost", "costs", "quote", "subscription"} & token_set:
        return "pricing"
    if {"industry", "industries"} & token_set:
        return "industry"
    if {"service", "services"} & token_set:
        return "service"
    if {"product", "products"} & token_set:
        return "product"

    product_list_phrases = (
        "what kind of products",
        "what kinds of products",
        "what products",
        "which products",
        "list all products",
        "list all the products",
        "list available products",
        "available products",
        "products do you have",
        "products you have",
        "products do you offer",
        "products you offer",
    )
    if any(phrase in normalized_query for phrase in product_list_phrases):
        return "product"

    service_list_phrases = (
        "what kind of services",
        "what kinds of services",
        "what services",
        "which services",
        "list all services",
        "available services",
        "services do you have",
        "services do you offer",
        "help choosing the right service",
    )
    if any(phrase in normalized_query for phrase in service_list_phrases):
        return "service"

    if product_aliases:
        return "product"

    if {"about", "mission", "company"} & token_set:
        return "about"

    return None


def query_profile(query: str) -> dict[str, Any]:
    normalized_query = normalize_query_text(query)
    product_aliases = detect_product_aliases(normalized_query)
    search_query = build_search_query(normalized_query, product_aliases)
    tokens = tokenize(normalized_query)
    keywords = [token for token in tokens if len(token) >= 3 and token not in STOPWORDS]
    entity_terms = [token for token in keywords if token not in GENERIC_QUERY_TERMS]
    for alias in product_aliases:
        for token in tokenize(alias):
            if token not in entity_terms and token not in GENERIC_QUERY_TERMS:
                entity_terms.append(token)
    intent = infer_intent(normalized_query, tokens, product_aliases)
    return {
        "query_text": normalized_query,
        "search_query": search_query,
        "tokens": tokens,
        "keywords": keywords,
        "entity_terms": entity_terms,
        "intent": intent,
        "product_aliases": product_aliases,
    }


def compute_rerank_score(
    row: dict[str, Any],
    *,
    raw_score: float,
    profile: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    title = (row.get("title") or "").lower()
    doc_id = (row.get("doc_id") or "").replace("-", " ").replace("_", " ").lower()
    text = (row.get("text") or "").lower()
    title_tokens = set(tokenize(title))
    doc_tokens = set(tokenize(doc_id))
    text_tokens = set(tokenize(text))

    boost = 0.0
    components: dict[str, float] = {}

    keywords = profile["keywords"]
    entity_terms = profile["entity_terms"]

    if keywords:
        keyword_title_matches = sum(token in title_tokens for token in keywords)
        keyword_doc_matches = sum(token in doc_tokens for token in keywords)
        keyword_text_matches = sum(token in text_tokens for token in keywords)

        keyword_overlap = keyword_text_matches / len(keywords)
        boost += keyword_overlap * 0.06
        boost += keyword_title_matches * 0.05
        boost += keyword_doc_matches * 0.06

        if keyword_overlap:
            components["keyword_overlap"] = keyword_overlap * 0.06
        if keyword_title_matches:
            components["keyword_title_match"] = keyword_title_matches * 0.05
        if keyword_doc_matches:
            components["keyword_doc_match"] = keyword_doc_matches * 0.06

    if entity_terms:
        entity_title_matches = sum(token in title_tokens for token in entity_terms)
        entity_doc_matches = sum(token in doc_tokens for token in entity_terms)
        entity_text_matches = sum(token in text_tokens for token in entity_terms)

        if entity_title_matches:
            value = entity_title_matches * 0.16
            boost += value
            components["entity_title_match"] = value
        if entity_doc_matches:
            value = entity_doc_matches * 0.18
            boost += value
            components["entity_doc_match"] = value
        if entity_text_matches:
            value = min(entity_text_matches * 0.03, 0.09)
            boost += value
            components["entity_text_match"] = value

        all_entities_present = all(token in text_tokens or token in title_tokens for token in entity_terms)
        if all_entities_present:
            boost += 0.08
            components["all_entity_terms_present"] = 0.08

        page_type_bonus = ENTITY_FOCUSED_PAGE_TYPE_BONUS.get(row.get("page_type", ""), 0.0)
        if page_type_bonus:
            boost += page_type_bonus
            components["page_type_bias"] = page_type_bonus

    product_aliases = profile.get("product_aliases") or []
    for alias in product_aliases:
        alias_tokens = set(tokenize(alias))
        if not alias_tokens:
            continue
        title_doc_matches = len(alias_tokens & (title_tokens | doc_tokens))
        text_matches = len(alias_tokens & text_tokens)
        if title_doc_matches:
            value = min(title_doc_matches * 0.28, 0.56)
            boost += value
            components["product_alias_title_doc_match"] = components.get("product_alias_title_doc_match", 0.0) + value
        elif text_matches:
            value = min(text_matches * 0.06, 0.18)
            boost += value
            components["product_alias_text_match"] = components.get("product_alias_text_match", 0.0) + value

    intent = profile.get("intent")
    if intent:
        intent_bonus = INTENT_PAGE_TYPE_BONUS.get(intent, {}).get(row.get("page_type", ""), 0.0)
        if intent_bonus:
            boost += intent_bonus
            components["intent_page_type_bias"] = intent_bonus

    rerank_score = raw_score + boost
    return rerank_score, components


def build_hits(
    *,
    scores: np.ndarray,
    indices: np.ndarray,
    rows: list[dict[str, Any]],
    allowed_page_types: set[str],
    top_k: int,
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    hits = []
    for raw_score, idx in zip(scores[0], indices[0], strict=False):
        if idx < 0:
            continue
        row = rows[idx]
        if allowed_page_types and row.get("page_type") not in allowed_page_types:
            continue
        rerank_score, score_components = compute_rerank_score(
            row,
            raw_score=float(raw_score),
            profile=profile,
        )
        hits.append(
            {
                "score": rerank_score,
                "raw_score": float(raw_score),
                "chunk_id": row.get("chunk_id"),
                "doc_id": row.get("doc_id"),
                "page_type": row.get("page_type"),
                "title": row.get("title"),
                "url": row.get("url"),
                "text": row.get("text"),
                "score_components": score_components,
            }
        )

    hits.sort(key=lambda hit: (hit["score"], hit["raw_score"]), reverse=True)
    return hits[:top_k]


def search_index(
    index: faiss.Index,
    query_vector: np.ndarray,
    rows: list[dict[str, Any]],
    allowed_page_types: set[str],
    top_k: int,
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    search_k = min(index.ntotal, max(top_k * 20, 50))
    while True:
        scores, indices = index.search(query_vector, search_k)
        hits = build_hits(
            scores=scores,
            indices=indices,
            rows=rows,
            allowed_page_types=allowed_page_types,
            top_k=top_k,
            profile=profile,
        )
        if len(hits) >= top_k or search_k >= index.ntotal:
            return hits
        search_k = min(index.ntotal, search_k * 2)


def preview(text: str, limit: int = 260) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be a positive integer.")

    query_text = " ".join(args.query).strip()
    profile = query_profile(query_text)
    manifest = load_manifest(Path(args.manifest))
    model = args.model or manifest.get("embedding_model")
    if not model:
        raise ValueError(
            "No embedding model provided. Use --model or ensure the manifest exists."
        )

    index = faiss.read_index(str(Path(args.index)))
    rows = load_rows(Path(args.metadata))
    if index.ntotal != len(rows):
        raise ValueError("FAISS index size does not match metadata row count.")

    allowed_page_types = set(args.page_type)

    check_ollama(args.ollama_url)
    query_vector = embed_query(args.ollama_url, model, query_text)
    hits = search_index(
        index,
        query_vector,
        rows,
        allowed_page_types,
        args.top_k,
        profile,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "query": query_text,
                    "embedding_model": model,
                    "results": hits,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print("=" * 60)
    print("KB Retrieval Results")
    print("=" * 60)
    print(f"Query: {query_text}")
    print(f"Embedding model: {model}")
    if allowed_page_types:
        print("Page type filter:", ", ".join(sorted(allowed_page_types)))
    print()

    if not hits:
        print("No matches found.")
        return

    for rank, hit in enumerate(hits, start=1):
        print(
            f"[{rank}] score={hit['score']:.4f} raw={hit['raw_score']:.4f} "
            f"page_type={hit['page_type']}"
        )
        print(f"    title: {hit['title']}")
        print(f"    chunk: {hit['chunk_id']}")
        print(f"    url:   {hit['url']}")
        print(f"    text:  {preview(hit['text'])}")
        print()


if __name__ == "__main__":
    main()
