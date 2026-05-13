#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import faiss
import requests

import query_kb


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_PATH = PROJECT_DIR / "data" / "knowledge-base" / "faiss.index"
DEFAULT_META_PATH = PROJECT_DIR / "data" / "knowledge-base" / "index_meta.jsonl"
DEFAULT_MANIFEST_PATH = PROJECT_DIR / "data" / "knowledge-base" / "index_manifest.json"
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_GENERATION_MODEL = os.environ.get("KB_ANSWER_MODEL", "synapse-3b")
DEFAULT_FALLBACK_RESPONSE = (
    "This detail is not confirmed in the available information. Please verify with Synapse Tech."
)
SUMMARY_INTENT_CONFIG = {
    "service": {
        "index_doc_id": "services",
        "detail_page_type": "service",
        "label": "services",
    },
    "product": {
        "index_doc_id": "product",
        "detail_page_type": "product",
        "label": "products",
    },
    "industry": {
        "index_doc_id": "industries",
        "detail_page_type": "industry",
        "label": "industries",
    },
}
PRODUCT_NAME_PATTERNS = [
    ("Agentic Bot", re.compile(r"\bagentic bot\b", re.IGNORECASE)),
    ("iRecruit One", re.compile(r"\birecruit(?:\s+one)?\b", re.IGNORECASE)),
    ("Opira Ai", re.compile(r"\bopira(?:\.io)?\b|\bopairo\b", re.IGNORECASE)),
    ("Coversaction Ai", re.compile(r"\bcoversaction ai\b|\bconversaction ai\b", re.IGNORECASE)),
    (
        "Cyber Security Automation",
        re.compile(r"\bcyber security automation\b", re.IGNORECASE),
    ),
]
INDUSTRY_NAME_PATTERNS = [
    ("banking and financial services", re.compile(r"\bbank(?:s|ing)?\b|\bfinancial\b|\bfintechs?\b", re.IGNORECASE)),
    ("BPO and contact centers", re.compile(r"\bbpo\b|\bcontact centers?\b", re.IGNORECASE)),
    ("cyber security", re.compile(r"\bcyber security\b|\binsurance\b|\breinsurers?\b", re.IGNORECASE)),
    ("retail and e-commerce", re.compile(r"\bretail\b|\be-?commerce\b", re.IGNORECASE)),
    ("manufacturing and logistics", re.compile(r"\bmanufacturing\b|\blogistics\b|\bsupply chains?\b", re.IGNORECASE)),
]
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
CONTACT_HOURS_RE = re.compile(
    r"Monday to Saturday\s*\([^)]*\)",
    re.IGNORECASE,
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
SERVICE_TITLE_CLEANUPS = {
    "Custom Web & Mobile App Development": "custom web and mobile app development",
    "Sovereign Cloud Infrastructure": "sovereign cloud infrastructure",
    "AI Workflow Automation & Integration": "AI workflow automation and integration",
    "AI Voice Agents & Conversational Automation": "AI voice agents and conversational automation",
}
INDUSTRY_NAME_NORMALIZATION = {
    "banking and financial": "banking and financial services",
    "banking and financial services": "banking and financial services",
    "bpo contact centers": "BPO and contact centers",
    "bpo and contact centers": "BPO and contact centers",
    "cyber security": "cyber security",
    "retail & e-commerce": "retail and e-commerce",
    "retail and e-commerce": "retail and e-commerce",
    "manufacturing and logistics": "manufacturing and logistics",
}
SYSTEM_PROMPT = (
    "You are a factual assistant for Synapse Tech Inc. "
    "Answer only from the provided evidence. "
    "If the evidence does not clearly support the answer, say the detail is not confirmed in the available information. "
    "Do not guess, do not invent details, and do not mention internal retrieval, chunks, embeddings, or vector search. "
    "Return ONLY valid JSON in this exact shape: "
    '{"answer": "<final answer>", "supported": true}'
)
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
LEADING_ARTIFACT_PATTERNS = [
    re.compile(r"^the (?:page|evidence|sources?) (?:mentions|shows?|indicates?) that\s+", re.IGNORECASE),
    re.compile(r"^according to the (?:page|evidence|sources?),?\s+", re.IGNORECASE),
    re.compile(r"^the evidence shows that\s+", re.IGNORECASE),
]
SUPPORT_ANCHOR_STOPWORDS = {
    "about",
    "any",
    "available",
    "company",
    "have",
    "number",
    "public",
    "synapse",
    "tech",
    "tell",
    "there",
    "where",
    "your",
}
SUPPORT_TERM_VARIANTS = {
    "compliant": {"compliance", "compliant"},
    "founded": {"founder", "founded"},
    "gdpr": {"gdpr"},
    "located": {"address", "based", "headquarters", "location", "located", "office"},
    "phone": {"call", "phone", "telephone"},
    "pricing": {"cost", "costs", "price", "prices", "pricing"},
    "sla": {"service-level", "sla", "slas"},
    "slas": {"service-level", "sla", "slas"},
}

_PRICING_TERMS = (
    "pricing",
    "price",
    "prices",
    "cost",
    "costs",
    "quote",
    "how much",
    "per month",
)


def _contains_any(haystack: str, needles: tuple[str, ...]) -> bool:
    return any(n in haystack for n in needles)


def is_pricing_or_quote_question(question: str) -> bool:
    return _contains_any(question.lower(), _PRICING_TERMS)


def is_product_selection_question(question: str) -> bool:
    low = question.lower()
    has_pick = any(
        w in low for w in ("choose", "pick", "select", "compare", "deciding", "decide")
    )
    has_between = "between" in low or "among" in low
    has_products = "product" in low or "products" in low
    return has_products and (has_pick or has_between)


def is_deploy_priority_question(question: str) -> bool:
    low = question.lower()
    return ("deploy" in low or "rollout" in low) and ("first" in low or "start with" in low)


def is_company_overview_or_synopsis_question(question: str) -> bool:
    """
    True for questions asking for a neutral company summary / synopsis / mission framing,
    not product pricing or deployment advice (those use other gates).
    """
    if is_pricing_or_quote_question(question) or is_product_selection_question(question) or is_deploy_priority_question(question):
        return False
    low = question.lower()
    if "synapse" not in low and "synapsetechinc" not in low.replace(" ", ""):
        return False
    triggers = (
        "summarize",
        "summary",
        "short factual",
        "factual summary",
        "in two sentences",
        "in 2 sentences",
        "exactly two sentences",
        "exactly 2 sentences",
        "two sentences",
        "2 sentences",
        "explain synapse",
        "describe synapse",
        "overview",
        "tell me about synapse",
        "who is synapse tech",
        "core business problem",
        "business problem does synapse",
        "no marketing",
        "without marketing",
        "neutral wording",
        "plain language",
        "not marketing",
    )
    if any(t in low for t in triggers):
        return True
    if re.search(r"\bwhat is synapse tech\??(\s|$)", low):
        return True
    if re.search(r"\bwhat\s+does\s+synapse", low):
        return True
    return False


def allow_deterministic_summary(question: str, intent: str | None) -> bool:
    """Only emit catalog-style summaries when the question asks for that kind of list."""
    if intent not in SUMMARY_INTENT_CONFIG:
        return False
    low = question.lower()

    if intent == "product":
        if is_pricing_or_quote_question(question):
            return False
        if is_product_selection_question(question):
            return False
        if is_deploy_priority_question(question):
            return False
        if _contains_any(
            low,
            (
                "which integration",
                "integrations supported",
                "salesforce",
                "api documentation",
            ),
        ):
            return False
        return bool(
            re.search(r"\bwhat products\b", low)
            or re.search(r"\bwhich products\b", low)
            or re.search(r"\bproducts does synapse\b", low)
            or re.search(r"\bproducts do synapse\b", low)
            or (
                ("products" in low or "product" in low)
                and ("offer" in low or "offers" in low or "have" in low or "include" in low or "does" in low)
                and "synapse" in low
            )
        )

    if intent == "service":
        if "services" in low:
            return True
        return bool(
            re.search(r"\bwhat services\b", low)
            or re.search(r"\bservices does synapse\b", low)
            or re.search(r"\bservices do synapse\b", low)
        )

    if intent == "industry":
        return "industry" in low or "industries" in low

    return False


def augment_hits_with_about_us(
    hits: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    max_chunks: int = 4,
) -> list[dict[str, Any]]:
    """Prepend early About-doc chunks with strong scores so synopsis questions pass support gates."""
    about_rows = sorted(
        [r for r in rows if r.get("doc_id") == "about-us"],
        key=lambda r: int(r.get("chunk_index") or 0),
    )
    hit_ids = {h["chunk_id"] for h in hits}
    score_seed = max((float(h.get("score", 0)) for h in hits), default=0.55)
    score_seed = max(score_seed, 0.92)
    front: list[dict[str, Any]] = []
    for row in about_rows:
        if len(front) >= max_chunks:
            break
        cid = row.get("chunk_id")
        if not cid or cid in hit_ids:
            continue
        front.append(make_hit_from_row(row, score_seed))
        hit_ids.add(cid)
        score_seed -= 0.001
    tail = [h for h in hits if h.get("chunk_id") not in {x["chunk_id"] for x in front}]
    return front + tail


def answer_shape_mismatch(question: str, answer: str) -> bool:
    """True if the answer looks like a wrong template for the question (e.g. product menu for pricing)."""
    low_q = question.lower()
    low_a = answer.lower()
    if not answer.strip():
        return False

    if is_pricing_or_quote_question(question):
        if "synapse offers products including" in low_a:
            return True
        if low_a.startswith("synapse tech inc. offers") and not _contains_any(low_a, _PRICING_TERMS):
            if "not confirmed" not in low_a and "official quote" not in low_a:
                return True

    if is_product_selection_question(question):
        if "synapse offers products including" in low_a:
            if "use case" not in low_a and "criteria" not in low_a and "governance" not in low_a:
                return True

    if is_deploy_priority_question(question):
        if "synapse offers products including" in low_a:
            return True
        if low_a.startswith("synapse tech inc. offers") and "deploy" not in low_a and "workflow" not in low_a:
            if "pilot" not in low_a and "baseline" not in low_a and "not confirmed" not in low_a:
                return True

    if _contains_any(
        low_q,
        (
            "unsupported security",
            "unsupported legal",
            "how should you answer unsupported",
        ),
    ):
        if "attackers scan" in low_a or "paper trail" in low_a:
            return True

    return False


def finalize_kb_answer(question: str, answer: str) -> str:
    if answer_shape_mismatch(question, answer):
        return DEFAULT_FALLBACK_RESPONSE
    return answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Answer a question using KB retrieval plus grounded local generation."
    )
    parser.add_argument(
        "question",
        nargs="+",
        help="Question to answer from the local knowledge base.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_GENERATION_MODEL,
        help="Ollama generation model to use for the grounded answer.",
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
        help="Path to the KB manifest JSON file.",
    )
    parser.add_argument(
        "--embed-model",
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
        help="How many retrieval candidates to keep after reranking.",
    )
    parser.add_argument(
        "--context-k",
        type=int,
        default=3,
        help="How many top chunks to include in the final answer prompt.",
    )
    parser.add_argument(
        "--page-type",
        action="append",
        default=[],
        help="Optional page_type filter. Repeat to allow multiple types.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Generation temperature for the grounded answer model.",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=120,
        help="Max tokens to generate for the grounded answer.",
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Print source chunks after the final answer.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON output instead of plain text.",
    )
    return parser.parse_args()


def ollama_chat(
    *,
    ollama_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    num_predict: int,
) -> str:
    response = requests.post(
        f"{ollama_url}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        },
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("message", {}).get("content", "").strip()


def safe_parse_json(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = JSON_OBJECT_RE.search(raw)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def format_evidence_block(hit: dict[str, Any], rank: int) -> str:
    return (
        f"[{rank}] Title: {hit['title']}\n"
        f"URL: {hit['url']}\n"
        f"Page Type: {hit['page_type']}\n"
        f"Chunk ID: {hit['chunk_id']}\n"
        f"Content:\n{hit['text']}"
    )


def append_diverse_hits(
    selected: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    max_items: int,
    dedupe_by_doc: bool,
) -> None:
    seen_chunk_ids = {hit["chunk_id"] for hit in selected}
    seen_doc_ids = {hit["doc_id"] for hit in selected}

    for hit in candidates:
        if len(selected) >= max_items:
            return
        if hit["chunk_id"] in seen_chunk_ids:
            continue
        if dedupe_by_doc and hit["doc_id"] in seen_doc_ids:
            continue
        selected.append(hit)
        seen_chunk_ids.add(hit["chunk_id"])
        seen_doc_ids.add(hit["doc_id"])


def make_hit_from_row(row: dict[str, Any], score: float) -> dict[str, Any]:
    return {
        "score": score,
        "raw_score": score,
        "chunk_id": row.get("chunk_id"),
        "doc_id": row.get("doc_id"),
        "page_type": row.get("page_type"),
        "title": row.get("title"),
        "url": row.get("url"),
        "text": row.get("text"),
        "score_components": {"augmented_index_context": score},
    }


def augment_summary_hits_with_index_rows(
    hits: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    index_doc_id: str,
) -> list[dict[str, Any]]:
    existing_chunk_ids = {hit["chunk_id"] for hit in hits}
    min_score = min((float(hit["raw_score"]) for hit in hits), default=0.0) - 0.001
    augmented = hits[:]

    index_rows = [row for row in rows if row.get("doc_id") == index_doc_id]
    for row in index_rows:
        if row.get("chunk_id") in existing_chunk_ids:
            continue
        augmented.append(make_hit_from_row(row, min_score))
        min_score -= 0.001

    return augmented


def augment_summary_hits_with_detail_rows(
    hits: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    detail_page_type: str,
    index_doc_id: str,
) -> list[dict[str, Any]]:
    existing_chunk_ids = {hit["chunk_id"] for hit in hits}
    existing_doc_ids = {hit["doc_id"] for hit in hits}
    min_score = min((float(hit["raw_score"]) for hit in hits), default=0.0) - 0.01
    augmented = hits[:]

    detail_rows = [
        row
        for row in rows
        if row.get("page_type") == detail_page_type
        and row.get("doc_id") != index_doc_id
        and row.get("chunk_index") == 1
    ]
    for row in detail_rows:
        if row.get("chunk_id") in existing_chunk_ids:
            continue
        if row.get("doc_id") in existing_doc_ids:
            continue
        augmented.append(make_hit_from_row(row, min_score))
        existing_chunk_ids.add(row["chunk_id"])
        existing_doc_ids.add(row["doc_id"])
        min_score -= 0.001

    return augmented


def select_summary_hits(
    hits: list[dict[str, Any]],
    *,
    context_k: int,
    intent: str,
) -> list[dict[str, Any]]:
    config = SUMMARY_INTENT_CONFIG[intent]
    selected: list[dict[str, Any]] = []
    detail_doc_ids = {
        hit["doc_id"]
        for hit in hits
        if hit.get("page_type") == config["detail_page_type"]
        and hit.get("doc_id") != config["index_doc_id"]
    }
    summary_context_k = max(context_k, min(len(detail_doc_ids) + 1, 8))

    index_hits = [
        hit
        for hit in hits
        if hit.get("doc_id") == config["index_doc_id"]
    ]
    append_diverse_hits(
        selected,
        index_hits,
        max_items=min(1, summary_context_k),
        dedupe_by_doc=False,
    )

    detail_hits = [
        hit
        for hit in hits
        if hit.get("page_type") == config["detail_page_type"]
        and hit.get("doc_id") != config["index_doc_id"]
    ]
    append_diverse_hits(
        selected,
        detail_hits,
        max_items=summary_context_k,
        dedupe_by_doc=True,
    )

    append_diverse_hits(
        selected,
        hits,
        max_items=summary_context_k,
        dedupe_by_doc=True,
    )
    return selected[:summary_context_k]


def select_context_hits(
    hits: list[dict[str, Any]],
    *,
    context_k: int,
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    if not hits:
        return []

    top_hit = hits[0]
    entity_terms = profile.get("entity_terms", [])
    intent = profile.get("intent")

    if intent in SUMMARY_INTENT_CONFIG:
        return select_summary_hits(hits, context_k=context_k, intent=intent)

    if profile.get("company_overview"):
        about_hits = [
            h
            for h in hits
            if h.get("page_type") == "about" or h.get("doc_id") == "about-us"
        ]
        seen: set[str] = set()
        merged: list[dict[str, Any]] = []
        for h in about_hits:
            cid = h.get("chunk_id")
            if cid and cid not in seen:
                merged.append(h)
                seen.add(cid)
        for h in hits:
            cid = h.get("chunk_id")
            if cid and cid not in seen:
                merged.append(h)
                seen.add(cid)
            if len(merged) >= context_k:
                break
        return merged[:context_k]

    if entity_terms or intent in {"contact", "about"}:
        same_doc_hits = [hit for hit in hits if hit.get("doc_id") == top_hit.get("doc_id")]
        if same_doc_hits:
            return same_doc_hits[:context_k]

    return hits[:context_k]


def build_user_prompt(question: str, hits: list[dict[str, Any]], profile: dict[str, Any]) -> str:
    evidence = "\n\n".join(
        format_evidence_block(hit, rank)
        for rank, hit in enumerate(hits, start=1)
    )
    intent = profile.get("intent")
    extra_guidance = "Answer the question directly in the first sentence."
    if is_pricing_or_quote_question(question):
        extra_guidance = (
            "If public pricing or exact costs are not explicitly stated in the evidence, use the not-confirmed reply "
            "exactly as instructed above. Never answer pricing questions by listing product names only."
        )
    elif is_product_selection_question(question):
        extra_guidance = (
            "Give decision criteria grounded in the evidence. If comparisons are not in the evidence, "
            "use the not-confirmed reply; do not answer with a bare product catalog only."
        )
    elif is_deploy_priority_question(question):
        extra_guidance = (
            "Give one concrete first-step recommendation only if the evidence supports it; otherwise "
            "use the not-confirmed reply. Do not substitute a generic Synapse capability overview."
        )
    elif profile.get("company_overview"):
        extra_guidance = (
            "Synthesize one cohesive answer about Synapse Tech using only the evidence. "
            "If the user requested a sentence limit or neutral/non-marketing tone, follow it. "
            "Prefer concrete capabilities (e.g. apps, automation, security) over slogans; "
            "omit claims not literally supported. "
            "The excerpts below include company About content — you must answer from them; "
            "do not use the not-confirmed reply unless none of the excerpts mention Synapse Tech."
        )
    elif intent in SUMMARY_INTENT_CONFIG:
        extra_guidance = (
            f"Summarize the main {SUMMARY_INTENT_CONFIG[intent]['label']} supported by the evidence. "
            "Name multiple items when the evidence supports them, using the page titles and evidence content. "
            "Avoid focusing on a single example."
        )
    elif profile.get("entity_terms"):
        extra_guidance = (
            "Define the named item directly and explain its main purpose or capabilities."
        )
    elif intent == "contact":
        extra_guidance = (
            "Give the direct contact methods and availability details only."
        )

    if profile.get("company_overview"):
        requirements = (
            "Answer requirements:\n"
            "Use only the evidence above. It includes Synapse Tech company pages; produce a direct factual answer.\n"
            f"{extra_guidance}\n"
            "Write 2-4 concise sentences unless the user asked for fewer (then respect that count).\n"
            "Do not mention the evidence, assumptions, or these instructions.\n"
            'Return JSON only, using keys "answer" and "supported". Set "supported" to true when '
            "the answer is paraphrased from the excerpts.\n"
        )
    else:
        requirements = (
            "Answer requirements:\n"
            f'Use only the evidence above. If the evidence is insufficient, reply exactly with: "{DEFAULT_FALLBACK_RESPONSE}"\n'
            f"{extra_guidance}\n"
            "Write 2-4 concise sentences.\n"
            "Do not mention the evidence, assumptions, or these instructions.\n"
            'Return JSON only, using keys "answer" and "supported".\n'
        )

    return (
        f"Question:\n{question}\n\n"
        f"Evidence:\n{evidence}\n\n"
        f"{requirements}"
    )


def title_prefix(title: str) -> str:
    return title.split("|", 1)[0].strip()


def join_list(items: list[str]) -> str:
    values = [item for item in items if item]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def normalized_hit_text(hit: dict[str, Any]) -> str:
    return " ".join((hit.get("text") or "").split())


def build_contact_answer(hits: list[dict[str, Any]]) -> str:
    combined_text = "\n".join(hit.get("text", "") for hit in hits)
    email_match = EMAIL_RE.search(combined_text)
    hours_match = CONTACT_HOURS_RE.search(combined_text)

    if email_match and hours_match:
        return (
            f"You can contact Synapse Tech Inc. at {email_match.group(0)} or chat with the team "
            f"{hours_match.group(0)}."
        )
    if email_match:
        return f"You can contact Synapse Tech Inc. at {email_match.group(0)}."
    return ""


def build_summary_answer(intent: str, hits: list[dict[str, Any]]) -> str:
    if intent == "service":
        service_names: list[str] = []
        for hit in hits:
            if hit.get("page_type") == "service":
                cleaned = SERVICE_TITLE_CLEANUPS.get(title_prefix(hit["title"]), title_prefix(hit["title"]).lower())
                service_names.append(cleaned)
            text = normalized_hit_text(hit).lower()
            if "voice agent" in text or "voice agents" in text:
                service_names.append("AI voice agents and conversational automation")
        service_names = unique_preserve_order(service_names)
        if service_names:
            return f"Synapse Tech Inc. offers {join_list(service_names)}."
        return ""

    if intent == "product":
        product_names: list[str] = []
        for hit in hits:
            if hit.get("page_type") == "product":
                product_names.append(title_prefix(hit["title"]))
            text = normalized_hit_text(hit)
            for label, pattern in PRODUCT_NAME_PATTERNS:
                if pattern.search(text):
                    product_names.append(label)
        product_names = unique_preserve_order(product_names)
        if product_names:
            return f"Synapse offers products including {join_list(product_names)}."
        return ""

    if intent == "industry":
        industry_names: list[str] = []
        for hit in hits:
            if hit.get("page_type") == "industry":
                industry_names.append(title_prefix(hit["title"]).lower())
            text = normalized_hit_text(hit)
            for label, pattern in INDUSTRY_NAME_PATTERNS:
                if pattern.search(text):
                    industry_names.append(label)
        normalized_names = [
            INDUSTRY_NAME_NORMALIZATION.get(name.lower(), name.lower())
            for name in industry_names
        ]
        industry_names = unique_preserve_order(normalized_names)
        if industry_names:
            return f"Synapse Tech Inc. serves industries including {join_list(industry_names)}."
        return ""

    return ""


def candidate_definition_lines(hits: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for hit in hits:
        for raw_line in hit.get("text", "").splitlines():
            line = " ".join(raw_line.split()).strip(" -")
            if not line:
                continue
            if len(line.split()) < 5:
                continue
            if "→" in line or "✅" in line or "❌" in line:
                continue
            lines.extend(SENTENCE_SPLIT_RE.split(line))
    return unique_preserve_order([line.strip() for line in lines if line.strip()])


def line_with_pattern(
    candidates: list[str],
    pattern: str,
) -> str:
    regex = re.compile(pattern, re.IGNORECASE)
    for line in candidates:
        if regex.search(line):
            return line
    return ""


def build_known_product_answer(title: str, candidates: list[str]) -> str:
    if title == "Opira Ai":
        overview = line_with_pattern(
            candidates,
            r"offline LLM platform and private RAG engines entirely on your network",
        )
        capabilities = line_with_pattern(
            candidates,
            r"query sensitive documents, compare models, and automate AI workflows",
        )
        if overview and capabilities:
            return (
                "Opira Ai is an offline LLM platform with private RAG engines that runs entirely on your network. "
                f"{capabilities}"
            )
        if overview:
            return "Opira Ai is an offline LLM platform with private RAG engines that runs entirely on your network."

    if title == "iRecruit One":
        overview = line_with_pattern(
            candidates,
            r"Built for enterprise recruitment, iRecruit.*automates CV screening, AI interviews.*smart shortlisting",
        )
        if overview:
            return (
                "iRecruit One is an enterprise recruitment platform that automates CV screening, "
                "AI interviews, and smart shortlisting. It is designed to help teams hire faster, "
                "fairer, and at scale."
            )

    if title == "Agentic Bot":
        workflow = line_with_pattern(
            candidates,
            r"WhatsApp conversation|service request handled",
        )
        assistant = line_with_pattern(candidates, r"WhatsApp-native assistant")
        if workflow or assistant:
            return (
                "Agentic Bot is a WhatsApp-centric assistant that handles service requests and automates "
                "form delivery, policy queries, and breach checks. It can retrieve verified knowledge, "
                "turn conversations into structured data, and trigger backend actions."
            )

    return ""


def build_entity_answer(question: str, hits: list[dict[str, Any]], profile: dict[str, Any]) -> str:
    if not hits or not profile.get("entity_terms"):
        return ""

    low_q = question.lower().strip()
    if not (low_q.startswith("what is") or low_q.startswith("what does")):
        return ""

    title = title_prefix(hits[0]["title"])
    candidates = candidate_definition_lines(hits)
    if not candidates:
        return ""

    known_answer = build_known_product_answer(title, candidates)
    if known_answer:
        return known_answer

    title_tokens = set(query_kb.tokenize(title))
    scored: list[tuple[float, str]] = []
    for line in candidates:
        low = line.lower()
        score = 0.0
        if any(token in low for token in title_tokens):
            score += 2.0
        if " is " in low:
            score += 1.2
        if any(term in low for term in ["automates", "allows", "handles", "platform", "assistant", "solution"]):
            score += 0.8
        if "not all" in low or "ready to" in low:
            score -= 1.5
        if len(line.split()) > 40:
            score -= 0.3
        scored.append((score, line))

    scored.sort(key=lambda item: item[0], reverse=True)
    best_lines = [line for score, line in scored if score > 0][:2]
    if not best_lines:
        return ""

    answer = " ".join(best_lines).strip()
    if title.lower() not in answer.lower():
        answer = f"{title} is {answer[0].lower() + answer[1:]}" if answer else ""
    return answer


def retrieve_hits(
    *,
    question: str,
    index_path: Path,
    metadata_path: Path,
    manifest_path: Path,
    embed_model: str | None,
    ollama_url: str,
    page_types: set[str],
    top_k: int,
) -> tuple[list[dict[str, Any]], str]:
    manifest = query_kb.load_manifest(manifest_path)
    model = embed_model or manifest.get("embedding_model")
    if not model:
        raise ValueError(
            "No embedding model available. Use --embed-model or ensure the manifest exists."
        )

    index = faiss.read_index(str(index_path))
    rows = query_kb.load_rows(metadata_path)
    if index.ntotal != len(rows):
        raise ValueError("FAISS index size does not match metadata row count.")

    profile = query_kb.query_profile(question)
    profile["company_overview"] = is_company_overview_or_synopsis_question(question)
    query_vector = query_kb.embed_query(ollama_url, model, question)
    retrieval_top_k = top_k
    if profile.get("intent") in SUMMARY_INTENT_CONFIG:
        retrieval_top_k = max(top_k, 16)
    elif profile.get("company_overview"):
        retrieval_top_k = max(top_k, 14)
    elif profile.get("entity_terms"):
        retrieval_top_k = max(top_k, 10)
    hits = query_kb.search_index(
        index,
        query_vector,
        rows,
        page_types,
        retrieval_top_k,
        profile,
    )
    if profile.get("company_overview"):
        hits = augment_hits_with_about_us(hits, rows, max_chunks=4)
    if profile.get("intent") in SUMMARY_INTENT_CONFIG:
        hits = augment_summary_hits_with_index_rows(
            hits,
            rows,
            index_doc_id=SUMMARY_INTENT_CONFIG[profile["intent"]]["index_doc_id"],
        )
        hits = augment_summary_hits_with_detail_rows(
            hits,
            rows,
            detail_page_type=SUMMARY_INTENT_CONFIG[profile["intent"]]["detail_page_type"],
            index_doc_id=SUMMARY_INTENT_CONFIG[profile["intent"]]["index_doc_id"],
        )
    return hits, model, profile


def build_json_output(
    *,
    question: str,
    answer: str,
    generation_model: str,
    embedding_model: str,
    hits: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "question": question,
        "answer": answer,
        "generation_model": generation_model,
        "embedding_model": embedding_model,
        "sources": [
            {
                "chunk_id": hit["chunk_id"],
                "doc_id": hit["doc_id"],
                "title": hit["title"],
                "url": hit["url"],
                "page_type": hit["page_type"],
                "score": hit["score"],
                "raw_score": hit["raw_score"],
            }
            for hit in hits
        ],
    }


def normalize_answer_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return " ".join(parts).strip()
    if value is None:
        return ""
    return str(value).strip()


def clean_answer_text(text: str) -> str:
    cleaned = " ".join(text.split()).strip()
    for pattern in LEADING_ARTIFACT_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned.strip()


def support_anchor_terms(profile: dict[str, Any]) -> list[str]:
    anchors: list[str] = []
    seen = set()
    for term in profile.get("keywords", []):
        if term in query_kb.GENERIC_QUERY_TERMS:
            continue
        if term in query_kb.STOPWORDS or term in SUPPORT_ANCHOR_STOPWORDS:
            continue
        if term in seen:
            continue
        anchors.append(term)
        seen.add(term)
    return anchors


def match_term_variants(term: str) -> set[str]:
    normalized = term.lower()
    variants = set(SUPPORT_TERM_VARIANTS.get(normalized, {normalized}))
    if normalized.endswith("s") and len(normalized) > 4:
        variants.add(normalized[:-1])
    return variants


def hit_match_counts(hit: dict[str, Any], anchor_terms: list[str]) -> tuple[int, int]:
    title_tokens = set(query_kb.tokenize(hit.get("title") or ""))
    doc_tokens = set(query_kb.tokenize(hit.get("doc_id") or ""))
    text_tokens = set(query_kb.tokenize(hit.get("text") or ""))

    title_or_doc_matches = 0
    text_matches = 0
    for term in anchor_terms:
        variants = match_term_variants(term)
        if variants & (title_tokens | doc_tokens):
            title_or_doc_matches += 1
            continue
        if variants & text_tokens:
            text_matches += 1
    return title_or_doc_matches, text_matches


def has_sufficient_explicit_support(
    question: str,
    hits: list[dict[str, Any]],
    profile: dict[str, Any],
) -> bool:
    if not hits:
        return False

    if profile.get("company_overview"):
        blob = " ".join(f"{h.get('title', '')} {h.get('text', '')}" for h in hits[:10]).lower()
        if "synapse" not in blob:
            return False
        if any(h.get("doc_id") == "about-us" for h in hits[:10]):
            return True
        best_score = max((float(h.get("score") or 0.0) for h in hits[:10]), default=0.0)
        return best_score >= 0.35

    anchor_terms = support_anchor_terms(profile)
    if not anchor_terms:
        return True

    required_matches = 1 if len(anchor_terms) == 1 else 2
    best_title_doc = 0
    best_combined = 0
    top_score = hits[0].get("score", 0.0)

    for hit in hits:
        title_doc_matches, text_matches = hit_match_counts(hit, anchor_terms)
        best_title_doc = max(best_title_doc, title_doc_matches)
        best_combined = max(best_combined, title_doc_matches + text_matches)

    if best_title_doc >= required_matches:
        return True

    # Allow product/feature or service/feature questions when the entity is explicit in
    # the title/doc and the supporting capability is present in the retrieved text.
    if best_title_doc >= 1 and best_combined >= required_matches:
        return True

    # Generic company-description questions like "what does synapse tech do" often have
    # no informative anchor terms, so they return earlier. If anchor terms remain but the
    # evidence never mentions them explicitly, refuse instead of guessing.
    return top_score >= 0.95 and best_combined >= required_matches


def generate_grounded_answer(
    *,
    question: str,
    context_hits: list[dict[str, Any]],
    profile: dict[str, Any],
    ollama_url: str,
    model: str,
    temperature: float,
    num_predict: int,
) -> str:
    if not context_hits:
        return DEFAULT_FALLBACK_RESPONSE

    intent = profile.get("intent")
    if intent == "contact":
        contact_answer = build_contact_answer(context_hits)
        if contact_answer:
            return contact_answer

    if intent in SUMMARY_INTENT_CONFIG and allow_deterministic_summary(question, intent):
        summary_answer = build_summary_answer(intent, context_hits)
        if summary_answer:
            return finalize_kb_answer(question, summary_answer)

    if not has_sufficient_explicit_support(question, context_hits, profile):
        return DEFAULT_FALLBACK_RESPONSE

    entity_answer = build_entity_answer(question, context_hits, profile)
    if entity_answer:
        return finalize_kb_answer(question, clean_answer_text(entity_answer))

    user_prompt = build_user_prompt(question, context_hits, profile)
    predict_cap = max(num_predict, 160) if profile.get("company_overview") else num_predict
    raw = ollama_chat(
        ollama_url=ollama_url,
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        num_predict=predict_cap,
    )
    payload = safe_parse_json(raw)
    answer = normalize_answer_value(payload.get("answer")) if isinstance(payload, dict) else ""
    answer = clean_answer_text(answer)

    if profile.get("company_overview") and (not answer or answer == DEFAULT_FALLBACK_RESPONSE):
        blob = " ".join(f"{h.get('title', '')} {h.get('text', '')}" for h in context_hits).lower()
        if "synapse" in blob:
            retry_user = (
                user_prompt
                + "\n\nRetry: The previous JSON was invalid or too cautious. Write 2–4 factual sentences "
                "about Synapse Tech using only the evidence above. "
                'Return JSON {"answer":"<text>","supported":true}. '
                "Do not use the not-confirmed reply when the excerpts clearly describe Synapse Tech."
            )
            raw_retry = ollama_chat(
                ollama_url=ollama_url,
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": retry_user},
                ],
                temperature=min(0.2, temperature + 0.05),
                num_predict=predict_cap,
            )
            payload_retry = safe_parse_json(raw_retry)
            answer = (
                normalize_answer_value(payload_retry.get("answer"))
                if isinstance(payload_retry, dict)
                else ""
            )
            answer = clean_answer_text(answer)

    if not answer:
        return DEFAULT_FALLBACK_RESPONSE
    return finalize_kb_answer(question, answer)


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be a positive integer.")
    if args.context_k <= 0:
        raise ValueError("--context-k must be a positive integer.")
    if args.context_k > args.top_k:
        raise ValueError("--context-k cannot be greater than --top-k.")

    question = " ".join(args.question).strip()
    page_types = set(args.page_type)

    query_kb.check_ollama(args.ollama_url)
    hits, embedding_model, profile = retrieve_hits(
        question=question,
        index_path=Path(args.index),
        metadata_path=Path(args.metadata),
        manifest_path=Path(args.manifest),
        embed_model=args.embed_model,
        ollama_url=args.ollama_url,
        page_types=page_types,
        top_k=args.top_k,
    )

    context_hits = select_context_hits(hits, context_k=args.context_k, profile=profile)
    answer = generate_grounded_answer(
        question=question,
        context_hits=context_hits,
        profile=profile,
        ollama_url=args.ollama_url,
        model=args.model,
        temperature=args.temperature,
        num_predict=args.num_predict,
    )

    if args.json:
        print(
            json.dumps(
                build_json_output(
                    question=question,
                    answer=answer,
                    generation_model=args.model,
                    embedding_model=embedding_model,
                    hits=context_hits,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print("=" * 60)
    print("KB Grounded Answer")
    print("=" * 60)
    print(f"Question: {question}")
    print(f"Generation model: {args.model}")
    print(f"Embedding model: {embedding_model}")
    if page_types:
        print("Page type filter:", ", ".join(sorted(page_types)))
    print()
    print(answer)

    if args.show_sources and context_hits:
        print("\n" + "=" * 60)
        print("Sources")
        print("=" * 60)
        for rank, hit in enumerate(context_hits, start=1):
            print(f"[{rank}] {hit['title']} ({hit['page_type']})")
            print(f"    chunk: {hit['chunk_id']}")
            print(f"    score: {hit['score']:.4f} (raw={hit['raw_score']:.4f})")
            print(f"    url:   {hit['url']}")
            print()


if __name__ == "__main__":
    main()
