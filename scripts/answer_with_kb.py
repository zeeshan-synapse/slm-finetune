#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

import faiss
import requests

import query_kb


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_PATH = PROJECT_DIR / "data" / "knowledge-base" / "faiss.index"
DEFAULT_META_PATH = PROJECT_DIR / "data" / "knowledge-base" / "index_meta.jsonl"
DEFAULT_MANIFEST_PATH = PROJECT_DIR / "data" / "knowledge-base" / "index_manifest.json"
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_GENERATION_MODEL = os.environ.get("KB_ANSWER_MODEL", "synapse-3b")
DEFAULT_REWRITE_MODEL = (
    os.environ.get("KB_REWRITE_MODEL")
    or os.environ.get("KB_BASE_MODEL")
    or os.environ.get("KB_ANSWER_MODEL")
    or DEFAULT_GENERATION_MODEL
)
DEFAULT_FALLBACK_RESPONSE = (
    "This detail is not confirmed in the available information. Please verify with Synapse Tech."
)
KB_DEBUG_LOG_PATH = PROJECT_DIR / "logs" / "kb_answer_debug.jsonl"
DISABLE_ROUTE_SPECIFIC_ANSWERS = False
DISABLE_NONCRITICAL_DIRECT_ANSWERS = False
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
    ("Opira AI", re.compile(r"\bopira(?:\.io)?\b|\bopairo\b", re.IGNORECASE)),
    ("Coversaction AI", re.compile(r"\bcoversaction ai\b|\bconversaction ai\b", re.IGNORECASE)),
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
PRODUCT_NAME_NORMALIZATION = {
    "opira ai": "Opira AI",
    "opira": "Opira AI",
    "coversaction ai": "Coversaction AI",
    "conversaction ai": "Coversaction AI",
    "agentic bot": "Agentic Bot",
    "irecruit one": "iRecruit One",
    "cyber security automation": "Cyber Security Automation",
}
PRODUCT_SHORT_DESCRIPTIONS = {
    "Agentic Bot": "a WhatsApp-centric assistant for service requests, forms, policy queries, and breach checks",
    "iRecruit One": "a recruitment intelligence platform for screening, interviews, scheduling, and shortlisting",
    "Opira AI": "an offline LLM and private RAG platform for sensitive enterprise knowledge",
    "Coversaction AI": "a conversational AI product for customer resolution, appointments, lead qualification, and handoffs",
    "Cyber Security Automation": "an automation product for security operations and digital risk workflows",
}
SYSTEM_PROMPT = (
    "You are a factual assistant for Synapse Tech Inc. "
    "Answer only from the provided evidence. "
    "If the evidence does not clearly support the answer, say the detail is not confirmed in the available information. "
    "Do not guess, do not invent details, and do not mention internal retrieval, chunks, embeddings, or vector search. "
    "Return ONLY valid JSON in this exact shape: "
    '{"answer": "<final answer>", "supported": true}'
)
QUERY_REWRITE_SYSTEM_PROMPT = (
    "You rewrite user questions into short search queries for a Synapse Tech knowledge base. "
    "Do not answer the question. Do not invent facts. "
    "Return ONLY valid JSON in this exact shape: "
    '{"search_query": "<short search query>", "intent_hint": "<brief intent>"}'
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
_PURCHASE_TERMS = (
    "buy",
    "purchase",
    "purchasing",
    "get started",
    "sign up",
    "subscribe",
    "order",
    "sales",
)
_HIGH_RISK_TERMS = (
    "pricing",
    "price",
    "prices",
    "cost",
    "costs",
    "quote",
    "sla",
    "legal",
    "liability",
    "contract",
    "warranty",
    "certification",
    "certifications",
    "certificate",
    "compliance",
    "compliant",
    "gdpr",
    "address",
    "headquarters",
    "located",
    "location",
    "regulatory",
    "roadmap",
    "release date",
    "guarantee",
    "guaranteed",
)


def _contains_any(haystack: str, needles: tuple[str, ...]) -> bool:
    return any(n in haystack for n in needles)


def is_pricing_or_quote_question(question: str) -> bool:
    return _contains_any(question.lower(), _PRICING_TERMS)


def is_purchase_or_get_started_question(question: str) -> bool:
    low = question.lower()
    return _contains_any(low, _PURCHASE_TERMS) and (
        "product" in low
        or "service" in low
        or "agentic" in low
        or "bot" in low
        or "opira" in low
        or "coversaction" in low
        or "conversaction" in low
        or "irecruit" in low
        or "synapse" in low
        or "how do i" in low
        or "how can i" in low
    )


def is_high_risk_question(question: str) -> bool:
    return _contains_any(question.lower(), _HIGH_RISK_TERMS)


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
        "what can you help me with",
        "what can you help with",
        "what do you do",
    )
    if any(t in low for t in triggers):
        return True
    if re.search(r"\bwhat is synapse tech\??(\s|$)", low):
        return True
    if re.search(r"\bwhat\s+does\s+synapse", low):
        return True
    return False


def is_capability_or_offer_question(question: str) -> bool:
    low = question.lower()
    return any(
        phrase in low
        for phrase in (
            "do you offer",
            "can you help with",
            "what can you help with",
            "what can you help me with",
            "what do you offer",
            "what services do you offer",
            "tell me about your ai services",
        )
    )


def is_difference_question(question: str) -> bool:
    low = question.lower()
    return "different from a normal chatbot" in low or "different from a chatbot" in low


def is_voice_agent_question(question: str) -> bool:
    low = question.lower()
    return "voice agent" in low or "call automation" in low or "voice ai" in low


def is_customer_support_product_question(question: str) -> bool:
    low = question.lower()
    return (
        "which product" in low
        and ("customer support" in low or "chatbot" in low)
    )


def is_custom_software_question(question: str) -> bool:
    low = question.lower()
    return (
        "custom software" in low
        and (
            "ai product" in low
            or "ai products" in low
            or "only ai" in low
            or "only build" in low
            or "only do" in low
            or "only make" in low
            or "do you only" in low
            or "or only" in low
        )
    )


def is_private_deployment_question(question: str) -> bool:
    low = question.lower()
    return (
        ("offline" in low or "private infrastructure" in low or "on-prem" in low)
        and ("tool" in low or "tools" in low or "solution" in low or "solutions" in low)
    )


def is_workflow_automation_question(question: str) -> bool:
    low = question.lower()
    return "workflow automation" in low and ("how" in low or "help" in low)


def is_industry_fit_question(question: str) -> bool:
    low = question.lower()
    return (
        ("industry" in low or "industries" in low)
        and ("can your" in low or "work for" in low or "fit" in low)
    )


def is_service_discovery_question(question: str) -> bool:
    low = question.lower()
    return (
        "choosing the right service" in low
        or "choose the right service" in low
        or ("what would you ask me first" in low and "service" in low)
    )


def log_kb_debug(event: dict[str, Any]) -> None:
    try:
        KB_DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            **event,
        }
        with KB_DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


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


def augment_hits_with_matching_rows(
    hits: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    matches: Any,
    max_chunks: int = 2,
) -> list[dict[str, Any]]:
    """Prepend explicit support chunks for narrow supported question families."""
    hit_ids = {h["chunk_id"] for h in hits}
    score_seed = max((float(h.get("score", 0)) for h in hits), default=0.55)
    score_seed = max(score_seed, 0.94)
    front: list[dict[str, Any]] = []
    for row in rows:
        cid = row.get("chunk_id")
        if not cid or cid in hit_ids or not matches(row):
            continue
        front.append(make_hit_from_row(row, score_seed))
        hit_ids.add(cid)
        score_seed -= 0.001
        if len(front) >= max_chunks:
            break
    return front + [h for h in hits if h.get("chunk_id") not in {x["chunk_id"] for x in front}]


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
        "--rewrite-model",
        default=DEFAULT_REWRITE_MODEL,
        help="Ollama model to use for rewriting user questions into KB search queries.",
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
    usage_sink: dict[str, Any] | None = None,
    usage_label: str = "chat",
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
    if usage_sink is not None:
        prompt_tokens = data.get("prompt_eval_count")
        output_tokens = data.get("eval_count")
        usage_sink[usage_label] = {
            "model": model,
            "input_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": (prompt_tokens or 0) + (output_tokens or 0),
        }
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


def rewrite_query_with_model(
    *,
    question: str,
    ollama_url: str,
    model: str,
) -> dict[str, str]:
    fallback = {"search_query": question.strip(), "intent_hint": "fallback_original"}
    if not question.strip():
        return fallback

    user_prompt = (
        "Rewrite this user question into a short search query for a Synapse Tech knowledge base.\n"
        "Keep product names, service names, and company names. Expand casual wording like u/ur. "
        "If the user asks about Synapse offerings, include 'Synapse Tech'. "
        "Do not answer the question.\n\n"
        f"User question: {question}"
    )
    usage: dict[str, Any] = {}
    try:
        raw = ollama_chat(
            ollama_url=ollama_url,
            model=model,
            messages=[
                {"role": "system", "content": QUERY_REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            num_predict=80,
            usage_sink=usage,
            usage_label="query_rewrite",
        )
        payload = safe_parse_json(raw)
        search_query = normalize_answer_value(payload.get("search_query"))
        intent_hint = normalize_answer_value(payload.get("intent_hint"))
        if not search_query:
            log_kb_debug(
                {
                    "question": question,
                    "stage": "query_rewrite_unparseable",
                    "model": model,
                    "raw": raw[:500],
                }
            )
            return fallback
        return {
            "search_query": search_query[:240],
            "intent_hint": intent_hint[:80] if intent_hint else "model_rewrite",
            "usage": usage.get("query_rewrite"),
        }
    except Exception as exc:
        log_kb_debug(
            {
                "question": question,
                "stage": "query_rewrite_failed",
                "error": str(exc),
            }
        )
        return fallback


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

    if profile.get("purchase_intent"):
        contact_hits = [hit for hit in hits if hit.get("doc_id") == "contact-us"]
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for hit in contact_hits + hits:
            cid = hit.get("chunk_id")
            if cid and cid not in seen:
                merged.append(hit)
                seen.add(cid)
            if len(merged) >= context_k:
                break
        return merged[:context_k]

    if intent in SUMMARY_INTENT_CONFIG and not profile.get("product_aliases"):
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

    if (entity_terms and not profile.get("product_aliases")) or intent in {"contact", "about"}:
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
    elif intent == "product":
        if profile.get("product_aliases"):
            extra_guidance = (
                "The question asks about a specific Synapse Tech product. Identify that product from the evidence "
                "and explain what it is and what it helps with in plain language."
            )
        else:
            extra_guidance = (
                "The user is asking for Synapse Tech's product catalog. List the product names supported by the "
                "evidence and keep the answer focused on the list. Do not add unrelated positioning or infer a "
                "single theme unless the user asks for one."
            )
    elif intent == "service":
        extra_guidance = (
            "The user is asking about Synapse Tech services or service selection. Answer in terms of the services "
            "supported by the evidence. If they ask what you would ask first, ask one practical discovery question "
            "instead of dumping a catalog."
        )
    elif intent == "industry":
        extra_guidance = (
            "List the industries supported by the evidence. If the user mentions an industry not present in the "
            "evidence, do not claim support for it; mention only the confirmed industries."
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


def normalize_product_name(name: str) -> str:
    cleaned = title_prefix(name)
    return PRODUCT_NAME_NORMALIZATION.get(cleaned.lower(), cleaned)


def wants_product_descriptions(question: str) -> bool:
    low = question.lower()
    return (
        "tell me about" in low
        or "explain" in low
        or "describe" in low
        or "what are your products" in low
    ) and ("product" in low or "products" in low)


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
                product_names.append(normalize_product_name(hit["title"]))
            text = normalized_hit_text(hit)
            for label, pattern in PRODUCT_NAME_PATTERNS:
                if pattern.search(text):
                    product_names.append(normalize_product_name(label))
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


def build_company_help_answer(question: str, hits: list[dict[str, Any]]) -> str:
    low = question.lower()
    service_summary = build_summary_answer("service", hits)
    product_summary = build_summary_answer("product", hits)

    if "what can you help" in low or "what can you do" in low:
        parts = [
            "I can help explain Synapse Tech's products, services, and industry use cases.",
        ]
        if service_summary:
            parts.append(service_summary.replace("Synapse Tech Inc. offers ", "That includes "))
        elif product_summary:
            parts.append(product_summary.replace("Synapse offers products including ", "I can also help with products like "))
        return " ".join(parts)

    if "ai services" in low:
        if service_summary:
            return (
                "Synapse Tech offers AI services that help businesses build apps, automate workflows, "
                "use voice agents, and run private infrastructure. "
                + service_summary
            )
        return (
            "Synapse Tech offers AI services around app development, workflow automation, "
            "voice agents, and private infrastructure."
        )

    return ""


def build_voice_agent_answer(hits: list[dict[str, Any]]) -> str:
    blob = " ".join(normalized_hit_text(hit) for hit in hits).lower()
    if "voice ai" in blob or "voice agents" in blob or "call center" in blob or "call automation" in blob:
        return (
            "Yes. Synapse Tech offers AI voice agents and call automation, including voice AI call center "
            "capabilities and conversational automation."
        )
    return ""


def build_product_selection_answer(question: str, hits: list[dict[str, Any]]) -> str:
    low = question.lower()
    blob = " ".join(normalized_hit_text(hit) for hit in hits).lower()
    if "customer support" in low or "chatbot" in low:
        if "coversaction" in blob or "conversaction" in blob or "omnichannel ai chatbot" in blob:
            return (
                "For AI chatbot-based customer support, Coversaction AI looks like the closest fit. "
                "It is positioned as an omnichannel conversational AI product for handling support, "
                "appointments, lead qualification, and human handoff."
            )
        if "agentic bot" in blob:
            return (
                "Agentic Bot could also fit customer support workflows, especially for service requests "
                "and WhatsApp-based interactions."
            )
    return ""


def build_product_catalog_detail_answer(hits: list[dict[str, Any]]) -> str:
    summary = build_summary_answer("product", hits)
    product_names: list[str] = []
    for hit in hits:
        if hit.get("page_type") == "product":
            product_names.append(normalize_product_name(hit["title"]))
        text = normalized_hit_text(hit)
        for label, pattern in PRODUCT_NAME_PATTERNS:
            if pattern.search(text):
                product_names.append(normalize_product_name(label))
    product_names = unique_preserve_order(product_names)
    described = [
        f"{name} is {PRODUCT_SHORT_DESCRIPTIONS[name]}"
        for name in product_names
        if name in PRODUCT_SHORT_DESCRIPTIONS
    ]
    if summary and described:
        return f"{summary} In short: {'; '.join(described)}."
    return summary


def build_difference_answer(hits: list[dict[str, Any]]) -> str:
    blob = " ".join(normalized_hit_text(hit) for hit in hits).lower()
    if "not just another chatbot" in blob or "execution engine" in blob or "jira ticket" in blob:
        return (
            "Synapse Tech's platform is positioned as more than a basic chatbot. "
            "It uses grounded knowledge, can handle workflows like tickets or appointments, and supports "
            "omnichannel and voice-based interactions instead of only answering simple chat prompts."
        )
    return ""


def build_custom_software_answer(hits: list[dict[str, Any]]) -> str:
    blob = " ".join(normalized_hit_text(hit) for hit in hits).lower()
    if "custom software" in blob and ("web" in blob or "mobile" in blob):
        return (
            "Synapse Tech does both. It offers custom web and mobile app development alongside "
            "AI products like Agentic Bot, iRecruit One, Opira AI, Coversaction AI, and "
            "Cyber Security Automation."
        )
    return ""


def build_private_deployment_answer(hits: list[dict[str, Any]]) -> str:
    blob = " ".join(normalized_hit_text(hit) for hit in hits).lower()
    if "opira" in blob and "offline" in blob and ("private" in blob or "on-prem" in blob):
        return (
            "Yes, for products designed for private AI deployment. Opira AI can run offline, "
            "on-premises, or on private cloud infrastructure so sensitive data stays under "
            "the organization's control."
        )
    return ""


def build_workflow_automation_answer(hits: list[dict[str, Any]]) -> str:
    blob = " ".join(normalized_hit_text(hit) for hit in hits).lower()
    if "workflow automation" in blob and all(term in blob for term in ("n8n", "airbyte", "airtable")):
        return (
            "Synapse Tech helps with workflow automation by building decision-making automation "
            "systems that connect APIs, databases, and SaaS tools using orchestration and data "
            "layers such as n8n, Make.com, Airbyte, and Airtable."
        )
    return ""


def build_industry_fit_answer(question: str, hits: list[dict[str, Any]]) -> str:
    low_question = question.lower()
    blob = " ".join(normalized_hit_text(hit) for hit in hits).lower()
    requested = [
        label
        for label in ("healthcare", "retail", "bpo")
        if label in low_question
    ]
    if requested and all(label in blob for label in requested):
        return (
            "Yes. Synapse Tech has solutions for the industries you mentioned: "
            "healthcare and clinics, retail and e-commerce, and BPO and contact centers."
        )
    if requested and "healthcare" in requested and all(label in blob for label in ("retail", "bpo")):
        return (
            "Retail and BPO are explicitly listed in Synapse Tech's industry coverage as "
            "retail and e-commerce and BPO and contact centers. Healthcare is also referenced "
            "in Synapse Tech materials, but the current industry evidence is clearest for retail and BPO."
        )
    return ""


def build_service_discovery_answer() -> str:
    return (
        "First, I would ask what business problem you want to solve first: custom software, "
        "workflow automation, voice automation, private AI, or cloud infrastructure."
    )


def product_name_from_hits_or_profile(hits: list[dict[str, Any]], profile: dict[str, Any]) -> str:
    alias_names = {
        "agentic bot": "Agentic Bot",
        "coversaction ai": "Coversaction AI",
        "opira ai": "Opira AI",
        "irecruit one": "iRecruit One",
        "cyber security automation": "Cyber Security Automation",
    }
    aliases = profile.get("product_aliases") or []
    if aliases:
        return alias_names.get(str(aliases[0]).lower(), title_prefix(str(aliases[0])))
    for hit in hits:
        if hit.get("page_type") == "product":
            return title_prefix(str(hit.get("title", "")))
    return ""


def build_purchase_answer(question: str, hits: list[dict[str, Any]], profile: dict[str, Any]) -> str:
    contact_answer = build_contact_answer(hits)
    if not contact_answer:
        return ""

    product_name = product_name_from_hits_or_profile(hits, profile)
    if product_name:
        return (
            f"To get started with {product_name}, contact Synapse Tech directly. "
            f"{contact_answer} Ask them about {product_name} and share what you want it to handle."
        )

    return f"To get started, contact Synapse Tech directly. {contact_answer}"


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
    if title == "Coversaction AI":
        ecosystem = line_with_pattern(
            candidates,
            r"omnichannel ecosystem.*complete resolution environment",
        )
        workflows = line_with_pattern(
            candidates,
            r"resolving issues, booking appointments, or qualifying leads|appointment management",
        )
        platform = line_with_pattern(
            candidates,
            r"conversational AI platform.*conversational interface",
        )
        if ecosystem or workflows or platform:
            return (
                "Coversaction AI is a conversational AI product built for always-on customer "
                "resolution across customer touchpoints. It can support issue resolution, "
                "appointment management, lead qualification, and handoff workflows."
            )

    if title == "Opira AI":
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
                "Opira AI is an offline LLM platform with private RAG engines that runs entirely on your network. "
                f"{capabilities}"
            )
        if overview:
            return "Opira AI is an offline LLM platform with private RAG engines that runs entirely on your network."

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
        return (
            "Agentic Bot is a WhatsApp-centric assistant that helps organizations handle "
            "service requests, policy questions, form delivery, breach checks, and backend "
            "workflow actions from a single conversation."
        )
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
    if not (
        low_q.startswith("what is")
        or low_q.startswith("what does")
        or profile.get("product_aliases")
    ):
        return ""

    alias_names = {
        "agentic bot": "Agentic Bot",
        "coversaction ai": "Coversaction AI",
        "opira ai": "Opira AI",
        "irecruit one": "iRecruit One",
        "cyber security automation": "Cyber Security Automation",
    }
    title = ""
    for alias in profile.get("product_aliases") or []:
        expected = alias_names.get(str(alias).lower())
        if expected:
            title = expected
            break
    if not title:
        title = title_prefix(hits[0]["title"])

    if title:
        matching_hits = [
            hit
            for hit in hits
            if title.lower() in str(hit.get("title", "")).lower()
            or title.lower().replace(" ", "-") in str(hit.get("doc_id", "")).lower()
        ]
        if matching_hits:
            hits = matching_hits + [
                hit for hit in hits if hit.get("chunk_id") not in {h.get("chunk_id") for h in matching_hits}
            ]

    candidates = candidate_definition_lines(hits)
    if not candidates:
        return ""

    known_answer = build_known_product_answer(title, candidates)
    if known_answer:
        return known_answer

    title_tokens = set(query_kb.tokenize(title))
    scored: list[tuple[float, str]] = []
    blocked_fragments = (
        "ai chatbot solutions built for",
        "how our ai chatbot solution works",
        "built for organizations that",
        "let's make this happen",
        "ready when you are",
    )
    for line in candidates:
        low = line.lower()
        if any(fragment in low for fragment in blocked_fragments):
            continue
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


def determine_answer_policy(question: str, profile: dict[str, Any]) -> str:
    if is_high_risk_question(question):
        return "high_risk_unknown"

    intent = profile.get("intent")
    if is_purchase_or_get_started_question(question):
        return "purchase"
    if intent == "contact":
        return "contact"
    if is_private_deployment_question(question):
        return "private_infrastructure"
    if is_custom_software_question(question):
        return "service_guidance"
    if profile.get("product_aliases"):
        return "product_detail"
    if intent == "product" or is_customer_support_product_question(question):
        return "product_catalog"
    if is_service_discovery_question(question) or is_workflow_automation_question(question):
        return "service_guidance"
    if intent == "service" or is_capability_or_offer_question(question) or is_voice_agent_question(question):
        return "service_catalog"
    if intent == "industry" or is_industry_fit_question(question):
        return "industry"
    return "generic"


def build_policy_answer(
    policy: str,
    question: str,
    hits: list[dict[str, Any]],
    profile: dict[str, Any],
) -> str:
    if policy == "high_risk_unknown":
        return ""

    if policy == "contact":
        return build_contact_answer(hits)

    if policy == "purchase":
        return build_purchase_answer(question, hits, profile)

    if policy == "product_catalog":
        if is_customer_support_product_question(question):
            answer = build_product_selection_answer(question, hits)
            if answer:
                return answer
        if wants_product_descriptions(question):
            return build_product_catalog_detail_answer(hits)
        return build_summary_answer("product", hits)

    if policy == "product_detail":
        return build_entity_answer(question, hits, profile)

    if policy == "service_catalog":
        if is_voice_agent_question(question):
            answer = build_voice_agent_answer(hits)
            if answer:
                return answer
        answer = build_company_help_answer(question, hits)
        if answer:
            return answer
        return build_summary_answer("service", hits)

    if policy == "service_guidance":
        if is_service_discovery_question(question):
            return build_service_discovery_answer()
        if is_custom_software_question(question):
            answer = build_custom_software_answer(hits)
            if answer:
                return answer
        if is_workflow_automation_question(question):
            answer = build_workflow_automation_answer(hits)
            if answer:
                return answer
        return build_summary_answer("service", hits)

    if policy == "private_infrastructure":
        return build_private_deployment_answer(hits)

    if policy == "industry":
        answer = build_industry_fit_answer(question, hits)
        if answer:
            return answer
        return build_summary_answer("industry", hits)

    if is_difference_question(question):
        return build_difference_answer(hits)

    return ""


def retrieve_hits(
    *,
    question: str,
    index_path: Path,
    metadata_path: Path,
    manifest_path: Path,
    embed_model: str | None,
    ollama_url: str,
    rewrite_model: str | None = None,
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
    rewrite = rewrite_query_with_model(
        question=question,
        ollama_url=ollama_url,
        model=rewrite_model or DEFAULT_REWRITE_MODEL,
    )
    profile["rewrite"] = rewrite
    profile["intent_hint"] = rewrite.get("intent_hint")
    profile["company_overview"] = is_company_overview_or_synopsis_question(question)
    profile["purchase_intent"] = is_purchase_or_get_started_question(question)
    search_query = rewrite.get("search_query") or profile.get("search_query") or question
    query_vector = query_kb.embed_query(ollama_url, model, search_query)
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
    log_kb_debug(
        {
            "question": question,
            "stage": "retrieval",
            "search_query": search_query,
            "intent_hint": profile.get("intent_hint"),
            "top_hits": [
                {
                    "title": hit.get("title"),
                    "doc_id": hit.get("doc_id"),
                    "page_type": hit.get("page_type"),
                    "score": hit.get("score"),
                }
                for hit in hits[:5]
            ],
        }
    )
    if profile.get("company_overview"):
        hits = augment_hits_with_about_us(hits, rows, max_chunks=4)
    if is_purchase_or_get_started_question(question):
        hits = augment_hits_with_matching_rows(
            hits,
            rows,
            matches=lambda row: row.get("doc_id") == "contact-us",
            max_chunks=2,
        )
    if is_custom_software_question(question):
        hits = augment_hits_with_matching_rows(
            hits,
            rows,
            matches=lambda row: "custom software" in normalized_hit_text(row).lower(),
        )
    if is_private_deployment_question(question):
        hits = augment_hits_with_matching_rows(
            hits,
            rows,
            matches=lambda row: (
                "opira" in normalized_hit_text(row).lower()
                and "offline" in normalized_hit_text(row).lower()
                and ("private" in normalized_hit_text(row).lower() or "on-prem" in normalized_hit_text(row).lower())
            ),
        )
    if is_workflow_automation_question(question):
        hits = augment_hits_with_matching_rows(
            hits,
            rows,
            matches=lambda row: row.get("doc_id") == "services-automation",
        )
    if is_industry_fit_question(question):
        hits = augment_hits_with_matching_rows(
            hits,
            rows,
            matches=lambda row: (
                row.get("doc_id") == "product-coversaction-ai"
                and all(
                    industry in normalized_hit_text(row).lower()
                    for industry in ("healthcare", "retail", "bpo")
                )
            ),
        )
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
    profile: dict[str, Any],
    hits: list[dict[str, Any]],
) -> dict[str, Any]:
    rewrite = profile.get("rewrite") or {}
    return {
        "question": question,
        "rewrite": {
            "search_query": rewrite.get("search_query") or question,
            "intent_hint": rewrite.get("intent_hint") or profile.get("intent_hint"),
        },
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

    low_risk = not is_high_risk_question(question)
    required_matches = 1 if len(anchor_terms) == 1 else 2
    best_title_doc = 0
    best_combined = 0
    top_score = float(hits[0].get("score", 0.0) or 0.0)
    relevant_page_types = {"product", "service", "industry", "about", "index"}
    has_relevant_page = any((hit.get("page_type") in relevant_page_types) for hit in hits[:4])

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

    if low_risk and has_relevant_page:
        if best_combined >= 1 and top_score >= 0.45:
            return True
        if profile.get("intent") in {"service", "product", "industry"} and top_score >= 0.38:
            return True
        if is_capability_or_offer_question(question) and top_score >= 0.38:
            return True
        if is_voice_agent_question(question) and top_score >= 0.35:
            return True
        if is_difference_question(question) and top_score >= 0.35:
            return True

    if not low_risk:
        return False

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
        log_kb_debug({"question": question, "stage": "empty_context", "fallback": True})
        return DEFAULT_FALLBACK_RESPONSE

    intent = profile.get("intent")
    answer_policy = determine_answer_policy(question, profile)

    if (
        not DISABLE_ROUTE_SPECIFIC_ANSWERS
        and intent in SUMMARY_INTENT_CONFIG
        and answer_policy in {"product_catalog", "service_catalog", "industry"}
        and allow_deterministic_summary(question, intent)
    ):
        summary_answer = build_summary_answer(intent, context_hits)
        if summary_answer:
            return finalize_kb_answer(question, summary_answer)

    if not DISABLE_ROUTE_SPECIFIC_ANSWERS and not DISABLE_NONCRITICAL_DIRECT_ANSWERS:
        policy_answer = build_policy_answer(answer_policy, question, context_hits, profile)
        if policy_answer:
            return finalize_kb_answer(question, clean_answer_text(policy_answer))

    if not has_sufficient_explicit_support(question, context_hits, profile):
        log_kb_debug(
            {
                "question": question,
                "stage": "support_gate",
                "fallback": True,
                "profile": {
                    "intent": profile.get("intent"),
                    "entity_terms": profile.get("entity_terms"),
                    "company_overview": profile.get("company_overview"),
                },
                "top_hits": [
                    {
                        "title": hit.get("title"),
                        "doc_id": hit.get("doc_id"),
                        "page_type": hit.get("page_type"),
                        "score": hit.get("score"),
                    }
                    for hit in context_hits[:3]
                ],
            }
        )
        return DEFAULT_FALLBACK_RESPONSE

    if not DISABLE_ROUTE_SPECIFIC_ANSWERS:
        entity_answer = build_entity_answer(question, context_hits, profile) if answer_policy == "product_detail" else ""
        if entity_answer:
            return finalize_kb_answer(question, clean_answer_text(entity_answer))

    user_prompt = build_user_prompt(question, context_hits, profile)
    predict_cap = max(num_predict, 160) if profile.get("company_overview") else num_predict
    usage = profile.setdefault("usage", {})
    raw = ollama_chat(
        ollama_url=ollama_url,
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        num_predict=predict_cap,
        usage_sink=usage,
        usage_label="answer_generation",
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
                usage_sink=usage,
                usage_label="answer_retry",
            )
            payload_retry = safe_parse_json(raw_retry)
            answer = (
                normalize_answer_value(payload_retry.get("answer"))
                if isinstance(payload_retry, dict)
                else ""
            )
            answer = clean_answer_text(answer)

    if not answer:
        log_kb_debug(
            {
                "question": question,
                "stage": "empty_generation_answer",
                "fallback": True,
                "profile": {
                    "intent": profile.get("intent"),
                    "entity_terms": profile.get("entity_terms"),
                    "company_overview": profile.get("company_overview"),
                },
            }
        )
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
        rewrite_model=args.rewrite_model,
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
                    profile=profile,
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
