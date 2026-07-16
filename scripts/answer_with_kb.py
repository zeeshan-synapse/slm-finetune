#!/usr/bin/env python3
import argparse
import copy
import json
import os
import re
import time
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

import faiss
import requests

import query_kb
from kb_paths import knowledge_base_dir


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_KB_DIR = knowledge_base_dir()
DEFAULT_INDEX_PATH = DEFAULT_KB_DIR / "faiss.index"
DEFAULT_META_PATH = DEFAULT_KB_DIR / "index_meta.jsonl"
DEFAULT_MANIFEST_PATH = DEFAULT_KB_DIR / "index_manifest.json"
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")
DEFAULT_BIEK_NAV_COVERAGE_PATH = PROJECT_DIR / "config" / "biek_nav_coverage.json"
DEFAULT_BIEK_CONTENT_MAP_PATH = PROJECT_DIR / "data" / "runtime" / "biek_nav_content_map.json"
DEFAULT_BIEK_ENRICHED_DOCS_PATH = PROJECT_DIR / "data" / "runtime" / "biek_enriched_documents.jsonl"
DEFAULT_BIEK_RAW_HTML_DIR = PROJECT_DIR / "data" / "raw" / "biek-extracted-html"
DEFAULT_GENERATION_MODEL = os.environ.get("KB_ANSWER_MODEL", "synapse-1.5b-v1")
DEFAULT_REWRITE_MODEL = (
    os.environ.get("KB_REWRITE_MODEL")
    or os.environ.get("KB_BASE_MODEL")
    or "qwen2.5:1.5b-instruct"
)
DEFAULT_CLASSIFIER_MODEL = (
    os.environ.get("KB_CLASSIFIER_MODEL")
    or os.environ.get("KB_BASE_MODEL")
    or os.environ.get("KB_REWRITE_MODEL")
    or "qwen2.5:1.5b-instruct"
)
RAG_COMBINED_PLANNING = os.environ.get(
    "RAG_COMBINED_PLANNING", "1"
).strip().lower() not in {"0", "false", "no", "off"}
DEFAULT_FALLBACK_RESPONSE = (
    "This detail is not confirmed in the available information. Please verify with Synapse Tech."
)
KB_DEBUG_LOG_PATH = PROJECT_DIR / "logs" / "kb_answer_debug.jsonl"
DISABLE_ROUTE_SPECIFIC_ANSWERS = False
DISABLE_NONCRITICAL_DIRECT_ANSWERS = False
RAG_GENERATE_ORDINARY_ANSWERS = os.environ.get(
    "RAG_GENERATE_ORDINARY_ANSWERS", ""
).strip().lower() in {"1", "true", "yes", "on"}
RAG_FAST_MODE = os.environ.get("RAG_FAST_MODE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
RAG_TRIM_EVIDENCE = os.environ.get(
    "RAG_TRIM_EVIDENCE", "1"
).strip().lower() not in {"0", "false", "no", "off"}
RAG_EVIDENCE_SENTENCES_PER_CHUNK = max(
    1, int(os.environ.get("RAG_EVIDENCE_SENTENCES_PER_CHUNK", "3"))
)
RAG_EVIDENCE_CHARS_PER_CHUNK = max(
    200, int(os.environ.get("RAG_EVIDENCE_CHARS_PER_CHUNK", "750"))
)
SAFETY_DETERMINISTIC_POLICIES = {
    "contact",
    "purchase",
    "high_risk_unknown",
}
HARD_DETERMINISTIC_POLICIES = {
    "contact",
    "purchase",
    "high_risk_unknown",
    "product_catalog",
    "ops_guidance",
    "company_overview",
    "voice_agent",
}
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
RUNTIME_CONFIG_DIR = PROJECT_DIR / "data" / "runtime"
DEFAULT_PRODUCT_CARDS = {
    "Agentic Bot": {
        "short_description": "a WhatsApp-centric assistant for service requests, forms, policy queries, breach checks, and workflow actions",
        "aliases": ["agentic bot", "agentic chatbot"],
        "detail_answer": "Agentic Bot is a WhatsApp-centric assistant that helps organizations handle service requests, policy questions, form delivery, breach checks, and backend workflow actions from a single conversation.",
    },
    "iRecruit One": {
        "short_description": "a recruitment intelligence platform for screening, interviews, scheduling, and shortlisting",
        "aliases": ["irecruit", "irecruit one"],
        "detail_answer": "iRecruit One is an enterprise recruitment platform that automates CV screening, AI interviews, and smart shortlisting. It is designed to help teams hire faster, fairer, and at scale.",
    },
    "Opira AI": {
        "short_description": "an offline LLM and private RAG platform for sensitive enterprise knowledge",
        "aliases": ["opira", "opira ai", "opira.io", "opairo"],
        "detail_answer": "Opira AI is an offline LLM platform with private RAG engines that runs entirely on your network. Enterprises can query sensitive documents, compare models, and automate AI workflows without sending data to the cloud.",
    },
    "Coversaction AI": {
        "short_description": "a conversational AI product for customer resolution, appointments, lead qualification, and handoffs",
        "aliases": ["coversaction ai", "conversaction ai", "conversational ai"],
        "detail_answer": "Coversaction AI is a conversational AI product built for always-on customer resolution across customer touchpoints. It can support issue resolution, appointment management, lead qualification, and handoff workflows.",
    },
    "Cyber Security Automation": {
        "short_description": "an automation product for security operations and digital risk workflows",
        "aliases": ["cyber security automation", "cybersecurity automation"],
        "detail_answer": "Cyber Security Automation helps automate security operations and digital risk workflows.",
    },
}
DEFAULT_RUNTIME_POLICIES = {
    "service_title_cleanups": {
        "Custom Web & Mobile App Development": "custom web and mobile app development",
        "Sovereign Cloud Infrastructure": "sovereign cloud infrastructure",
        "AI Workflow Automation & Integration": "AI workflow automation and integration",
        "AI Voice Agents & Conversational Automation": "AI voice agents and conversational automation",
    },
    "custom_software": {
        "answer": "Synapse Tech does both. It offers custom web and mobile app development alongside AI products like Agentic Bot, iRecruit One, Opira AI, Coversaction AI, and Cyber Security Automation.",
    },
    "private_infrastructure": {
        "supported_product": "Opira AI",
        "answer": "Not all Synapse tools are confirmed to run offline. The available information specifically supports Opira AI for offline, on-premises, or private cloud deployment.",
    },
    "workflow_automation": {
        "answer": "Synapse Tech helps with workflow automation by connecting the tools, APIs, databases, and SaaS systems your team already uses into one coordinated workflow. It uses orchestration and data layers such as n8n, Make.com, Airbyte, and Airtable to reduce manual handoffs, copy-paste work, and repetitive operational tasks.",
    },
    "voice_agent": {
        "answer": "Yes. Synapse Tech provides AI voice agents and conversational automation for customer calls, call-center workflows, context-aware conversations, and handoff to human teams when needed.",
    },
    "operational_guidance": {
        "retail_first_deploy": "For a retail support team, start with Coversaction AI or conversational automation for repetitive customer questions, ticket triage, appointments, and human handoff. Keep agents responsible for exceptions and sensitive cases.",
        "banking_first_deploy": "For banking operations, start with one repetitive approval or document-review workflow using AI workflow automation. Keep human approval for regulated, exceptional, or high-impact decisions.",
        "bpo_first_deploy": "For a BPO service desk, start with conversational automation for ticket triage, status updates, repetitive requests, and escalation to human agents.",
        "pilot_success_30": "Measure one workflow against a pre-pilot baseline. Track first-response time, resolution time, error rate, escalation rate, and adoption over 30 days before expanding scope.",
        "low_risk_pilot": "Start with one narrow, high-volume workflow, keep human review for exceptions, and compare results against a baseline. Expand only after the first KPI trend is stable.",
        "automation_human_oversight": "Automate repetitive work and routine routing while keeping people responsible for exceptions, approvals, sensitive cases, and high-impact decisions.",
        "manual_approvals_first_step": "Start by mapping one high-volume approval flow, separating rule-based approvals from exceptions, and keeping human review for sensitive or high-impact decisions. Measure approval cycle time, error rate, and escalation volume before automating more flows.",
        "implementation_effort": "Estimate implementation effort as a range based on workflow scope, integrations, data readiness, security requirements, and human-review needs. Start with a small pilot before committing to a full rollout timeline.",
        "data_before_deployment": "Before deployment, gather the current workflow steps, source documents, decision rules, exception paths, system integrations, access requirements, and baseline performance metrics.",
        "roi_framework": "Think about ROI as time saved, fewer errors, better throughput, and improved service quality against a baseline. Do not assume a fixed percentage unless Synapse confirms it for the specific scope.",
        "highly_manual_process": "Map the highest-volume repetitive steps, select one workflow with clear rules, and automate that first. Measure cycle time, error rate, and manual handoffs before expanding.",
        "limited_product_choice": "Choose based on the confirmed use case, required channels, deployment model, integration needs, and governance requirements. Treat missing details as unconfirmed and verify them before deciding."
    },
    "company_overview": {
        "neutral_sentence": "Synapse Tech develops custom software, workflow automation, conversational AI, private AI, and cloud infrastructure for business operations.",
        "neutral_two_sentences": "Synapse Tech develops custom software, workflow automation, conversational AI, private AI, and cloud infrastructure. Its products and services support areas such as customer service, recruitment, internal knowledge access, and operational automation."
    },
}


def load_runtime_json(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else fallback
    except Exception:
        return fallback


PRODUCT_CARDS = load_runtime_json(RUNTIME_CONFIG_DIR / "product_cards.json", DEFAULT_PRODUCT_CARDS)
RUNTIME_POLICIES = load_runtime_json(RUNTIME_CONFIG_DIR / "policies.json", DEFAULT_RUNTIME_POLICIES)
_KB_RESOURCE_CACHE: dict[tuple[str, str, str], tuple[Any, list[dict[str, Any]], dict[str, Any]]] = {}
_QUERY_PLAN_CACHE: dict[tuple[str, str, bool], tuple[dict[str, str], dict[str, Any]]] = {}
_EMBEDDING_CACHE: dict[tuple[str, str, str], Any] = {}
_RETRIEVAL_CACHE: dict[tuple[Any, ...], tuple[list[dict[str, Any]], str, dict[str, Any]]] = {}
_CONTEXT_SELECTION_CACHE: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
_BIEK_NAV_COVERAGE_CACHE: list[dict[str, Any]] | None = None
_BIEK_NAV_CONTENT_MAP_CACHE: dict[str, Any] | None = None
_BIEK_ENRICHED_DOCS_CACHE: list[dict[str, Any]] | None = None
_BIEK_RAW_HTML_CACHE: list[dict[str, Any]] | None = None

CONTEXT_PACKS = {
    "product_catalog": [
        "product",
        "product-agentic-bot",
        "product-irecruit-one",
        "product-opira-ai",
        "product-coversaction-ai",
        "product-cyber-security-automation",
    ],
    "service_catalog": [
        "services",
        "services-custom-development",
        "services-automation",
        "services-voice-agent",
        "services-cloud-ai",
    ],
    "company_overview": ["about-us", "services", "product"],
    "private_infrastructure": ["product-opira-ai", "services-cloud-ai"],
    "voice_agent": ["services-voice-agent", "product-coversaction-ai"],
    "contact": ["contact-us"],
    "purchase": ["contact-us"],
    "high_risk_unknown": ["contact-us"],
}

ROUTE_QUERY_EXPANSIONS = {
    "recruitment_automation": "iRecruit One recruitment automation cv screening ai interviews",
    "customer_support_conversations": "Coversaction AI customer support conversations chatbot handoff",
    "private_infrastructure": "Opira AI offline on-premises private cloud sensitive documents",
    "voice_agents": "AI voice agents call automation conversational automation",
    "workflow_automation": "AI workflow automation integration APIs databases SaaS tools",
    "custom_development": "custom web mobile app development business workflows",
}


def is_biek_knowledge_domain(knowledge_domain: str | None) -> bool:
    return str(knowledge_domain or "").strip().lower().startswith("biek")


def load_kb_resources(
    index_path: Path,
    metadata_path: Path,
    manifest_path: Path,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    key = (str(index_path.resolve()), str(metadata_path.resolve()), str(manifest_path.resolve()))
    cached = _KB_RESOURCE_CACHE.get(key)
    if cached is not None:
        return cached

    manifest = query_kb.load_manifest(manifest_path)
    index = faiss.read_index(str(index_path))
    rows = query_kb.load_rows(metadata_path)
    if index.ntotal != len(rows):
        raise ValueError("FAISS index size does not match metadata row count.")

    resources = (index, rows, manifest)
    _KB_RESOURCE_CACHE[key] = resources
    return resources


def compile_product_patterns(product_cards: dict[str, Any]) -> list[tuple[str, re.Pattern[str]]]:
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for name, card in product_cards.items():
        aliases = [name]
        if isinstance(card, dict):
            aliases.extend(str(alias) for alias in card.get("aliases", []) if alias)
        alias_pattern = "|".join(
            re.escape(alias).replace(r"\ ", r"\s+")
            for alias in sorted(set(aliases), key=len, reverse=True)
        )
        if alias_pattern:
            patterns.append((name, re.compile(rf"\b(?:{alias_pattern})\b", re.IGNORECASE)))
    return patterns


PRODUCT_NAME_PATTERNS = compile_product_patterns(PRODUCT_CARDS)
USE_CASE_METADATA_ROUTES = [
    {
        "route": "customer_support_conversations",
        "doc_ids": ["product-coversaction-ai"],
        "need_terms": (
            "customer service",
            "customer support",
            "support conversations",
            "support chatbot",
            "chatbot for customer",
            "chatbot",
            "appointments",
            "lead qualification",
            "handoff",
        ),
        "avoid_terms": (),
    },
    {
        "route": "recruitment_automation",
        "doc_ids": ["product-irecruit-one"],
        "need_terms": (
            "recruitment automation",
            "hiring automation",
            "recruiting automation",
            "resume screening",
            "cv screening",
            "candidate matching",
            "ai interviews",
            "shortlisting",
            "hire faster",
            "applicant tracking",
        ),
        "avoid_terms": (
            "customer service",
            "customer support",
            "support chatbot",
            "customer conversations",
        ),
    },
    {
        "route": "private_infrastructure",
        "doc_ids": ["product-opira-ai"],
        "need_terms": (
            "offline",
            "on-prem",
            "on premises",
            "private infrastructure",
            "private cloud",
            "behind firewall",
            "data sovereignty",
            "sensitive documents",
        ),
        "avoid_terms": (),
    },
    {
        "route": "voice_agents",
        "doc_ids": ["services-voice-agent"],
        "need_terms": (
            "voice agent",
            "voice agents",
            "call automation",
            "call center",
            "phone calls",
            "inbound calls",
            "outbound calls",
        ),
        "avoid_terms": (),
    },
    {
        "route": "workflow_automation",
        "doc_ids": ["services-automation"],
        "need_terms": (
            "workflow automation",
            "automate workflows",
            "manual handoffs",
            "copy paste",
            "connect apis",
            "connect databases",
            "saas tools",
        ),
        "avoid_terms": (),
    },
    {
        "route": "custom_development",
        "doc_ids": ["services-custom-development"],
        "need_terms": (
            "custom software",
            "custom app",
            "custom application",
            "web application",
            "mobile app",
            "mobile application",
            "custom development",
        ),
        "avoid_terms": (),
    },
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
SERVICE_TITLE_CLEANUPS = RUNTIME_POLICIES.get("service_title_cleanups", DEFAULT_RUNTIME_POLICIES["service_title_cleanups"])
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
    alias.lower(): name
    for name, card in PRODUCT_CARDS.items()
    for alias in [name, *(card.get("aliases", []) if isinstance(card, dict) else [])]
}
PRODUCT_SHORT_DESCRIPTIONS = {
    name: card.get("short_description", "")
    for name, card in PRODUCT_CARDS.items()
    if isinstance(card, dict) and card.get("short_description")
}
SYSTEM_PROMPT = (
    "You are a factual assistant for Synapse Tech Inc. "
    "Answer only from the provided evidence. "
    "If the evidence does not clearly support the answer, say the detail is not confirmed in the available information. "
    "Do not guess, do not invent details, and do not mention internal retrieval, chunks, embeddings, or vector search. "
    "Return ONLY valid JSON in this exact shape: "
    '{"answer": "<final answer>", "supported": true}'
)
NATURAL_ANSWER_SYSTEM_PROMPT = (
    "You are a factual assistant for Synapse Tech Inc. "
    "Answer the user's original question naturally using only the provided evidence. "
    "Do not invent facts or mention retrieval, chunks, prompts, or internal instructions. "
    "Return only the final user-facing answer."
)
QUERY_REWRITE_SYSTEM_PROMPT = (
    "You rewrite user questions into short search queries for a Synapse Tech knowledge base. "
    "Do not answer the question. Do not invent facts. "
    "Return ONLY valid JSON in this exact shape: "
    '{"search_query": "<short search query>", "intent_hint": "<brief intent>"}'
)
INTENT_CLASSIFIER_SYSTEM_PROMPT = (
    "You classify user questions for a Synapse Tech RAG assistant. "
    "Do not answer the question. Do not invent facts. "
    "Choose exactly one intent from: product_catalog, product_detail, service_catalog, "
    "service_guidance, private_infrastructure, contact, purchase, industry, company_overview, "
    "generic, high_risk_unknown. "
    "Use high_risk_unknown for pricing, legal, compliance, certifications, SLA, roadmap, or guarantees. "
    "Return ONLY valid JSON in this exact shape: "
    '{"intent": "<intent>", "entity": "<product/service/company if any>", "risk": "normal|high_risk", "confidence": 0.0}'
)
QUERY_PLANNER_SYSTEM_PROMPT = (
    "You prepare KB retrieval for a Synapse Tech RAG assistant. "
    "Do not answer the question. Do not invent facts. "
    "Rewrite the user question into a short search query and classify the intent. "
    "Choose exactly one intent from: product_catalog, product_detail, service_catalog, "
    "service_guidance, private_infrastructure, contact, purchase, industry, company_overview, "
    "generic, high_risk_unknown. "
    "Use high_risk_unknown for pricing, legal, compliance, certifications, SLA, roadmap, or guarantees. "
    "Return ONLY valid JSON in this exact shape: "
    '{"search_query": "<short search query>", "intent_hint": "<brief intent>", '
    '"intent": "<intent>", "entity": "<product/service/company if any>", '
    '"risk": "normal|high_risk", "confidence": 0.0}'
)


def rewrite_system_prompt_for_domain(knowledge_domain: str) -> str:
    if is_biek_knowledge_domain(knowledge_domain):
        return (
            "You rewrite user questions into short search queries for a BIEK knowledge base. "
            "Do not answer the question. Do not invent facts. "
            "Return ONLY valid JSON in this exact shape: "
            '{"search_query": "<short search query>", "intent_hint": "<brief intent>"}'
        )
    return QUERY_REWRITE_SYSTEM_PROMPT


def intent_classifier_system_prompt_for_domain(knowledge_domain: str) -> str:
    if is_biek_knowledge_domain(knowledge_domain):
        return (
            "You classify user questions for a BIEK RAG assistant. "
            "Do not answer the question. Do not invent facts. "
            "Choose exactly one intent from: product_catalog, product_detail, service_catalog, "
            "service_guidance, private_infrastructure, contact, purchase, industry, company_overview, "
            "generic, high_risk_unknown. "
            "For BIEK, use generic for ordinary informational questions about forms, notifications, fees, "
            "results, datesheets, or procedures. Use high_risk_unknown only for unsupported legal/compliance/"
            "guarantee-style requests, not for ordinary BIEK questions. "
            "Return ONLY valid JSON in this exact shape: "
            '{"intent": "<intent>", "entity": "<topic if any>", "risk": "normal|high_risk", "confidence": 0.0}'
        )
    return INTENT_CLASSIFIER_SYSTEM_PROMPT


def query_planner_system_prompt_for_domain(knowledge_domain: str) -> str:
    if is_biek_knowledge_domain(knowledge_domain):
        return (
            "You prepare KB retrieval for a BIEK RAG assistant. "
            "Do not answer the question. Do not invent facts. "
            "Rewrite the user question into a short search query and classify the intent. "
            "Choose exactly one intent from: product_catalog, product_detail, service_catalog, "
            "service_guidance, private_infrastructure, contact, purchase, industry, company_overview, "
            "generic, high_risk_unknown. "
            "For BIEK, use generic for normal knowledge-base questions about notifications, forms, "
            "datesheets, fees, results, and procedures. "
            "Return ONLY valid JSON in this exact shape: "
            '{"search_query": "<short search query>", "intent_hint": "<brief intent>", '
            '"intent": "<intent>", "entity": "<topic if any>", '
            '"risk": "normal|high_risk", "confidence": 0.0}'
        )
    return QUERY_PLANNER_SYSTEM_PROMPT
BASELINE_MODEL_PROFILE = {
    "name": "baseline",
    "top_k": 5,
    "context_k": 3,
    "temperature": 0.1,
    "num_predict": 120,
    "plain_temperature": 0.1,
    "plain_num_predict": 180,
    "rewrite_instructions": "",
    "classifier_instructions": "",
    "answer_instructions": "",
    "plain_answer_instructions": "",
}
MODEL_PROFILES = {
    "qwen2.5:3b": {
        "name": "qwen2.5-3b-grounded",
        "top_k": 6,
        "context_k": 3,
        "temperature": 0.05,
        "num_predict": 140,
        "plain_temperature": 0.1,
        "plain_num_predict": 180,
        "rewrite_instructions": (
            "Use a short, literal query. Preserve exact company, product, and service names. "
            "Do not add inferred industries, capabilities, products, or explanatory wording."
        ),
        "classifier_instructions": (
            "Classify the user's goal literally. For recommendations, retain the requested use case and entity. "
            "Use service_guidance for 'which product fits this need' questions. Set entity to an empty string unless "
            "the user explicitly names a product or service. Never guess the recommended product during classification. "
            "Do not convert customer support into cyber security or private infrastructure."
        ),
        "answer_instructions": (
            "Answer concisely and directly. Ignore unrelated products in the context. "
            "For a recommendation, select one closest product unless the user asks for alternatives. "
            "Do not combine capabilities from different products."
        ),
        "plain_answer_instructions": "Answer directly and concisely without adding unrelated alternatives.",
    },
    "qwen2.5:7b": {
        "name": "qwen2.5-7b-grounded",
        "top_k": 6,
        "context_k": 3,
        "temperature": 0.0,
        "num_predict": 150,
        "plain_temperature": 0.05,
        "plain_num_predict": 180,
        "rewrite_instructions": (
            "Produce a compact literal search query containing only terms present in the question or canonical "
            "Synapse product names. Do not expand the question with inferred use cases or conclusions."
        ),
        "classifier_instructions": (
            "Choose the narrowest supported intent and preserve only explicitly named entities. Use service_guidance "
            "for 'which product fits this need' questions and set entity to an empty string when no product is named. "
            "Never choose or guess the recommended product during classification."
        ),
        "answer_instructions": (
            "Every factual claim must be supported by the supplied context. Compare evidence internally, but for "
            "a recommendation output only the best-supported product. Ignore weak or unrelated chunks and never "
            "merge capabilities belonging to different products."
        ),
        "plain_answer_instructions": "Stay focused on the exact question and avoid speculative elaboration.",
    },
    "llama3:latest": {
        "name": "llama3-grounded",
        "top_k": 6,
        "context_k": 3,
        "temperature": 0.1,
        "num_predict": 150,
        "plain_temperature": 0.1,
        "plain_num_predict": 180,
        "rewrite_instructions": (
            "Preserve product terminology exactly and keep the query literal. Do not reinterpret acronyms, "
            "expand product meanings, or add technologies and industries absent from the question."
        ),
        "classifier_instructions": (
            "Classify only the requested goal. Preserve exact product names and distinguish product catalog, "
            "product detail, recommendation, private deployment, and high-risk unknown requests. Use service_guidance "
            "for recommendations, and set entity to an empty string unless the user explicitly names it."
        ),
        "answer_instructions": (
            "Use natural factual language without copying marketing slogans. Preserve terminology exactly: LLM "
            "means large language model, never language learning model. Do not introduce integrations, industries, "
            "technologies, guarantees, or capabilities unless the context explicitly supports them."
        ),
        "plain_answer_instructions": "Use precise terminology and avoid unsupported reinterpretation or marketing language.",
    },
}


def get_model_profile(model: str | None) -> dict[str, Any]:
    """Return a complete profile while preserving baseline behavior for unknown models."""
    selected = MODEL_PROFILES.get((model or "").strip(), {})
    return {**BASELINE_MODEL_PROFILE, **selected}


def profile_system_prompt(base_prompt: str, model: str, instruction_key: str) -> str:
    instructions = str(get_model_profile(model).get(instruction_key) or "").strip()
    if not instructions:
        return base_prompt
    return f"{base_prompt}\n\nModel-specific instructions:\n{instructions}"


def shared_grounded_answer_instructions() -> str:
    return (
        "For product-catalog questions, cover every product supported by the supplied context rather than only "
        "the highest-ranked product. For a single-product recommendation, choose only the closest supported fit: "
        "iRecruit One for recruitment automation, Coversaction AI for customer-support conversations, and Opira AI "
        "for offline or private infrastructure when those facts are supported by the context. For multi-part "
        "high-risk questions, address pricing, SLA, certifications, legal, compliance, roadmap, and guarantees "
        "separately without guessing."
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
_OUTCOME_TERMS = (
    "roi",
    "return on investment",
    "support cost",
    "support costs",
    "cost reduction",
    "productivity",
    "percentage",
    "percent",
    "reduction",
    "outcome",
    "90 days",
    "deadline",
)
_KNOWN_INTEGRATION_NAMES = (
    "salesforce",
    "hubspot",
    "zapier",
    "google drive",
    "workday",
    "sap",
    "crm",
    "api endpoint",
    "rest api",
)
_CUSTOMER_REFERENCE_TERMS = (
    "customer logo",
    "customer logos",
    "customer name",
    "customer names",
    "client logo",
    "client logos",
    "client name",
    "client names",
    "references",
    "case study",
    "case studies",
    "testimonial",
    "testimonials",
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


def is_direct_contact_question(question: str) -> bool:
    low = question.lower()
    return _contains_any(
        low,
        (
            "contact synapse",
            "contact you",
            "contact the team",
            "contact sales",
            "email",
            "phone",
            "call you",
            "reach synapse",
            "reach out",
            "how can i contact",
            "how do i contact",
            "talk to sales",
            "speak to someone",
        ),
    )


def is_high_risk_question(question: str) -> bool:
    return high_risk_category(question) is not None


def is_outcome_guarantee_question(question: str) -> bool:
    low = question.lower()
    return _contains_any(low, ("guarante", "promise")) and _contains_any(low, _OUTCOME_TERMS)


def is_sla_question(question: str) -> bool:
    return _contains_any(question.lower(), ("sla", "uptime", "service level", "response-time", "response time"))


def is_roadmap_question(question: str) -> bool:
    return _contains_any(question.lower(), ("roadmap", "release date", "launch date", "upcoming release", "upcoming releases"))


def is_cert_or_compliance_question(question: str) -> bool:
    low = question.lower()
    return _contains_any(
        low,
        (
            "soc 2",
            "iso",
            "certification",
            "certifications",
            "certificate",
            "compliance standard",
            "compliance standards",
            "regulatory approval",
        ),
    )


def is_legal_terms_question(question: str) -> bool:
    return _contains_any(question.lower(), ("legal", "liability", "contract", "contractual", "warranty", "indemn", "terms"))


def is_general_integration_list_question(question: str) -> bool:
    low = question.lower()
    return "integration" in low and _contains_any(
        low,
        (
            "which integrations",
            "what integrations",
            "integration list",
            "integrations are supported",
            "integrations are explicitly supported",
            "integrations are explicitly confirmed",
        ),
    )


def is_specific_integration_question(question: str) -> bool:
    low = question.lower()
    if _contains_any(low, _KNOWN_INTEGRATION_NAMES):
        return True
    return "integration" in low and not is_general_integration_list_question(question)


def is_public_api_docs_question(question: str) -> bool:
    low = question.lower()
    return "api" in low and _contains_any(low, ("documentation", "docs", "publicly", "public"))


def is_model_architecture_question(question: str) -> bool:
    low = question.lower()
    return "model" in low and _contains_any(
        low,
        ("architecture", "parameters", "parameter count", "training details", "technical details"),
    )


def is_customer_reference_question(question: str) -> bool:
    return _contains_any(question.lower(), _CUSTOMER_REFERENCE_TERMS)


def is_benchmark_specific_question(question: str) -> bool:
    low = question.lower()
    return _contains_any(low, ("benchmark", "benchmarks", "performance number", "performance numbers")) and _contains_any(
        low,
        ("exact", "share", "provide", "number", "numbers", "metric", "metrics", "result", "results"),
    )


def high_risk_category(question: str) -> str | None:
    if is_outcome_guarantee_question(question):
        return "guarantee"
    if is_pricing_or_quote_question(question):
        return "pricing"
    if is_sla_question(question):
        return "sla"
    if is_roadmap_question(question):
        return "roadmap"
    if is_cert_or_compliance_question(question):
        return "certification_compliance"
    if (
        is_general_integration_list_question(question)
        or is_specific_integration_question(question)
        or is_public_api_docs_question(question)
    ):
        return "integration"
    if is_model_architecture_question(question):
        return "technical_detail"
    if is_customer_reference_question(question):
        return "customer_reference"
    if is_benchmark_specific_question(question):
        return "benchmark"
    if is_legal_terms_question(question):
        return "legal"
    return None


def ops_guidance_intent(question: str) -> str | None:
    low = question.lower()
    if is_deploy_priority_question(question):
        if "retail" in low:
            return "retail_first_deploy"
        if "bank" in low or "financial" in low:
            return "banking_first_deploy"
        if "bpo" in low or "service desk" in low or "contact center" in low:
            return "bpo_first_deploy"
    if "which kpi should improve first" in low or "which kpi should improve" in low:
        return "first_kpi"
    if "pilot success in 30 days" in low or "measure pilot success in 30 days" in low:
        return "pilot_success_30"
    if "low-risk pilot" in low or "low risk pilot" in low:
        return "low_risk_pilot"
    if "automation" in low and "human oversight" in low:
        return "automation_human_oversight"
    if "approval" in low and (
        "manual" in low
        or "too many" in low
        or "bottleneck" in low
        or "what should we do first" in low
    ):
        return "manual_approvals_first_step"
    if "implementation effort" in low:
        return "implementation_effort"
    if "data" in low and "before deployment" in low:
        return "data_before_deployment"
    if "think about roi for automation" in low or "roi for automation" in low:
        return "roi_framework"
    if "process" in low and "highly manual" in low:
        return "highly_manual_process"
    if "choose between products" in low and "limited details" in low:
        return "limited_product_choice"
    return None


def build_ops_guidance_answer(intent: str | None) -> str:
    configured = RUNTIME_POLICIES.get("operational_guidance", {})
    if intent and isinstance(configured, dict):
        configured_answer = configured.get(intent)
        if configured_answer:
            return str(configured_answer)

    if intent == "first_kpi":
        return (
            "Track first-response time first, because it usually shows whether routing and manual handling are improving. "
            "Then compare resolution time, error rate, and escalation volume against the baseline."
        )
    if intent == "pilot_success_30":
        return (
            "Measure one workflow against a pre-pilot baseline. Track first-response time, resolution time, error rate, "
            "and adoption over 30 days before expanding scope."
        )
    if intent == "low_risk_pilot":
        return (
            "Start with one narrow, high-volume workflow, keep human review for exceptions, and compare results against a baseline. "
            "Expand only after the first KPI trend is stable."
        )
    if intent == "roi_framework":
        return (
            "Think about ROI as time saved, fewer errors, better throughput, and improved service quality against a baseline. "
            "Do not assume a fixed percentage unless Synapse confirms it for the specific scope."
        )
    return ""


def is_product_selection_question(question: str) -> bool:
    low = question.lower()
    has_pick = any(
        w in low for w in ("choose", "pick", "select", "compare", "deciding", "decide")
    )
    has_between = "between" in low or "among" in low
    has_products = "product" in low or "products" in low
    return has_products and (has_pick or has_between)


def is_single_product_recommendation_question(question: str) -> bool:
    low = question.lower()
    if re.search(r"\b(?:what|which)\s+products\b", low) or _contains_any(
        low,
        (
            "list products",
            "list all products",
            "products do you offer",
            "products does synapse",
            "tell me about your products",
        ),
    ):
        return False
    asks_for_product = _contains_any(
        low,
        (
            "which product",
            "which synapse product",
            "what product",
            "what synapse product",
            "product for",
            "product that",
            "do you have a product",
            "recommend a product",
            "best product",
        ),
    ) or bool(re.search(r"\b(?:which|what)\s+\w+\s+product\b", low))
    expresses_need = _contains_any(
        low,
        (
            "i need",
            "we need",
            "i want",
            "we want",
            "looking for",
            "help with",
        ),
    )
    return asks_for_product or (expresses_need and "product" in low)


def metadata_routes_for_question(question: str) -> list[dict[str, Any]]:
    """Return retrieval routes based on the user's requested job-to-be-done.

    Business context words are intentionally weaker than need words. For example,
    "recruitment agency needing customer service" should route to Coversaction AI,
    not iRecruit One.
    """
    low = question.lower()
    routes: list[dict[str, Any]] = []
    for route in USE_CASE_METADATA_ROUTES:
        need_terms = route.get("need_terms", ())
        avoid_terms = route.get("avoid_terms", ())
        if any(term in low for term in avoid_terms):
            continue
        matched_terms = [term for term in need_terms if term in low]
        if not matched_terms:
            continue
        routes.append(
            {
                "route": route["route"],
                "doc_ids": list(route["doc_ids"]),
                "matched_terms": matched_terms,
            }
        )
    return routes


def product_names_in_question(question: str) -> list[str]:
    names: list[str] = []
    for name, pattern in PRODUCT_NAME_PATTERNS:
        if pattern.search(question):
            names.append(name)
    return unique_preserve_order(names)


def is_product_comparison_question(question: str) -> bool:
    low = question.lower()
    comparison_signal = _contains_any(
        low,
        ("compare", "comparison", " vs ", " versus ", "difference between", "choose between"),
    )
    return comparison_signal and len(product_names_in_question(question)) >= 2


def requested_sentence_count(question: str) -> int | None:
    low = question.lower()
    match = re.search(r"\bexactly\s+(\d+)\s+sentences?\b", low)
    if match:
        return int(match.group(1))
    word_counts = {
        "one sentence": 1,
        "two sentences": 2,
        "three sentences": 3,
        "four sentences": 4,
    }
    for phrase, count in word_counts.items():
        if f"exactly {phrase}" in low:
            return count
    return None


def answer_sentence_count(answer: str) -> int:
    return len(
        [
            part
            for part in re.split(r"(?<=[.!?])(?:\s+|$)", answer.strip())
            if part.strip()
        ]
    )


def requests_non_marketing_tone(question: str) -> bool:
    low = question.lower()
    return _contains_any(
        low,
        (
            "no marketing",
            "without marketing",
            "non-marketing",
            "neutral wording",
            "neutral",
            "plain factual",
            "factual summary",
            "short factual",
        ),
    )


def contains_marketing_language(answer: str) -> bool:
    low = answer.lower()
    return _contains_any(
        low,
        (
            "intelligent advantage",
            "strategic advantage",
            "empower businesses",
            "cutting-edge",
            "revolutionary",
            "game-changing",
            "best-in-class",
            "industry-leading",
            "seamless",
            "transformative",
            "move faster",
            "operate smarter",
            "scale confidently",
            "provide intelligence rather than tools",
            "eliminate complexity",
            "unlock",
            "future-ready",
            "autonomously",
            "on its own",
            "makes decisions",
            "make decisions",
            "eliminating",
            "eliminate",
            "significant advantage",
            "advantage over",
            "mission is to build intelligence",
            "intelligence-powered",
            "decision-making processes",
            "freeing up resources",
            "strategic initiatives",
            "complete resolution environment",
            "microservices",
            "oauth",
            "scalable apis",
            "single byte",
            "proprietary data",
            "vendor lock-ins",
            "lock-ins",
        ),
    )


def instruction_following_issues(question: str, answer: str) -> list[str]:
    issues: list[str] = []
    normalized = answer.strip().lower()
    if normalized in {"<text>", "<answer>", "answer", "response"} or re.search(
        r"<(?:text|answer|response)>",
        normalized,
    ):
        issues.append("placeholder_output")
    expected_sentences = requested_sentence_count(question)
    if expected_sentences is not None and answer_sentence_count(answer) != expected_sentences:
        issues.append("wrong_sentence_count")
    if requests_non_marketing_tone(question) and contains_marketing_language(answer):
        issues.append("marketing_language")
    return issues


def should_force_instruction_fallback(question: str, answer: str, profile: dict[str, Any]) -> bool:
    if profile.get("product_aliases") and requested_sentence_count(question) is not None:
        return bool(contains_marketing_language(answer) or product_detail_answer_needs_cleanup(question, answer, profile))
    if requests_non_marketing_tone(question) and contains_marketing_language(answer):
        return True
    return False


def comparison_answer_issues(question: str, answer: str) -> list[str]:
    if not is_product_comparison_question(question):
        return []
    low = answer.lower()
    missing = [name for name in product_names_in_question(question) if name.lower() not in low]
    return ["missing_comparison_side"] if missing else []


def build_product_comparison_fallback(question: str) -> str:
    names = product_names_in_question(question)
    if len(names) < 2 or any(not PRODUCT_SHORT_DESCRIPTIONS.get(name) for name in names[:2]):
        return ""
    return (
        f"{names[0]} focuses on {PRODUCT_SHORT_DESCRIPTIONS[names[0]]}. "
        f"{names[1]} focuses on {PRODUCT_SHORT_DESCRIPTIONS[names[1]]}."
    )


def build_instruction_fallback(question: str) -> str:
    product_names = product_names_in_question(question)
    if product_names:
        product_name = product_names[0]
        card = PRODUCT_CARDS.get(product_name, {})
        if isinstance(card, dict):
            detail_answer = str(card.get("detail_answer") or "").strip()
            short_description = str(card.get("short_description") or "").strip()
            expected_sentences = requested_sentence_count(question)
            if detail_answer:
                sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(detail_answer) if s.strip()]
                if expected_sentences == 1 and sentences:
                    return sentences[0]
                if expected_sentences == 2:
                    if len(sentences) >= 2:
                        return " ".join(sentences[:2])
                    if short_description:
                        return f"{sentences[0] if sentences else detail_answer} It is {short_description}."
                return detail_answer

    overview = RUNTIME_POLICIES.get("company_overview", {})
    if not isinstance(overview, dict):
        return ""
    expected_sentences = requested_sentence_count(question)
    if expected_sentences == 2:
        return str(overview.get("neutral_two_sentences") or "")
    if requests_non_marketing_tone(question):
        return str(overview.get("neutral_sentence") or "")
    return str(overview.get("neutral_sentence") or overview.get("neutral_two_sentences") or "")


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


def augment_hits_for_product_comparison(
    hits: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    product_names: list[str],
    *,
    chunks_per_product: int = 2,
) -> list[dict[str, Any]]:
    hit_ids = {h["chunk_id"] for h in hits}
    score_seed = max((float(h.get("score", 0)) for h in hits), default=0.55)
    score_seed = max(score_seed, 0.96)
    front: list[dict[str, Any]] = []

    matching_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in product_names}
    for row in rows:
        if row.get("page_type") != "product":
            continue
        row_name = normalize_product_name(title_prefix(str(row.get("title", ""))))
        cid = row.get("chunk_id")
        if row_name in matching_rows and cid:
            matching_rows[row_name].append(row)

    for chunk_index in range(chunks_per_product):
        for product_name in product_names:
            product_rows = matching_rows.get(product_name, [])
            if chunk_index >= len(product_rows):
                continue
            row = product_rows[chunk_index]
            cid = row.get("chunk_id")
            if not cid or any(hit.get("chunk_id") == cid for hit in front):
                continue
            front.append(make_hit_from_row(row, score_seed))
            score_seed -= 0.001

    return front + [
        hit
        for hit in hits
        if hit.get("chunk_id") not in {front_hit["chunk_id"] for front_hit in front}
    ]


def answer_shape_mismatch(question: str, answer: str) -> bool:
    """True if the answer looks like a wrong template for the question (e.g. product menu for pricing)."""
    low_q = question.lower()
    low_a = answer.lower()
    if not answer.strip():
        return False

    contact_only = (
        ("info@synapsetechinc.com" in low_a or "contact synapse" in low_a)
        and len(answer.split()) <= 35
    )
    if contact_only and not (
        is_direct_contact_question(question)
        or is_purchase_or_get_started_question(question)
        or is_pricing_or_quote_question(question)
    ):
        return True

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


def workflow_answer_too_shallow(question: str, answer: str) -> bool:
    """Catch workflow answers that only name tools without explaining what Synapse does."""
    if not is_workflow_automation_question(question) or not answer.strip():
        return False

    low = answer.lower()
    tool_terms = ("n8n", "make.com", "airbyte", "airtable")
    action_terms = (
        "connect",
        "api",
        "database",
        "saas",
        "orchestration",
        "pipeline",
        "workflow",
        "manual",
        "handoff",
        "copy-paste",
        "repetitive",
        "decision",
        "automate",
        "end-to-end",
        "data movement",
        "visibility",
        "logging",
    )
    tool_count = sum(1 for term in tool_terms if term in low)
    action_count = sum(1 for term in action_terms if term in low)

    if action_count < 3:
        return True
    if "automation" not in low and "automate" not in low:
        return True
    if tool_count >= 2 and action_count < 3:
        return True
    if "each of them solves a different part of the operational puzzle" in low:
        return True
    if re.search(r"\b(?:our|synapse(?: tech(?: inc\.)?)?'?s?)\s+products?\s+(?:like|include|includes)\s+(n8n|make\.com|airbyte|airtable)", low):
        return True
    if re.search(r"\b(?:our|synapse(?: tech(?: inc\.)?)?'?s?)\s+products?\s+(?:like|include|includes)\s+", low):
        return True
    if re.search(r"\bsynapse tech(?: inc\.)?'?s?\s+products?\s+help with workflow automation", low):
        return True
    if re.search(r"\bour\s+products?\s+connect\s+(apis|tools|databases|saas)", low):
        return True
    if low.startswith("the product helps with workflow automation"):
        return True
    if "tools include" in low and tool_count >= 2:
        return True
    if "make.com for engineering custom web platforms" in low:
        return True
    if len(low.split()) < 24 and tool_count >= 1:
        return True
    return False


def product_overview_answer_too_shallow(question: str, answer: str) -> bool:
    """Product overview answers should name the actual products, not only describe themes."""
    if not wants_product_descriptions(question) or not answer.strip():
        return False

    low = answer.lower()
    product_hits = sum(
        1
        for name in PRODUCT_SHORT_DESCRIPTIONS
        if name.lower() in low
    )
    return product_hits < 3


def product_detail_answer_needs_cleanup(question: str, answer: str, profile: dict[str, Any]) -> bool:
    if determine_answer_policy(question, profile) != "product_detail" or not answer.strip():
        return False
    low_q = question.lower()
    low = answer.lower()
    if "used for" in low_q and any(
        fragment in low
        for fragment in (
            "omnichannel ai chatbot",
            "complex decision trees",
            "multi-tenant database isolation",
            "oauth 2.0",
            "manages apis",
        )
    ):
        return True
    return any(
        fragment in low
        for fragment in (
            "automates every form",
            "policy query & breach check",
            "policy query and breach check",
            "ai chatbot solutions built for",
            "how our ai chatbot solution works",
            "eliminate vulnerabilities",
            "guarantee",
            "guaranteed",
            "without risk",
        )
    )


def finalize_kb_answer(question: str, answer: str) -> str:
    low_a = answer.lower()
    if "available biek information" in low_a or "official biek website" in low_a:
        return answer
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
        "--classifier-model",
        default=DEFAULT_CLASSIFIER_MODEL,
        help="Ollama model to use for classifying user intent before answer policy selection.",
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
    json_format: bool = True,
) -> str:
    payload = {
        "model": model,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    if json_format:
        payload["format"] = "json"
    started = time.perf_counter()
    response = requests.post(
        f"{ollama_url}/api/chat",
        json=payload,
        timeout=180,
    )
    elapsed_s = time.perf_counter() - started
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
            "elapsed_s": round(elapsed_s, 4),
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
    knowledge_domain: str = "synapse",
) -> dict[str, str]:
    fallback = {"search_query": question.strip(), "intent_hint": "fallback_original"}
    if not question.strip():
        return fallback

    model_profile = get_model_profile(model)
    rewrite_guidance = str(model_profile.get("rewrite_instructions") or "").strip()
    if is_biek_knowledge_domain(knowledge_domain):
        user_prompt = (
            "Rewrite this user question into a short search query for a BIEK knowledge base.\n"
            "Keep exact BIEK terms like enrolment, date sheet, migration form, certificate form, "
            "verification form, HSC Part-I, fees, results, and notifications. Expand casual wording like u/ur. "
            "Do not answer the question.\n"
            f"{rewrite_guidance + chr(10) if rewrite_guidance else ''}\n"
            f"User question: {question}"
        )
    else:
        user_prompt = (
            "Rewrite this user question into a short search query for a Synapse Tech knowledge base.\n"
            "Keep product names, service names, and company names. Expand casual wording like u/ur. "
            "If the user asks about Synapse offerings, include 'Synapse Tech'. "
            "Do not answer the question.\n"
            f"{rewrite_guidance + chr(10) if rewrite_guidance else ''}\n"
            f"User question: {question}"
        )
    usage: dict[str, Any] = {}
    try:
        raw = ollama_chat(
            ollama_url=ollama_url,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": profile_system_prompt(
                        rewrite_system_prompt_for_domain(knowledge_domain),
                        model,
                        "rewrite_instructions",
                    ),
                },
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


CLASSIFIER_INTENTS = {
    "product_catalog",
    "product_detail",
    "service_catalog",
    "service_guidance",
    "private_infrastructure",
    "contact",
    "purchase",
    "industry",
    "company_overview",
    "generic",
    "high_risk_unknown",
}
CLASSIFIER_INTENT_ALIASES = {
    "products": "product_catalog",
    "product": "product_catalog",
    "product_list": "product_catalog",
    "product_listing": "product_catalog",
    "product_overview": "product_catalog",
    "product_summary": "product_catalog",
    "product_information": "product_catalog",
    "product_info": "product_catalog",
    "product_details": "product_detail",
    "service": "service_catalog",
    "services": "service_catalog",
    "service_list": "service_catalog",
    "service_listing": "service_catalog",
    "guidance": "service_guidance",
    "service_selection": "service_guidance",
    "recommendation": "service_guidance",
    "offline": "private_infrastructure",
    "private": "private_infrastructure",
    "pricing": "high_risk_unknown",
    "price": "high_risk_unknown",
    "legal": "high_risk_unknown",
    "compliance": "high_risk_unknown",
    "certification": "high_risk_unknown",
    "certifications": "high_risk_unknown",
    "sla": "high_risk_unknown",
    "about": "company_overview",
    "company": "company_overview",
    "overview": "company_overview",
}


def normalize_classifier_intent(value: Any) -> str:
    raw = normalize_answer_value(value).lower().strip()
    raw = raw.replace(" ", "_").replace("-", "_")
    raw = CLASSIFIER_INTENT_ALIASES.get(raw, raw)
    return raw if raw in CLASSIFIER_INTENTS else "generic"


def normalize_classifier_entity(value: Any) -> str:
    entity = normalize_answer_value(value)
    if not entity:
        return ""
    low = entity.lower().strip()
    return PRODUCT_NAME_NORMALIZATION.get(low, entity[:120])


def classify_question_with_model(
    *,
    question: str,
    search_query: str,
    ollama_url: str,
    model: str,
    knowledge_domain: str = "synapse",
) -> dict[str, Any]:
    fallback = {
        "intent": "generic",
        "entity": "",
        "risk": "normal",
        "confidence": 0.0,
        "source": "fallback",
    }
    if not question.strip():
        return fallback

    model_profile = get_model_profile(model)
    classifier_guidance = str(model_profile.get("classifier_instructions") or "").strip()
    if is_biek_knowledge_domain(knowledge_domain):
        user_prompt = (
            "Classify this user question for a BIEK assistant.\n"
            "Important examples:\n"
            "- Generic: enrolment, forms, migration form, certificate form, datesheet, notifications, fees, results.\n"
            "- High risk: unsupported legal/compliance/guarantee-style claims.\n"
            "- Meta help: what can you help me with, what can you do.\n"
            "Most ordinary BIEK website/help questions should be generic.\n"
            f"{classifier_guidance + chr(10) if classifier_guidance else ''}\n"
            f"Original question: {question}\n"
            f"Search query: {search_query}"
        )
    else:
        user_prompt = (
            "Classify this user question for a Synapse Tech assistant.\n"
            "Important examples:\n"
            "- Product catalog: list products, what products do you offer, tell me about your products.\n"
            "- Product detail: tell me about Opira AI, what does Agentic Bot do.\n"
            "- Service guidance: help choosing a service, custom software vs AI products, workflow automation.\n"
            "- Private infrastructure: offline, on-premises, private cloud, private deployment.\n"
            "- Purchase/contact: buy, purchase, get started, contact sales.\n"
            "- High risk: pricing, legal, compliance, certifications, SLA, roadmap, guarantees.\n"
            f"{classifier_guidance + chr(10) if classifier_guidance else ''}\n"
            f"Original question: {question}\n"
            f"Search query: {search_query}"
        )
    usage: dict[str, Any] = {}
    try:
        raw = ollama_chat(
            ollama_url=ollama_url,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": profile_system_prompt(
                        intent_classifier_system_prompt_for_domain(knowledge_domain),
                        model,
                        "classifier_instructions",
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            num_predict=90,
            usage_sink=usage,
            usage_label="intent_classification",
        )
        payload = safe_parse_json(raw)
        raw_intent = normalize_answer_value(payload.get("intent"))
        intent = normalize_classifier_intent(raw_intent)
        if intent == "generic" and raw_intent and raw_intent.lower().replace(" ", "_").replace("-", "_") != "generic":
            log_kb_debug(
                {
                    "question": question,
                    "stage": "intent_classification_unrecognized_label",
                    "raw_intent": raw_intent,
                    "raw": raw[:500],
                }
            )
        entity = normalize_classifier_entity(payload.get("entity"))
        risk = normalize_answer_value(payload.get("risk")).lower().strip()
        if risk not in {"normal", "high_risk"}:
            risk = "high_risk" if intent == "high_risk_unknown" else "normal"
        confidence_raw = payload.get("confidence", 0.0)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(confidence, 1.0))
        return {
            "intent": intent,
            "entity": entity,
            "risk": risk,
            "confidence": confidence,
            "usage": usage.get("intent_classification"),
            "source": "model",
        }
    except Exception as exc:
        log_kb_debug(
            {
                "question": question,
                "stage": "intent_classification_failed",
                "error": str(exc),
            }
        )
        return fallback


def plan_query_with_model(
    *,
    question: str,
    ollama_url: str,
    model: str,
    knowledge_domain: str = "synapse",
) -> tuple[dict[str, str], dict[str, Any]]:
    rewrite_fallback = {"search_query": question.strip(), "intent_hint": "fallback_original"}
    classification_fallback = {
        "intent": "generic",
        "entity": "",
        "risk": "normal",
        "confidence": 0.0,
        "source": "fallback",
    }
    if not question.strip():
        return rewrite_fallback, classification_fallback

    model_profile = get_model_profile(model)
    rewrite_guidance = str(model_profile.get("rewrite_instructions") or "").strip()
    classifier_guidance = str(model_profile.get("classifier_instructions") or "").strip()
    extra_guidance = "\n".join(
        part for part in (rewrite_guidance, classifier_guidance) if part
    )
    if is_biek_knowledge_domain(knowledge_domain):
        user_prompt = (
            "Prepare this user question for BIEK KB retrieval.\n"
            "Important examples:\n"
            "- Generic: enrolment, exam forms, migration form, certificate form, datesheet, notifications, fees, results.\n"
            "- Meta help: what can you help me with, what can you do.\n"
            "- High risk: unsupported legal/compliance/guarantee-style claims.\n"
            "Keep the search query short and literal. Preserve exact BIEK terms and year mentions. "
            "Do not answer the question.\n"
            f"{extra_guidance + chr(10) if extra_guidance else ''}\n"
            f"Original question: {question}"
        )
    else:
        user_prompt = (
            "Prepare this user question for Synapse Tech KB retrieval.\n"
            "Important examples:\n"
            "- Product catalog: list products, what products do you offer, tell me about your products.\n"
            "- Product detail: tell me about Opira AI, what does Agentic Bot do.\n"
            "- Service guidance: help choosing a service, custom software vs AI products, workflow automation.\n"
            "- Private infrastructure: offline, on-premises, private cloud, private deployment.\n"
            "- Purchase/contact: buy, purchase, get started, contact sales.\n"
            "- High risk: pricing, legal, compliance, certifications, SLA, roadmap, guarantees.\n"
            "Keep the search query short and literal. Preserve exact company, product, and service names. "
            "Do not answer the question.\n"
            f"{extra_guidance + chr(10) if extra_guidance else ''}\n"
            f"Original question: {question}"
        )
    usage: dict[str, Any] = {}
    try:
        raw = ollama_chat(
            ollama_url=ollama_url,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": profile_system_prompt(
                        query_planner_system_prompt_for_domain(knowledge_domain),
                        model,
                        "classifier_instructions",
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            num_predict=120,
            usage_sink=usage,
            usage_label="query_planning",
        )
        payload = safe_parse_json(raw)
        search_query = normalize_answer_value(payload.get("search_query"))
        if not search_query:
            log_kb_debug(
                {
                    "question": question,
                    "stage": "query_planning_unparseable",
                    "model": model,
                    "raw": raw[:500],
                }
            )
            return rewrite_fallback, classification_fallback

        intent_hint = normalize_answer_value(payload.get("intent_hint"))
        raw_intent = normalize_answer_value(payload.get("intent"))
        intent = normalize_classifier_intent(raw_intent)
        entity = normalize_classifier_entity(payload.get("entity"))
        risk = normalize_answer_value(payload.get("risk")).lower().strip()
        if risk not in {"normal", "high_risk"}:
            risk = "high_risk" if intent == "high_risk_unknown" else "normal"
        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(confidence, 1.0))

        usage_item = usage.get("query_planning")
        return (
            {
                "search_query": search_query[:240],
                "intent_hint": intent_hint[:80] if intent_hint else "model_planning",
                "usage": usage_item,
            },
            {
                "intent": intent,
                "entity": entity,
                "risk": risk,
                "confidence": confidence,
                "usage": usage_item,
                "source": "combined_model",
            },
        )
    except Exception as exc:
        log_kb_debug(
            {
                "question": question,
                "stage": "query_planning_failed",
                "error": str(exc),
            }
        )
        return rewrite_fallback, classification_fallback


def classifier_intent_to_query_intent(intent: str) -> str | None:
    if intent in {"product_catalog", "product_detail"}:
        return "product"
    if intent in {"service_catalog", "service_guidance", "private_infrastructure"}:
        return "service"
    if intent == "industry":
        return "industry"
    if intent == "contact" or intent == "purchase":
        return "contact"
    if intent == "company_overview":
        return "about"
    if intent == "high_risk_unknown":
        return "pricing"
    return None


def apply_model_classification_to_profile(profile: dict[str, Any], classification: dict[str, Any]) -> None:
    profile["model_classification"] = classification
    profile["model_intent"] = classification.get("intent") or "generic"
    profile["model_entity"] = classification.get("entity") or ""
    profile["model_risk"] = classification.get("risk") or "normal"
    profile["model_confidence"] = classification.get("confidence", 0.0)

    query_intent = classifier_intent_to_query_intent(profile["model_intent"])
    if query_intent and not profile.get("intent"):
        profile["intent"] = query_intent

    entity = profile.get("model_entity") or ""
    if entity:
        normalized_entity = PRODUCT_NAME_NORMALIZATION.get(entity.lower(), entity)
        for token in query_kb.tokenize(normalized_entity):
            if token not in profile.get("entity_terms", []) and token not in query_kb.GENERIC_QUERY_TERMS:
                profile.setdefault("entity_terms", []).append(token)
        for label, _pattern in PRODUCT_NAME_PATTERNS:
            if normalized_entity.lower() == label.lower():
                alias = label.lower()
                if alias not in profile.get("product_aliases", []):
                    profile.setdefault("product_aliases", []).append(alias)


def split_evidence_units(text: str) -> list[str]:
    units: list[str] = []
    for raw_line in re.split(r"\n+", text):
        line = " ".join(raw_line.split()).strip()
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“])", line)
        units.extend(part.strip() for part in parts if part.strip())
    return units


def evidence_policy_anchors(question: str, profile: dict[str, Any]) -> set[str]:
    anchors: set[str] = set()
    policy = profile.get("answer_policy") or determine_answer_policy(question, profile)
    if policy == "company_overview":
        anchors.update({"software", "automation", "conversational", "cloud", "infrastructure", "business"})
    elif policy == "private_infrastructure":
        anchors.update({
            "offline", "private", "on-premise", "on-premises", "cloud", "opira",
            "data", "network", "internet", "security", "sovereignty",
        })
    elif policy == "service_guidance":
        anchors.update({"workflow", "automation", "custom", "web", "mobile", "api", "database", "saas"})
    elif policy in {"product_catalog", "product_detail"}:
        anchors.update({"agentic", "irecruit", "opira", "coversaction", "conversaction", "cyber"})
    elif policy == "voice_agent":
        anchors.update({"voice", "call", "conversation", "handoff", "customer"})
    elif policy in {"contact", "purchase"}:
        anchors.update({"contact", "email", "chat", "monday", "saturday", "info"})
    elif policy == "high_risk_unknown":
        anchors.update({"pricing", "price", "sla", "certification", "compliance", "guarantee"})
    return anchors


def trim_evidence_text(
    question: str,
    hit: dict[str, Any],
    profile: dict[str, Any],
) -> str:
    text = str(hit.get("text") or "").strip()
    if not RAG_TRIM_EVIDENCE or len(text) <= RAG_EVIDENCE_CHARS_PER_CHUNK:
        return text

    units = split_evidence_units(text)
    if not units:
        return text[:RAG_EVIDENCE_CHARS_PER_CHUNK].rstrip() + "..."

    keywords = {
        token
        for token in profile.get("keywords", [])
        if token not in query_kb.GENERIC_QUERY_TERMS
    }
    entity_terms = set(profile.get("entity_terms", []))
    title_tokens = set(query_kb.tokenize(str(hit.get("title") or "")))
    anchors = evidence_policy_anchors(question, profile)
    aliases = [str(alias).lower() for alias in profile.get("product_aliases", [])]

    scored: list[tuple[float, int, str]] = []
    for index, unit in enumerate(units):
        lowered = unit.lower()
        tokens = set(query_kb.tokenize(lowered))
        score = 0.0
        score += len(tokens & entity_terms) * 4.0
        score += len(tokens & keywords) * 1.5
        score += len(tokens & title_tokens) * 1.0
        score += len(tokens & anchors) * 1.5
        score += sum(2.0 for alias in aliases if alias and alias in lowered)
        if index == 0:
            score += 0.25
        if len(tokens) < 8 and profile.get("answer_policy") not in {"contact", "purchase"}:
            score -= 3.0
        scored.append((score, index, unit))

    selected = sorted(
        sorted(scored, key=lambda item: (item[0], -item[1]), reverse=True)[
            :RAG_EVIDENCE_SENTENCES_PER_CHUNK
        ],
        key=lambda item: item[1],
    )
    excerpts: list[str] = []
    used_chars = 0
    for _score, _index, unit in selected:
        separator = 1 if excerpts else 0
        remaining = RAG_EVIDENCE_CHARS_PER_CHUNK - used_chars - separator
        if remaining <= 0:
            break
        if len(unit) > remaining:
            unit = unit[: max(1, remaining - 3)].rstrip() + "..."
        excerpts.append(unit)
        used_chars += len(unit) + separator
    return "\n".join(excerpts) if excerpts else text[:RAG_EVIDENCE_CHARS_PER_CHUNK]


def format_evidence_block(
    hit: dict[str, Any],
    rank: int,
    content: str,
) -> str:
    return (
        f"[{rank}] Title: {hit['title']}\n"
        f"Page Type: {hit['page_type']}\n"
        f"Content:\n{content}"
    )


def build_evidence_prompt(
    question: str,
    hits: list[dict[str, Any]],
    profile: dict[str, Any],
) -> str:
    original_chars = sum(len(str(hit.get("text") or "")) for hit in hits)
    trimmed_contents = [trim_evidence_text(question, hit, profile) for hit in hits]
    prompt_chars = sum(len(content) for content in trimmed_contents)
    profile["context_compression"] = {
        "enabled": RAG_TRIM_EVIDENCE,
        "chunks": len(hits),
        "original_chars": original_chars,
        "prompt_chars": prompt_chars,
        "reduction_pct": round(
            (1.0 - prompt_chars / original_chars) * 100.0,
            1,
        ) if original_chars else 0.0,
    }
    return "\n\n".join(
        format_evidence_block(hit, rank, content)
        for rank, (hit, content) in enumerate(zip(hits, trimmed_contents, strict=False), start=1)
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


def normalized_text_fingerprint(text: str) -> str:
    words = query_kb.tokenize(text)
    return " ".join(words[:80])


def token_jaccard(a: str, b: str) -> float:
    a_tokens = set(query_kb.tokenize(a))
    b_tokens = set(query_kb.tokenize(b))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def dedupe_similar_hits(
    hits: list[dict[str, Any]],
    *,
    similarity_threshold: float = 0.88,
) -> list[dict[str, Any]]:
    """Remove repeated or near-identical retrieved chunks while preserving rank order."""
    deduped: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()
    seen_fingerprints: set[str] = set()
    removed = 0

    for hit in hits:
        chunk_id = str(hit.get("chunk_id") or "")
        if chunk_id and chunk_id in seen_chunk_ids:
            removed += 1
            continue

        text = str(hit.get("text") or "")
        fingerprint = normalized_text_fingerprint(text)
        if fingerprint and fingerprint in seen_fingerprints:
            removed += 1
            continue

        if any(token_jaccard(text, str(existing.get("text") or "")) >= similarity_threshold for existing in deduped):
            removed += 1
            continue

        deduped.append(hit)
        if chunk_id:
            seen_chunk_ids.add(chunk_id)
        if fingerprint:
            seen_fingerprints.add(fingerprint)

    return deduped


def route_query_expansion(profile: dict[str, Any]) -> str:
    additions: list[str] = []
    for route in profile.get("metadata_routes", []):
        expansion = ROUTE_QUERY_EXPANSIONS.get(str(route.get("route") or ""))
        if expansion:
            additions.append(expansion)
    return " ".join(dict.fromkeys(" ".join(additions).split()))


def context_pack_doc_ids(question: str, profile: dict[str, Any]) -> list[str]:
    policy = profile.get("answer_policy") or determine_answer_policy(question, profile)
    doc_ids = list(CONTEXT_PACKS.get(policy, []))

    for route in profile.get("metadata_routes", []):
        for doc_id in route.get("doc_ids", []):
            if doc_id not in doc_ids:
                doc_ids.insert(0, doc_id)

    if profile.get("intent") in SUMMARY_INTENT_CONFIG:
        index_doc_id = SUMMARY_INTENT_CONFIG[profile["intent"]]["index_doc_id"]
        if index_doc_id not in doc_ids:
            doc_ids.insert(0, index_doc_id)

    return doc_ids


def augment_hits_with_context_pack(
    hits: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    question: str,
    profile: dict[str, Any],
    chunks_per_doc: int = 1,
) -> list[dict[str, Any]]:
    """Prepend small, known-good context packs for common question families."""
    doc_ids = context_pack_doc_ids(question, profile)
    if not doc_ids:
        return hits

    existing_chunk_ids = {hit.get("chunk_id") for hit in hits}
    score_seed = max((float(hit.get("score", 0)) for hit in hits), default=0.55)
    score_seed = max(score_seed, 0.965)
    front: list[dict[str, Any]] = []

    for doc_id in doc_ids:
        doc_rows = sorted(
            [row for row in rows if row.get("doc_id") == doc_id],
            key=lambda row: int(row.get("chunk_index") or 0),
        )
        added_for_doc = 0
        for row in doc_rows:
            cid = row.get("chunk_id")
            if not cid or cid in existing_chunk_ids:
                continue
            front.append(make_hit_from_row(row, score_seed))
            existing_chunk_ids.add(cid)
            score_seed -= 0.001
            added_for_doc += 1
            if added_for_doc >= chunks_per_doc:
                break

    profile["context_pack"] = {
        "doc_ids": doc_ids,
        "added_chunks": len(front),
    }
    return front + [hit for hit in hits if hit.get("chunk_id") not in {item.get("chunk_id") for item in front}]


def generation_rerank_score(question: str, hit: dict[str, Any], profile: dict[str, Any]) -> float:
    score = float(hit.get("score") or 0.0)
    text = normalized_hit_text(hit).lower()
    title = str(hit.get("title") or "").lower()
    doc_id = str(hit.get("doc_id") or "")
    page_type = str(hit.get("page_type") or "")
    question_tokens = {
        token for token in query_kb.tokenize(question) if token not in query_kb.STOPWORDS
    }
    text_tokens = set(query_kb.tokenize(text))

    if question_tokens:
        score += min(len(question_tokens & text_tokens) / len(question_tokens), 1.0) * 0.08

    metadata_doc_ids = {
        route_doc_id
        for route in profile.get("metadata_routes", [])
        for route_doc_id in route.get("doc_ids", [])
    }
    if doc_id in metadata_doc_ids:
        score += 0.42

    pack_doc_ids = set(context_pack_doc_ids(question, profile))
    if doc_id in pack_doc_ids:
        score += 0.18

    policy = profile.get("answer_policy") or determine_answer_policy(question, profile)
    if policy in {"product_catalog", "product_detail", "service_guidance"} and page_type == "product":
        score += 0.06
    if policy in {"service_catalog", "voice_agent"} and page_type == "service":
        score += 0.06
    if policy in {"contact", "purchase", "high_risk_unknown"} and doc_id == "contact-us":
        score += 0.25
    if profile.get("company_overview") and (page_type == "about" or doc_id == "about-us"):
        score += 0.16
    if title and any(alias in title for alias in profile.get("product_aliases", [])):
        score += 0.2
    if is_biek_knowledge_domain(profile.get("knowledge_domain")):
        year_bonus, year_debug = biek_year_rerank_bonus(question, hit)
        if year_bonus:
            score += year_bonus
            score_components = dict(hit.get("score_components") or {})
            score_components["biek_year_bias"] = round(year_bonus, 4)
            hit["score_components"] = score_components
        hit["biek_year_rerank"] = year_debug

    return score


def rerank_hits_for_generation(
    question: str,
    hits: list[dict[str, Any]],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    reranked: list[dict[str, Any]] = []
    for hit in hits:
        updated = dict(hit)
        updated["generation_score"] = generation_rerank_score(question, updated, profile)
        reranked.append(updated)
    reranked.sort(
        key=lambda item: (
            float(item.get("generation_score") or 0.0),
            float(item.get("score") or 0.0),
            float(item.get("raw_score") or 0.0),
        ),
        reverse=True,
    )
    return reranked


def dynamic_context_k(base_context_k: int, hits: list[dict[str, Any]], profile: dict[str, Any]) -> int:
    if not hits:
        return base_context_k

    policy = profile.get("answer_policy") or "generic"
    intent = profile.get("intent")
    confidence = profile.get("retrieval_confidence") or {}
    score_gap = float(confidence.get("score_gap") or 0.0)
    top_similarity = float(confidence.get("top_similarity") or 0.0)

    selected = base_context_k
    if policy == "product_catalog" or intent in SUMMARY_INTENT_CONFIG:
        selected = max(selected, min(6, len(hits)))
    elif policy in {"service_catalog", "industry"}:
        selected = max(selected, min(5, len(hits)))
    elif policy in {"contact", "purchase", "high_risk_unknown"}:
        selected = min(selected, 2)
    elif policy in {"product_detail", "private_infrastructure", "voice_agent"}:
        selected = min(max(selected, 2), 3)
    elif score_gap < 0.03 or top_similarity < 0.42:
        selected = min(max(selected + 1, 4), 5)

    return max(1, min(selected, len(hits)))


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
    summary_context_k = max(context_k, min(len(detail_doc_ids) + 1, 6))

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

    resolved_context_k = dynamic_context_k(context_k, hits, profile)
    profile["dynamic_context"] = {
        "requested_context_k": context_k,
        "selected_context_k": resolved_context_k,
        "reason": profile.get("answer_policy") or profile.get("intent") or "generic",
    }
    context_cache_key = (
        tuple(str(hit.get("chunk_id") or "") for hit in hits[:12]),
        resolved_context_k,
        profile.get("answer_policy"),
        profile.get("intent"),
        tuple(
            (route.get("route"), tuple(route.get("doc_ids", [])))
            for route in profile.get("metadata_routes", [])
        ),
    )
    cached_context = _CONTEXT_SELECTION_CACHE.get(context_cache_key)
    if cached_context is not None:
        profile["context_selection_cache_hit"] = True
        return copy.deepcopy(cached_context)

    context_k = resolved_context_k
    top_hit = hits[0]
    entity_terms = profile.get("entity_terms", [])
    intent = profile.get("intent")
    selected: list[dict[str, Any]]

    if profile.get("product_comparison"):
        selected = hits[:context_k]
        _CONTEXT_SELECTION_CACHE[context_cache_key] = copy.deepcopy(selected)
        return selected

    if profile.get("single_product_recommendation"):
        product_hits = [
            hit
            for hit in hits
            if hit.get("page_type") == "product" and hit.get("doc_id") != "product"
        ]
        metadata_doc_ids = [
            doc_id
            for route in profile.get("metadata_routes", [])
            for doc_id in route.get("doc_ids", [])
        ]
        preferred_product_hits = [
            hit for hit in product_hits if hit.get("doc_id") in metadata_doc_ids
        ]
        if preferred_product_hits:
            selected = preferred_product_hits[:context_k]
            _CONTEXT_SELECTION_CACHE[context_cache_key] = copy.deepcopy(selected)
            return selected
        if product_hits:
            score_by_doc: dict[str, float] = {}
            for hit in product_hits:
                doc_id = str(hit.get("doc_id") or "")
                score_by_doc[doc_id] = score_by_doc.get(doc_id, 0.0) + float(
                    hit.get("score") or 0.0
                )
            best_doc_id = max(score_by_doc, key=score_by_doc.get)
            best_product_hits = [
                hit for hit in product_hits if hit.get("doc_id") == best_doc_id
            ]
            selected = best_product_hits[:context_k]
            _CONTEXT_SELECTION_CACHE[context_cache_key] = copy.deepcopy(selected)
            return selected

    if profile.get("purchase_intent") or profile.get("pricing_intent"):
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
        selected = merged[:context_k]
        _CONTEXT_SELECTION_CACHE[context_cache_key] = copy.deepcopy(selected)
        return selected

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
        selected = merged[:context_k]
        _CONTEXT_SELECTION_CACHE[context_cache_key] = copy.deepcopy(selected)
        return selected

    if intent in SUMMARY_INTENT_CONFIG and not profile.get("product_aliases"):
        selected = select_summary_hits(hits, context_k=context_k, intent=intent)
        _CONTEXT_SELECTION_CACHE[context_cache_key] = copy.deepcopy(selected)
        return selected

    if (entity_terms and not profile.get("product_aliases")) or intent in {"contact", "about"}:
        same_doc_hits = [hit for hit in hits if hit.get("doc_id") == top_hit.get("doc_id")]
        if same_doc_hits:
            selected = same_doc_hits[:context_k]
            _CONTEXT_SELECTION_CACHE[context_cache_key] = copy.deepcopy(selected)
            return selected

    selected = hits[:context_k]
    _CONTEXT_SELECTION_CACHE[context_cache_key] = copy.deepcopy(selected)
    return selected


def product_card_for_prompt(hits: list[dict[str, Any]], profile: dict[str, Any]) -> dict[str, Any]:
    if determine_answer_policy("", profile) != "product_detail" and not profile.get("product_aliases"):
        return {}
    product_name = product_name_from_hits_or_profile(hits, profile)
    card = PRODUCT_CARDS.get(product_name)
    if not isinstance(card, dict):
        return {}
    return {
        "name": product_name,
        "type": card.get("type", "product"),
        "short_description": card.get("short_description", ""),
        "aliases": card.get("aliases", []),
    }


def format_structured_runtime_context(
    question: str,
    hits: list[dict[str, Any]],
    profile: dict[str, Any],
) -> str:
    guidance_intent = ops_guidance_intent(question)
    if guidance_intent:
        guidance = build_ops_guidance_answer(guidance_intent)
        if guidance:
            return (
                "Structured runtime context:\n"
                f"Approved operational guidance: {guidance}\n\n"
                "Use this as grounded context, but generate the final answer naturally."
            )

    if is_product_comparison_question(question):
        cards: list[dict[str, Any]] = []
        for product_name in product_names_in_question(question):
            card = PRODUCT_CARDS.get(product_name)
            if not isinstance(card, dict):
                continue
            cards.append(
                {
                    "name": product_name,
                    "short_description": card.get("short_description", ""),
                    "aliases": card.get("aliases", []),
                }
            )
        if cards:
            return (
                "Structured runtime context:\n"
                "Product cards JSON:\n"
                f"{json.dumps(cards, ensure_ascii=False)}\n\n"
                "Compare every named product. Use these cards to stabilize names and scope, "
                "then ground the final comparison in the retrieved evidence."
            )

    if determine_answer_policy(question, profile) != "product_detail":
        return ""
    product_card = product_card_for_prompt(hits, profile)
    if not product_card:
        return ""
    return (
        "Structured runtime context:\n"
        "Product card JSON:\n"
        f"{json.dumps(product_card, ensure_ascii=False)}\n\n"
        "Use this product card only to stabilize product name, aliases, and approved short wording. "
        "Still answer from the evidence and do not add unsupported facts."
    )


def build_user_prompt(
    question: str,
    hits: list[dict[str, Any]],
    profile: dict[str, Any],
    *,
    natural_output: bool = False,
    model: str = "",
) -> str:
    evidence = build_evidence_prompt(question, hits, profile)
    structured_context = format_structured_runtime_context(question, hits, profile)
    intent = profile.get("intent")
    extra_guidance = "Answer the question directly in the first sentence."
    if is_pricing_or_quote_question(question):
        extra_guidance = (
            "If public pricing or exact costs are not explicitly stated in the evidence, use the not-confirmed reply "
            "exactly as instructed above. Never answer pricing questions by listing product names only."
        )
    elif is_product_comparison_question(question):
        names = product_names_in_question(question)
        extra_guidance = (
            f"Compare both named products explicitly: {join_list(names)}. Give each product its own factual "
            "description, then state the practical difference in use case. Do not omit either product and do "
            "not invent capabilities that are absent from the evidence."
        )
    elif is_product_selection_question(question):
        extra_guidance = (
            "Give decision criteria grounded in the evidence. If comparisons are not in the evidence, "
            "use the not-confirmed reply; do not answer with a bare product catalog only."
        )
    elif is_single_product_recommendation_question(question):
        extra_guidance = (
            "Recommend the single Synapse Tech product supported by the evidence as the closest fit for "
            "the user's stated need. Explain briefly why it fits. Do not list unrelated products."
        )
    elif is_deploy_priority_question(question):
        extra_guidance = (
            "Give one concrete first-step recommendation only if the evidence supports it; otherwise "
            "use the not-confirmed reply. Do not substitute a generic Synapse capability overview."
        )
    elif is_custom_software_question(question):
        extra_guidance = (
            "The user is asking whether Synapse Tech only builds custom software or also offers AI products. "
            "Answer the contrast directly from the evidence. If the evidence mentions custom web/mobile app "
            "development and pre-built AI products, say that Synapse Tech does both. Do not answer with only "
            "a product catalog."
        )
    elif is_workflow_automation_question(question):
        extra_guidance = (
            "The user is asking how Synapse Tech helps with workflow automation. Explain the supported "
            "automation capabilities from the evidence, such as connecting tools, APIs, databases, SaaS "
            "systems, orchestration, or workflow actions. Focus on what this helps the business do, not only "
            "on listing tool names. n8n, Make.com, Airbyte, and Airtable are third-party tools/platforms "
            "in the automation stack, not Synapse Tech products. Do not use the not-confirmed reply when "
            "the evidence contains workflow automation details."
        )
    elif is_private_deployment_question(question):
        extra_guidance = (
            "The user is asking about offline or private infrastructure. Answer only for products where the "
            "evidence supports private/offline deployment. If the evidence supports Opira AI running offline, "
            "on-premises, or private cloud, mention that specifically. If the user asks about tools broadly, "
            "say that not all Synapse tools are confirmed to run offline. Do not claim every Synapse tool works offline."
        )
    elif wants_product_descriptions(question):
        extra_guidance = (
            "The user is asking for a short explanation of Synapse Tech products. Use exact product names from "
            "the evidence and add a brief grounded phrase for what each product does. Do not invent product names."
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
                "and explain what it is and what it helps with in plain language. Use the structured product card "
                "to stabilize product name and approved short wording, but generate the final answer naturally. "
                "If the retrieved evidence has that product title or product page, treat the product as supported "
                "and paraphrase the evidence."
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

    model_profile = get_model_profile(model)
    model_answer_guidance = str(model_profile.get("answer_instructions") or "").strip()
    if model_answer_guidance:
        extra_guidance = (
            f"{extra_guidance}\n"
            f"{shared_grounded_answer_instructions()}\n"
            f"{model_answer_guidance}"
        )

    output_requirement = (
        "Return only the natural final answer.\n"
        if natural_output
        else 'Return JSON only, using keys "answer" and "supported".\n'
    )
    if profile.get("company_overview"):
        requirements = (
            "Answer requirements:\n"
            "Use only the evidence above. It includes Synapse Tech company pages; produce a direct factual answer.\n"
            f"{extra_guidance}\n"
            "Write 2-4 concise sentences unless the user asked for fewer (then respect that count).\n"
            "Do not mention the evidence, assumptions, or these instructions.\n"
            f"{output_requirement}"
        )
    else:
        requirements = (
            "Answer requirements:\n"
            f'Use only the evidence above. If the evidence is insufficient, reply exactly with: "{DEFAULT_FALLBACK_RESPONSE}"\n'
            f"{extra_guidance}\n"
            "Write 2-4 concise sentences.\n"
            "Do not invent exact timelines, percentages, guarantees, or outcomes unless they appear in the evidence.\n"
            "Do not mention the evidence, assumptions, or these instructions.\n"
            f"{output_requirement}"
        )

    prompt = (
        f"Question:\n{question}\n\n"
        f"Evidence:\n{evidence}\n\n"
        f"{structured_context + chr(10) + chr(10) if structured_context else ''}"
        f"{requirements}"
    )
    profile["prompt_metrics"] = {
        "question_chars": len(question),
        "evidence_chars": len(evidence),
        "structured_context_chars": len(structured_context),
        "requirements_chars": len(requirements),
        "user_prompt_chars": len(prompt),
        "selected_context_chunks": len(hits),
    }
    return prompt


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


def infer_retrieval_domain(index_path: Path, metadata_path: Path, manifest_path: Path) -> str:
    haystack = " ".join(
        (
            str(index_path).lower(),
            str(metadata_path).lower(),
            str(manifest_path).lower(),
        )
    )
    if "/biek/" in haystack or "biek" in metadata_path.name.lower():
        return "biek"
    return "synapse"


def parse_biek_time_intent(question: str) -> dict[str, Any]:
    lowered = question.lower()
    year_matches = [int(match) for match in re.findall(r"\b(20\d{2})\b", lowered)]
    explicit_year = max(year_matches) if year_matches else None
    has_recency_intent = _contains_any(
        lowered,
        (
            "latest",
            "current",
            "recent",
            "last date",
            "this year",
            "nowadays",
            "upcoming",
        ),
    )
    has_time_sensitive_intent = has_recency_intent or _contains_any(
        lowered,
        (
            "enrolment",
            "enrollment",
            "exam form",
            "examination form",
            "fee",
            "fees",
            "date sheet",
            "datesheet",
            "notification",
            "notifications",
            "result",
            "results",
            "schedule",
            "deadline",
            "annual examination",
            "supplementary",
        ),
    )
    return {
        "explicit_year": explicit_year,
        "has_recency_intent": has_recency_intent,
        "has_time_sensitive_intent": has_time_sensitive_intent,
    }


def extract_biek_hit_year(hit: dict[str, Any]) -> int | None:
    academic_year = str(hit.get("academic_year") or "").strip()
    academic_years = [int(match) for match in re.findall(r"\b(20\d{2})\b", academic_year)]
    if academic_years:
        return max(academic_years)

    issue_date = str(hit.get("issue_date") or "").strip()
    issue_match = re.search(r"\b(20\d{2})\b", issue_date)
    if issue_match:
        return int(issue_match.group(1))

    for field in ("title", "url"):
        value = str(hit.get(field) or "")
        matches = [int(match) for match in re.findall(r"\b(20\d{2})\b", value)]
        if matches:
            return max(matches)
    return None


def is_biek_time_sensitive_hit(hit: dict[str, Any]) -> bool:
    document_type = str(hit.get("document_type") or "").lower()
    section = str(hit.get("section") or "").lower()
    action_type = str(hit.get("action_type") or "").lower()
    title_blob = f"{hit.get('title') or ''} {normalized_hit_text(hit)[:1200]}".lower()

    if document_type in {"notification", "datesheet", "results_document"}:
        return True
    if section in {"notifications", "datesheet", "results"}:
        return True
    if document_type == "form" and action_type in {
        "exam_form_submission",
        "enrolment",
        "registration",
        "fee_voucher",
    }:
        return True
    if document_type == "form" and _contains_any(
        title_blob,
        (
            "enrolment",
            "enrollment",
            "registration",
            "exam form",
            "examination form",
            "fee voucher",
            "fee structure",
            "date sheet",
            "datesheet",
        ),
    ):
        return True
    return False


def biek_year_rerank_bonus(question: str, hit: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    intent = parse_biek_time_intent(question)
    if intent["explicit_year"] is None and not intent["has_time_sensitive_intent"]:
        return 0.0, {"applied": False, "reason": "non_time_sensitive_question"}
    if not is_biek_time_sensitive_hit(hit):
        return 0.0, {"applied": False, "reason": "non_time_sensitive_hit"}

    hit_year = extract_biek_hit_year(hit)
    if hit_year is None:
        return 0.0, {"applied": False, "reason": "missing_hit_year"}

    explicit_year = intent["explicit_year"]
    has_recency_intent = bool(intent["has_recency_intent"])
    bonus = 0.0

    if explicit_year is not None:
        if hit_year == explicit_year:
            bonus = 0.34
        elif hit_year == explicit_year - 1:
            bonus = 0.08
        elif hit_year < explicit_year - 1:
            bonus = -0.14
        else:
            bonus = -0.04
    elif has_recency_intent:
        if hit_year >= 2026:
            bonus = 0.26
        elif hit_year == 2025:
            bonus = 0.15
        else:
            bonus = -0.12
    else:
        if hit_year >= 2026:
            bonus = 0.14
        elif hit_year == 2025:
            bonus = 0.07
        else:
            bonus = -0.06

    return bonus, {
        "applied": True,
        "explicit_year": explicit_year,
        "has_recency_intent": has_recency_intent,
        "has_time_sensitive_intent": bool(intent["has_time_sensitive_intent"]),
        "hit_year": hit_year,
        "time_sensitive_hit": True,
    }


def is_biek_domain(profile: dict[str, Any]) -> bool:
    return is_biek_knowledge_domain(profile.get("knowledge_domain"))


def biek_supported_current_years() -> tuple[int, int]:
    return (2026, 2025)


def normalize_biek_nav_text(value: str) -> str:
    text = str(value or "").lower()
    replacements = (
        (r"\be[\s\-]?sheet\b", "esheet"),
        (r"\bdate[\s\-]?sheet\b", "datesheet"),
        (r"\bmark[\s\-]?sheet\b", "marksheet"),
        (r"\benrollment\b", "enrolment"),
        (r"\bcomputerised\b", "computerized"),
        (r"\bmcqs\b", "mcq"),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def canonicalize_biek_nav_token(token: str) -> str:
    token = normalize_biek_nav_text(token)
    if not token:
        return ""

    direct_map = {
        "banks": "bank",
        "cards": "card",
        "certificates": "certificate",
        "committees": "committee",
        "downloads": "download",
        "esheets": "esheet",
        "forms": "form",
        "members": "member",
        "models": "model",
        "notifications": "notification",
        "papers": "paper",
        "results": "result",
        "statistics": "statistic",
        "studies": "study",
        "verifications": "verification",
        "vouchers": "voucher",
    }
    if token in direct_map:
        return direct_map[token]

    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
        return token[:-1]
    return token


def biek_nav_tokens(value: str) -> list[str]:
    stopwords = {
        "a",
        "about",
        "an",
        "and",
        "are",
        "available",
        "can",
        "check",
        "do",
        "does",
        "find",
        "for",
        "have",
        "here",
        "i",
        "is",
        "me",
        "of",
        "on",
        "page",
        "please",
        "show",
        "the",
        "this",
        "u",
        "website",
        "what",
        "where",
        "with",
        "you",
        "your",
        "biek",
    }
    tokens = []
    for token in normalize_biek_nav_text(value).split():
        token = canonicalize_biek_nav_token(token)
        if len(token) <= 1 or token in stopwords:
            continue
        tokens.append(token)
    return tokens


def load_biek_nav_coverage() -> list[dict[str, Any]]:
    global _BIEK_NAV_COVERAGE_CACHE
    if _BIEK_NAV_COVERAGE_CACHE is not None:
        return _BIEK_NAV_COVERAGE_CACHE

    try:
        payload = json.loads(DEFAULT_BIEK_NAV_COVERAGE_PATH.read_text(encoding="utf-8"))
        rows = payload.get("rows") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            rows = []
    except Exception:
        rows = []

    _BIEK_NAV_COVERAGE_CACHE = list(rows)
    return _BIEK_NAV_COVERAGE_CACHE


def find_biek_nav_row_by_label(label: str) -> dict[str, Any] | None:
    target = normalize_biek_nav_text(label)
    if not target:
        return None
    for row in load_biek_nav_coverage():
        if normalize_biek_nav_text(str(row.get("label") or "")) == target:
            return row
    return None


def biek_nav_row_link(label: str) -> str:
    row = find_biek_nav_row_by_label(label)
    if row is None:
        return ""
    return str(row.get("url") or "").strip()


def load_biek_nav_content_map() -> dict[str, Any]:
    global _BIEK_NAV_CONTENT_MAP_CACHE
    if _BIEK_NAV_CONTENT_MAP_CACHE is not None:
        return _BIEK_NAV_CONTENT_MAP_CACHE

    payload: dict[str, Any] = {"rows": []}
    try:
        with DEFAULT_BIEK_CONTENT_MAP_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            payload = data
    except Exception:
        payload = {"rows": []}

    _BIEK_NAV_CONTENT_MAP_CACHE = payload
    return _BIEK_NAV_CONTENT_MAP_CACHE


def normalize_biek_target_url(value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        return ""
    normalized = normalized.replace("http://", "https://")
    normalized = normalized.replace("https://biek.edu.pk", "https://www.biek.edu.pk")
    return normalized.rstrip("/")


def load_biek_enriched_documents() -> list[dict[str, Any]]:
    global _BIEK_ENRICHED_DOCS_CACHE
    if _BIEK_ENRICHED_DOCS_CACHE is not None:
        return _BIEK_ENRICHED_DOCS_CACHE

    rows: list[dict[str, Any]] = []
    try:
        with DEFAULT_BIEK_ENRICHED_DOCS_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except Exception:
        rows = []

    _BIEK_ENRICHED_DOCS_CACHE = rows
    return _BIEK_ENRICHED_DOCS_CACHE


def load_biek_raw_html_pages() -> list[dict[str, Any]]:
    global _BIEK_RAW_HTML_CACHE
    if _BIEK_RAW_HTML_CACHE is not None:
        return _BIEK_RAW_HTML_CACHE

    rows: list[dict[str, Any]] = []
    for path in sorted(DEFAULT_BIEK_RAW_HTML_DIR.glob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        lines = [line.rstrip() for line in text.splitlines()]
        if len(lines) < 2:
            continue
        title = lines[0].strip()
        url_line = lines[1].strip()
        url = url_line.split("URL:", 1)[1].strip() if url_line.lower().startswith("url:") else url_line
        source_page = ""
        content_start = 2
        if len(lines) > 2 and lines[2].lower().startswith("source:"):
            source_page = lines[2].split(":", 1)[1].strip()
            content_start = 3
        body = "\n".join(lines[content_start:]).strip()
        rows.append(
            {
                "source_kind": "html_page",
                "source_url": normalize_biek_target_url(url),
                "source_page": normalize_biek_target_url(source_page),
                "title": title,
                "text": body,
                "path": str(path),
            }
        )

    _BIEK_RAW_HTML_CACHE = rows
    return _BIEK_RAW_HTML_CACHE


def biek_content_docs() -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for row in load_biek_enriched_documents():
        docs.append(
            {
                "source_kind": str(row.get("source_kind") or row.get("document_type") or ""),
                "source_url": normalize_biek_target_url(str(row.get("source_url") or row.get("url") or "")),
                "source_page": normalize_biek_target_url(str(row.get("source_page") or "")),
                "title": str(row.get("title") or ""),
                "text": str(row.get("clean_body_text") or ""),
                "document_type": str(row.get("document_type") or ""),
                "section": str(row.get("section") or ""),
                "download_links": [normalize_biek_target_url(link) for link in row.get("download_links", []) if str(link).strip()],
                "metadata": row,
            }
        )
    docs.extend(load_biek_raw_html_pages())
    return docs


def biek_content_text(doc: dict[str, Any]) -> str:
    text = str(doc.get("text") or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def biek_doc_label(doc: dict[str, Any]) -> str:
    return clean_answer_text(
        str(
            doc.get("title")
            or doc.get("label")
            or doc.get("source_url")
            or doc.get("source_page")
            or "BIEK content"
        )
    )


def is_weak_biek_content(text: str) -> bool:
    cleaned = biek_content_text({"text": text})
    if len(cleaned) < 80:
        return True
    words = cleaned.split()
    if len(words) < 15:
        return True
    return False


def biek_text_sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return []
    parts = re.split(r"(?<=[.!?])\s+", compact)
    return [part.strip(" -") for part in parts if part.strip(" -")]


def biek_text_snippets(query_text: str, text: str, *, limit: int = 3, max_chars: int = 700) -> list[str]:
    units = split_evidence_units(text)
    if not units:
        units = biek_text_sentences(text)
    if not units:
        return []

    query_tokens = set(biek_coverage_query_tokens(query_text)) | set(biek_nav_tokens(query_text))
    scored: list[tuple[float, int, str]] = []
    for index, unit in enumerate(units):
        unit_tokens = set(biek_nav_tokens(unit))
        overlap = len(query_tokens & unit_tokens)
        score = float(overlap)
        if index == 0:
            score += 0.25
        if len(unit_tokens) < 4:
            score -= 0.5
        scored.append((score, index, unit.strip()))

    chosen: list[str] = []
    used_chars = 0
    for _score, _index, unit in sorted(scored, key=lambda item: (item[0], -item[1]), reverse=True):
        if not unit:
            continue
        if chosen and _score <= 0:
            continue
        separator = 1 if chosen else 0
        remaining = max_chars - used_chars - separator
        if remaining <= 0:
            break
        if len(unit) > remaining:
            unit = unit[: max(1, remaining - 3)].rstrip(" ,;:") + "..."
        if unit not in chosen:
            chosen.append(unit)
            used_chars += len(unit) + separator
        if len(chosen) >= limit:
            break

    if not chosen:
        fallback = " ".join(units[:limit]).strip()
        if len(fallback) > max_chars:
            fallback = fallback[: max_chars - 3].rstrip(" ,;:") + "..."
        if fallback:
            return [fallback]
    return chosen


def extract_relevant_biek_excerpt(question: str, text: str, *, limit: int = 3) -> list[str]:
    sentences = biek_text_sentences(text)
    if not sentences:
        return []
    query_tokens = set(biek_coverage_query_tokens(question))
    scored: list[tuple[int, int, str]] = []
    for index, sentence in enumerate(sentences):
        tokens = set(biek_nav_tokens(sentence))
        overlap = len(query_tokens & tokens)
        scored.append((overlap, -index, sentence))
    scored.sort(reverse=True)
    chosen: list[str] = []
    for overlap, _neg_index, sentence in scored:
        if overlap <= 0 and chosen:
            continue
        if sentence not in chosen:
            chosen.append(sentence)
        if len(chosen) >= limit:
            break
    if not chosen:
        return sentences[:limit]
    return chosen


def extract_biek_contact_details(text: str) -> tuple[list[str], list[str]]:
    phones: list[str] = []
    emails: list[str] = []
    for phone in re.findall(r"\(?\+?\d[\d()\-\s]{6,}\d", text):
        cleaned = " ".join(phone.split())
        if cleaned not in phones:
            phones.append(cleaned)
    for email in re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
        if email not in emails:
            emails.append(email)
    return phones[:4], emails[:3]


def has_person_like_biek_content(text: str) -> bool:
    if re.search(r"\b(MR|MS|MRS|DR|PROF|COL)\.?\b", text):
        return True
    if re.search(r"\b[A-Z]{2,}(?: [A-Z]{2,}){1,5}\b", text):
        return True
    return False


def is_biek_identity_row(row: dict[str, Any]) -> bool:
    blob = " ".join(
        [
            str(row.get("label") or ""),
            str(row.get("parent") or ""),
            str(row.get("url") or ""),
        ]
    ).lower()
    return _contains_any(
        blob,
        (
            "chairman",
            "board members",
            "bmembers",
            "committee",
            "committees",
        ),
    )


def has_strong_biek_identity_page_content(row: dict[str, Any], doc: dict[str, Any] | None) -> bool:
    if doc is None:
        return False

    text = biek_content_text(doc)
    if is_weak_biek_content(text):
        return False

    surface = " ".join(
        [
            str(row.get("label") or ""),
            str(row.get("url") or ""),
            str(doc.get("source_url") or ""),
            str(doc.get("source_page") or ""),
            str(doc.get("title") or ""),
        ]
    ).lower()
    if not _contains_any(surface, ("chairman", "board members", "bmembers", "committee", "committees")):
        return False

    low = text.lower()
    mismatch_tokens = (
        "tender",
        "procurement",
        "contract award",
        "bid evaluation",
        "evaluation report",
        "result gazette",
        "result declaration",
        "notification",
        "schedule for submission",
        "pay order",
        "bidding document",
    )
    if any(token in low for token in mismatch_tokens):
        return False

    if "chairman" in surface:
        return has_person_like_biek_content(text) or "chairman" in low
    if _contains_any(surface, ("board members", "bmembers")):
        return "board members" in low or has_person_like_biek_content(text)
    if _contains_any(surface, ("committee", "committees")):
        return "committee" in low or "committees" in low
    return False


def find_biek_doc_by_text_file(path: str, docs: list[dict[str, Any]]) -> dict[str, Any] | None:
    target = str(path or "").strip()
    if not target:
        return None
    for doc in docs:
        metadata = doc.get("metadata") or {}
        doc_paths = [
            str(doc.get("path") or "").strip(),
            str(metadata.get("text_file") or "").strip(),
        ]
        if target in doc_paths:
            return doc
    return None


def biek_content_map_row_for_nav_row(row: dict[str, Any]) -> dict[str, Any] | None:
    row_id = str(row.get("id") or "").strip()
    if not row_id:
        return None
    for item in load_biek_nav_content_map().get("rows", []):
        if str(item.get("id") or "").strip() == row_id:
            return item
    return None


def resolve_biek_nav_primary_doc(row: dict[str, Any], docs: list[dict[str, Any]]) -> dict[str, Any] | None:
    content_row = biek_content_map_row_for_nav_row(row)
    preferred_source = (content_row or {}).get("preferred_source") or {}
    preferred_text_file = str(preferred_source.get("text_file") or "").strip()
    if preferred_text_file:
        matched = find_biek_doc_by_text_file(preferred_text_file, docs)
        if matched is not None:
            return matched
    return find_biek_content_for_nav_row(row, docs)


def biek_target_support_docs(
    row: dict[str, Any],
    coverage_rows: list[dict[str, Any]],
    docs: list[dict[str, Any]],
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    support: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    label = str(row.get("label") or "").strip().lower()
    child_rows = [
        candidate
        for candidate in coverage_rows
        if str(candidate.get("parent") or "").strip().lower() == label
    ]

    for child_row in child_rows:
        child_doc = resolve_biek_nav_primary_doc(child_row, docs)
        if child_doc is None:
            continue
        child_url = normalize_biek_target_url(str(child_doc.get("source_url") or child_doc.get("source_page") or ""))
        if child_url and child_url in seen_urls:
            continue
        support.append(child_doc)
        if child_url:
            seen_urls.add(child_url)
        if len(support) >= limit:
            break

    if len(support) < limit:
        for support_doc in search_biek_content_docs(str(row.get("label") or ""), docs, limit=limit + 2):
            support_url = normalize_biek_target_url(str(support_doc.get("source_url") or support_doc.get("source_page") or ""))
            if support_url and support_url in seen_urls:
                continue
            support.append(support_doc)
            if support_url:
                seen_urls.add(support_url)
            if len(support) >= limit:
                break

    return support[:limit]


def build_biek_target_evidence_pack(
    question: str,
    row: dict[str, Any],
    coverage_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    docs = biek_content_docs()
    primary_doc = resolve_biek_nav_primary_doc(row, docs)
    target_kind = str(row.get("handling_mode") or "").strip().lower()
    evidence: list[dict[str, str]] = []

    if is_biek_identity_row(row) and not has_strong_biek_identity_page_content(row, primary_doc):
        return []

    def append_doc(doc: dict[str, Any] | None, *, role: str, label_override: str | None = None) -> None:
        if doc is None:
            return
        text = biek_content_text(doc)
        if not text:
            return
        query_text = f"{question} {row.get('label') or ''} {label_override or ''}".strip()
        snippet_limit = 4 if target_kind == "container_summary" else 3
        max_chars = 900 if target_kind == "container_summary" else 650
        snippets = biek_text_snippets(query_text, text, limit=snippet_limit, max_chars=max_chars)
        if not snippets and not is_weak_biek_content(text):
            snippets = [text[:max_chars].strip()]
        if not snippets:
            return
        source_url = normalize_biek_target_url(str(doc.get("source_url") or doc.get("source_page") or row.get("url") or ""))
        evidence.append(
            {
                "role": role,
                "label": clean_answer_text(label_override or biek_doc_label(doc)),
                "url": source_url,
                "text": " ".join(snippets).strip(),
            }
        )

    append_doc(primary_doc, role="primary", label_override=str(row.get("label") or "").strip() or None)

    if target_kind == "container_summary":
        child_label = str(row.get("label") or "").strip()
        for support_doc in biek_target_support_docs(row, coverage_rows, docs, limit=4):
            append_doc(support_doc, role="child", label_override=child_label)

    if not evidence and primary_doc is not None:
        append_doc(primary_doc, role="primary")

    deduped: list[dict[str, str]] = []
    seen_keys: set[tuple[str, str]] = set()
    for item in evidence:
        key = (item.get("url") or "", item.get("text") or "")
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(item)
    return deduped[:5]


def format_biek_target_evidence_pack(evidence_pack: list[dict[str, str]]) -> str:
    blocks: list[str] = []
    for index, item in enumerate(evidence_pack, start=1):
        role = clean_answer_text(item.get("role") or "evidence")
        label = clean_answer_text(item.get("label") or f"BIEK source {index}")
        url = item.get("url") or ""
        text = clean_answer_text(item.get("text") or "")
        blocks.append(
            f"[{index}] Role: {role}\n"
            f"Label: {label}\n"
            f"URL: {url}\n"
            f"Evidence:\n{text}"
        )
    return "\n\n".join(blocks)


def generate_biek_target_scoped_answer(
    *,
    question: str,
    row: dict[str, Any],
    coverage_rows: list[dict[str, Any]],
    profile: dict[str, Any],
    ollama_url: str,
    model: str,
    temperature: float,
    num_predict: int,
) -> str | None:
    if str(row.get("handling_mode") or "").strip().lower() == "special_lookup":
        return build_biek_special_lookup_answer(question, row)

    if is_biek_identity_row(row):
        docs = biek_content_docs()
        primary_doc = resolve_biek_nav_primary_doc(row, docs)
        if not has_strong_biek_identity_page_content(row, primary_doc):
            return build_biek_identity_safe_answer()

    evidence_pack = build_biek_target_evidence_pack(question, row, coverage_rows)
    profile["biek_target_evidence"] = {
        "row_id": row.get("id"),
        "row_label": row.get("label"),
        "evidence_count": len(evidence_pack),
        "sources": [
            {
                "role": item.get("role"),
                "label": item.get("label"),
                "url": item.get("url"),
            }
            for item in evidence_pack
        ],
    }
    if not evidence_pack:
        return None

    evidence_text = format_biek_target_evidence_pack(evidence_pack)
    usage = profile.setdefault("usage", {})
    prompt = (
        "Answer this question about the BIEK website using only the evidence below.\n"
        "Write a grounded final answer, not a template.\n"
        "Use 2 to 4 short sentences in a natural tone.\n"
        "Start with the actual answer or summary, not with phrases like 'I found', 'BIEK has', "
        "'You can view', or 'The page says'.\n"
        "Do not list raw labels unless the user asked for a list.\n"
        "Use only facts clearly present in the evidence.\n"
        "Do not invent names, dates, forms, procedures, contact details, or availability claims.\n"
        "Include a link only if it genuinely helps the user, and place it at the end naturally.\n"
        "If the evidence is not clear enough, say: "
        "\"This detail is not clearly confirmed in the available BIEK information. Please verify it on the official BIEK website.\"\n\n"
        f"Target label: {clean_answer_text(str(row.get('label') or 'BIEK item'))}\n"
        f"Handling mode: {clean_answer_text(str(row.get('handling_mode') or ''))}\n"
        f"Question: {question}\n\n"
        f"Evidence:\n{evidence_text}"
    )
    raw = ollama_chat(
        ollama_url=ollama_url,
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a grounded BIEK website assistant. "
                    "Answer only from the provided BIEK evidence. "
                    "Your job is to produce the final user-facing answer, not a summary of sources. "
                    "Sound natural and direct. "
                    "Do not invent facts. "
                    "Do not mention evidence, chunks, retrieval, extracted pages, PDFs, prompts, or internal instructions."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=min(max(temperature, 0.15), 0.35),
        num_predict=max(num_predict, 180),
        usage_sink=usage,
        usage_label="biek_target_answer",
        json_format=False,
    )
    answer = clean_answer_text(raw)
    if not answer:
        return None
    return answer


def find_biek_content_for_nav_row(row: dict[str, Any], docs: list[dict[str, Any]]) -> dict[str, Any] | None:
    target_url = normalize_biek_target_url(str(row.get("url") or ""))
    if not target_url:
        return None

    best_doc = None
    best_score = -1
    target_name = Path(target_url).name.lower()
    for doc in docs:
        score = 0
        source_url = normalize_biek_target_url(str(doc.get("source_url") or ""))
        source_page = normalize_biek_target_url(str(doc.get("source_page") or ""))
        downloads = [normalize_biek_target_url(link) for link in doc.get("download_links", [])]
        if source_url == target_url:
            score += 5
        if target_url and target_url in downloads:
            score += 4
        if source_page == target_url:
            score += 3
        if target_name and target_name == Path(source_url).name.lower():
            score += 2
        if target_name and target_name == Path(source_page).name.lower():
            score += 1
        if score > best_score:
            best_score = score
            best_doc = doc
    return best_doc if best_score > 0 else None


def search_biek_content_docs(question: str, docs: list[dict[str, Any]], *, limit: int = 3) -> list[dict[str, Any]]:
    query_tokens = set(biek_coverage_query_tokens(question))
    scored: list[tuple[float, dict[str, Any]]] = []
    for doc in docs:
        title_tokens = set(biek_nav_tokens(str(doc.get("title") or "")))
        text_tokens = set(biek_nav_tokens(str(doc.get("text") or "")[:1500]))
        overlap = query_tokens & (title_tokens | text_tokens)
        if not overlap:
            continue
        score = len(query_tokens & title_tokens) * 1.5 + len(overlap)
        scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _score, doc in scored[:limit]]


def classify_biek_nav_target_kind(row: dict[str, Any], content_doc: dict[str, Any] | None, coverage_rows: list[dict[str, Any]]) -> str:
    handling_mode = str(row.get("handling_mode") or "").strip().lower()
    url = normalize_biek_target_url(str(row.get("url") or ""))
    label = str(row.get("label") or "").strip()
    if handling_mode == "special_lookup":
        return "special_lookup"
    if handling_mode == "document_summary" or url.lower().endswith(".pdf"):
        return "document_summary"
    if handling_mode == "container_summary" or row.get("level") == 0:
        return "container_summary"
    if any(str(candidate.get("parent") or "").strip().lower() == label.lower() for candidate in coverage_rows):
        return "container_summary"
    return "info_page"


def is_biek_detail_request(question: str) -> bool:
    return _contains_any(
        question.lower(),
        (
            "how to",
            "process",
            "procedure",
            "requirement",
            "requirements",
            "instruction",
            "instructions",
            "what does it require",
            "what do i need",
            "how do i",
            "steps",
            "documents needed",
            "supporting documents",
            "fee",
            "fees",
        ),
    )


def biek_document_purpose_snippet(content_doc: dict[str, Any]) -> str:
    meta = content_doc.get("metadata") or {}
    action_type = str(meta.get("action_type") or "").replace("_", " ").strip()
    if action_type:
        return f"It is used for {action_type}."

    text = biek_content_text(content_doc)
    if not text:
        return ""
    excerpt = extract_relevant_biek_excerpt("purpose use for form", text, limit=1)
    if not excerpt:
        return ""
    snippet = re.sub(r"\s+", " ", excerpt[0]).strip()
    if len(snippet) > 180:
        snippet = snippet[:177].rstrip(" ,;:") + "..."
    if snippet and not snippet.endswith((".", "!", "?")):
        snippet += "."
    return snippet


def build_biek_info_page_answer(question: str, row: dict[str, Any], content_doc: dict[str, Any]) -> str:
    text = biek_content_text(content_doc)
    url = str(row.get("url") or content_doc.get("source_url") or "").strip()
    label = clean_answer_text(str(row.get("label") or "this BIEK page"))

    if is_biek_identity_row(row) and not has_strong_biek_identity_page_content(row, content_doc):
        return build_biek_identity_safe_answer()

    if is_weak_biek_content(text):
        return f"I found the {label} page on the BIEK website. You can view it here: {url}"

    if is_biek_contact_question(question) or "contact" in label.lower():
        phones, emails = extract_biek_contact_details(text)
        detail_parts: list[str] = []
        if phones:
            detail_parts.append(f"phone numbers {', '.join(phones)}")
        if emails:
            detail_parts.append(f"email {', '.join(emails)}")
        if detail_parts:
            return f"BIEK contact details in the available page include {join_list(detail_parts)}. The page is here: {url}"

    excerpt = extract_relevant_biek_excerpt(question, text, limit=3)
    if not excerpt:
        return f"I found the {label} page on the BIEK website. You can view it here: {url}"
    summary = " ".join(excerpt)
    return f"{summary} You can view the {label} page here: {url}"


def build_biek_document_answer(question: str, row: dict[str, Any], content_doc: dict[str, Any]) -> str:
    meta = content_doc.get("metadata") or {}
    url = str(row.get("url") or content_doc.get("source_url") or "").strip()
    label = clean_answer_text(str(row.get("label") or content_doc.get("title") or "this BIEK document"))
    purpose = biek_document_purpose_snippet(content_doc)

    if not is_biek_detail_request(question):
        if purpose:
            return f"Yes. {purpose} {url}"
        return f"Yes. You can find the {label} here: {url}"

    parts: list[str] = []
    if purpose:
        parts.append(purpose)
    fee_details = meta.get("fee_details") or []
    if fee_details:
        parts.append(f"The available fee information includes {', '.join(fee_details[:2])}.")
    text = biek_content_text(content_doc)
    excerpt = extract_relevant_biek_excerpt(question, text, limit=2)
    if excerpt:
        snippet = " ".join(excerpt)
        snippet = re.sub(r"\s+", " ", snippet).strip()
        if len(snippet) > 240:
            snippet = snippet[:237].rstrip(" ,;:") + "..."
        if snippet not in parts:
            parts.append(snippet)
    if not parts:
        parts.append(f"You can find the {label} on the BIEK website.")
    return f"Yes. {' '.join(parts)} {url}"


def build_biek_container_answer(row: dict[str, Any], coverage_rows: list[dict[str, Any]], content_doc: dict[str, Any] | None) -> str:
    label = clean_answer_text(str(row.get("label") or "this section"))
    url = str(row.get("url") or "").strip()
    children = [
        clean_answer_text(str(candidate.get("label") or ""))
        for candidate in coverage_rows
        if str(candidate.get("parent") or "").strip().lower() == label.lower()
    ]
    children = [child for child in children if child][:6]
    text = biek_content_text(content_doc or {})
    if children:
        child_text = join_list(children[:5])
        if text and not is_weak_biek_content(text):
            excerpt = extract_relevant_biek_excerpt(label, text, limit=1)
            if excerpt:
                return f"{excerpt[0]} Under {label}, BIEK provides items such as {child_text}. The page is here: {url}"
        return f"Under {label}, BIEK provides items such as {child_text}. The page is here: {url}"
    if text and not is_weak_biek_content(text):
        excerpt = extract_relevant_biek_excerpt(label, text, limit=2)
        if excerpt:
            return f"{' '.join(excerpt)} The page is here: {url}"
    return f"I found the {label} section on the BIEK website. You can view it here: {url}"


def build_biek_content_backed_answer(question: str, row: dict[str, Any], coverage_rows: list[dict[str, Any]]) -> str:
    docs = biek_content_docs()
    primary_doc = find_biek_content_for_nav_row(row, docs)
    target_kind = classify_biek_nav_target_kind(row, primary_doc, coverage_rows)
    support_docs = search_biek_content_docs(question, docs, limit=2)

    if target_kind == "special_lookup":
        return build_biek_special_lookup_answer(question, row)
    if target_kind == "container_summary":
        return build_biek_container_answer(row, coverage_rows, primary_doc)

    chosen_doc = primary_doc
    if chosen_doc is None or (
        target_kind == "info_page"
        and is_weak_biek_content(biek_content_text(chosen_doc))
        and support_docs
    ):
        chosen_doc = support_docs[0]
    elif (
        target_kind == "info_page"
        and "who" in question.lower()
        and chosen_doc is not None
        and not has_person_like_biek_content(biek_content_text(chosen_doc))
    ):
        for support_doc in support_docs:
            if has_person_like_biek_content(biek_content_text(support_doc)):
                chosen_doc = support_doc
                break

    if chosen_doc is None:
        return build_biek_exact_match_answer(question, row)
    if target_kind == "document_summary":
        return build_biek_document_answer(question, row, chosen_doc)
    return build_biek_info_page_answer(question, row, chosen_doc)


def normalize_biek_coverage_query(question: str) -> str:
    text = normalize_biek_nav_text(question)
    filler_patterns = (
        r"^can you\b",
        r"^can u\b",
        r"^could you\b",
        r"^could u\b",
        r"^do you have\b",
        r"^do u have\b",
        r"^i want\b",
        r"^i need\b",
        r"^please\b",
        r"^show me\b",
        r"^give me\b",
        r"^tell me\b",
        r"^what is\b",
        r"^what are\b",
        r"^what does\b",
        r"^can you tell me about\b",
        r"^can u tell me about\b",
    )
    for pattern in filler_patterns:
        text = re.sub(pattern, "", text).strip()
    return text


def biek_coverage_query_tokens(question: str) -> list[str]:
    request_words = {
        "about",
        "available",
        "check",
        "direct",
        "download",
        "finding",
        "getting",
        "give",
        "help",
        "how",
        "know",
        "link",
        "page",
        "tell",
    }
    tokens = []
    for token in biek_nav_tokens(normalize_biek_coverage_query(question)):
        if token in request_words:
            continue
        tokens.append(token)
    return tokens


def biek_coverage_row_tokens(row: dict[str, Any]) -> list[str]:
    parts = [
        str(row.get("label") or ""),
        str(row.get("parent") or ""),
        str(row.get("handling_mode") or ""),
    ]
    return biek_nav_tokens(" ".join(parts))


def is_biek_public_item_style_question(question: str) -> bool:
    lowered = question.lower()
    return _contains_any(
        lowered,
        (
            "do you have",
            "do u have",
            "give me",
            "show me",
            "tell me about",
            "can you tell me about",
            "can u tell me about",
            "can you give me",
            "can u give me",
            "where can i find",
            "where do i find",
            "is there",
            "download",
            "link",
            "page",
        ),
    )


def biek_public_item_vocabulary(coverage_rows: list[dict[str, Any]]) -> set[str]:
    vocab: set[str] = set()
    for row in coverage_rows:
        if str(row.get("handling_mode") or "").strip().lower() == "exclude":
            continue
        vocab.update(biek_coverage_row_tokens(row))
    return vocab


def biek_coverage_candidate_rows(coverage_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in coverage_rows
        if str(row.get("handling_mode") or "").strip().lower() != "exclude"
        and str(row.get("label") or "").strip()
    ]


def score_biek_coverage_match(question: str, row: dict[str, Any]) -> dict[str, Any]:
    query_text = normalize_biek_coverage_query(question)
    question_text = normalize_biek_nav_text(query_text)
    query_tokens = set(biek_coverage_query_tokens(question))
    label = str(row.get("label") or "")
    parent = str(row.get("parent") or "")
    label_text = normalize_biek_nav_text(label)
    row_tokens = set(biek_coverage_row_tokens(row))
    label_tokens = set(biek_nav_tokens(label))
    parent_tokens = set(biek_nav_tokens(parent))
    handling_mode = str(row.get("handling_mode") or "").strip().lower()

    if not row_tokens:
        return {"score": 0.0, "query_coverage": 0.0, "row_coverage": 0.0, "overlap": []}

    overlap = query_tokens & row_tokens
    query_coverage = len(overlap) / max(len(query_tokens), 1) if query_tokens else 0.0
    row_coverage = len(overlap) / max(len(label_tokens or row_tokens), 1)
    label_coverage = len(query_tokens & label_tokens) / max(len(label_tokens), 1) if label_tokens else 0.0
    parent_coverage = len(query_tokens & parent_tokens) / max(len(parent_tokens), 1) if parent_tokens else 0.0

    score = (query_coverage * 0.62) + (row_coverage * 0.23) + (label_coverage * 0.15)

    if label_text and label_text in question_text:
        score += 0.95
    elif question_text and question_text in label_text:
        score += 0.28

    if label_coverage == 1.0 and label_tokens:
        score += 0.18
    if parent_coverage > 0:
        score += parent_coverage * 0.06
    if handling_mode == "special_lookup" and "result" in query_tokens:
        score += 0.18
    if handling_mode == "document_summary" and _contains_any(question.lower(), ("link", "download", "page")):
        score += 0.08

    return {
        "score": score,
        "query_coverage": query_coverage,
        "row_coverage": row_coverage,
        "label_coverage": label_coverage,
        "overlap": sorted(overlap),
    }


def classify_biek_coverage_match(question: str, coverage_rows: list[dict[str, Any]]) -> dict[str, Any]:
    query_tokens = set(biek_coverage_query_tokens(question))
    candidates = biek_coverage_candidate_rows(coverage_rows)
    coverage_vocab = biek_public_item_vocabulary(coverage_rows)
    vocab_overlap = query_tokens & coverage_vocab
    applies = bool(query_tokens and (vocab_overlap or is_biek_public_item_style_question(question)))

    if not applies:
        return {
            "applies": False,
            "match_type": "no_match",
            "query_tokens": sorted(query_tokens),
            "rows": [],
            "best_row": None,
        }

    scored_rows: list[dict[str, Any]] = []
    for row in candidates:
        scored_rows.append(
            {
                "row": row,
                **score_biek_coverage_match(question, row),
            }
        )

    scored_rows.sort(
        key=lambda item: (
            float(item.get("score") or 0.0),
            float(item.get("query_coverage") or 0.0),
            float(item.get("row_coverage") or 0.0),
        ),
        reverse=True,
    )
    best = scored_rows[0] if scored_rows else None
    if best is None:
        return {
            "applies": True,
            "match_type": "no_match",
            "query_tokens": sorted(query_tokens),
            "rows": [],
            "best_row": None,
        }

    best_row = best["row"]
    label_text = normalize_biek_nav_text(str(best_row.get("label") or ""))
    query_text = normalize_biek_nav_text(normalize_biek_coverage_query(question))
    generic_overlap_tokens = {
        "download",
        "form",
        "link",
        "notification",
        "page",
        "paper",
        "result",
    }
    meaningful_overlap = {
        token
        for token in (best.get("overlap") or [])
        if token not in generic_overlap_tokens
    }
    exact_match = (
        (label_text and label_text in query_text)
        or (
            float(best.get("query_coverage") or 0.0) >= 0.85
            and float(best.get("row_coverage") or 0.0) >= 0.72
            and float(best.get("score") or 0.0) >= 0.85
            and bool(meaningful_overlap)
        )
    )
    close_match = (
        not exact_match
        and float(best.get("query_coverage") or 0.0) >= 0.5
        and float(best.get("score") or 0.0) >= 0.45
        and bool(meaningful_overlap)
    )

    related_rows = []
    if exact_match:
        related_rows = [best]
        match_type = "exact_match"
    elif close_match:
        best_score = float(best.get("score") or 0.0)
        for item in scored_rows:
            if len(related_rows) >= 3:
                break
            if float(item.get("query_coverage") or 0.0) < 0.5:
                continue
            if float(item.get("score") or 0.0) < max(0.45, best_score * 0.72):
                continue
            related_rows.append(item)
        match_type = "close_match"
    else:
        match_type = "no_match"

    return {
        "applies": True,
        "match_type": match_type,
        "query_tokens": sorted(query_tokens),
        "rows": related_rows,
        "best_row": best_row if match_type != "no_match" else None,
        "best_score": float(best.get("score") or 0.0),
    }


def build_biek_special_lookup_answer(question: str, row: dict[str, Any]) -> str:
    label = clean_answer_text(str(row.get("label") or "this BIEK resource"))
    url = str(row.get("url") or "").strip()
    low = question.lower()
    if "result" in low:
        if url:
            return (
                f"BIEK has a {label} page here: {url}. "
                "For roll-number-specific results, use the BIEK results lookup flow."
            )
        return f"BIEK has a {label} page. For roll-number-specific results, use the BIEK results lookup flow."
    if url:
        return f"BIEK has a {label} page here: {url}"
    return f"BIEK has a {label} page."


def build_biek_exact_match_answer(question: str, row: dict[str, Any]) -> str:
    label = clean_answer_text(str(row.get("label") or "this BIEK resource"))
    url = str(row.get("url") or "").strip()
    handling_mode = str(row.get("handling_mode") or "").strip().lower()
    question_low = question.lower().strip()
    verb = "are" if label.lower().endswith("s") or " forms" in label.lower() else "is"

    if handling_mode == "special_lookup":
        return build_biek_special_lookup_answer(question, row)
    if _contains_any(question_low, ("where", "download", "link")):
        return f"You can find the {label} here: {url}"
    if question_low.startswith(("do you have", "can you help me find", "can u help me find", "is ")):
        return f"Yes, the {label} {verb} available here: {url}"
    return f"The {label} {verb} available here: {url}"


def build_biek_close_match_answer(question: str, matched_rows: list[dict[str, Any]]) -> str:
    if not matched_rows:
        return "This detail is not clearly confirmed in the available BIEK information. Please verify it on the official BIEK website."

    rows = [item["row"] for item in matched_rows]
    labels = [clean_answer_text(str(row.get("label") or "")) for row in rows if str(row.get("label") or "").strip()]
    labels = unique_preserve_order([label for label in labels if label])[:3]
    urls = unique_preserve_order([str(row.get("url") or "").strip() for row in rows if str(row.get("url") or "").strip()])
    handling_modes = {
        str(row.get("handling_mode") or "").strip().lower()
        for row in rows
    }
    requested = " ".join(biek_coverage_query_tokens(question)).strip() or "that item"
    lowered_labels = [label.lower() for label in labels]

    if "admit" in requested and "card" in requested and labels and len(urls) == 1:
        manual_present = any("manual admit card" in label for label in lowered_labels)
        computerized_present = any("computerized admit card" in label for label in lowered_labels)
        if manual_present and computerized_present:
            return (
                "I could not clearly confirm a direct BIEK admit card download page in the available information. "
                "However, BIEK does provide duplicate admit card forms, including Duplicate Manual Admit Card "
                "and Duplicate Computerized Admit Card. Both are covered through this PDF: "
                f"{urls[0]}"
            )

    if handling_modes == {"special_lookup"} and "result" in " ".join(biek_coverage_query_tokens(question)):
        link_text = f" You can start here: {urls[0]}" if urls else ""
        return (
            f"I could not clearly confirm that exact BIEK result item in the available information. "
            f"However, related BIEK result sections include {join_list(labels)}.{link_text} "
            "For roll-number-specific results, use the BIEK results lookup flow."
        )

    if labels and urls:
        if len(urls) == 1:
            return (
                f"I could not clearly confirm a direct BIEK {requested} item in the available information. "
                f"However, related BIEK items include {join_list(labels)}. You can access them here: {urls[0]}"
            )
        return (
            f"I could not clearly confirm a direct BIEK {requested} item in the available information. "
            f"However, related BIEK items include {join_list(labels)}. "
            f"You can check these links: {', '.join(urls[:2])}"
        )
    if labels:
        return (
            f"I could not clearly confirm a direct BIEK {requested} item in the available information. "
            f"However, related BIEK items include {join_list(labels)}."
        )
    return "This detail is not clearly confirmed in the available BIEK information. Please verify it on the official BIEK website."


def build_biek_no_match_answer() -> str:
    return "This detail is not clearly confirmed in the available BIEK information. Please verify it on the official BIEK website."


def is_biek_nav_section_question(question: str) -> bool:
    lowered = question.lower()
    return _contains_any(
        lowered,
        (
            "what can i find under",
            "what can i find in",
            "what is under",
            "what's under",
            "whats under",
            "what is available under",
            "what's available under",
            "whats available under",
        ),
    )


def match_biek_nav_section(question: str, coverage_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    question_text = normalize_biek_nav_text(question)
    top_level_rows = [row for row in coverage_rows if row.get("level") == 0]

    best_row: dict[str, Any] | None = None
    best_score = 0.0
    for row in top_level_rows:
        label = str(row.get("label") or "")
        label_text = normalize_biek_nav_text(label)
        if not label_text:
            continue

        label_tokens = set(biek_nav_tokens(label))
        question_tokens = set(biek_nav_tokens(question))
        overlap = question_tokens & label_tokens
        label_overlap_ratio = len(overlap) / max(len(label_tokens), 1) if label_tokens else 0.0
        score = label_overlap_ratio
        if label_text in question_text:
            score += 0.8
        if score > best_score:
            best_score = score
            best_row = row

    if best_row is None:
        return None
    if best_score >= 0.8:
        return best_row
    if best_score >= 0.5 and is_biek_nav_section_question(question):
        return best_row
    return None


def build_biek_nav_section_summary(section_label: str, coverage_rows: list[dict[str, Any]]) -> str:
    children = [
        row for row in coverage_rows
        if str(row.get("parent") or "").strip().lower() == section_label.strip().lower()
    ]
    if not children:
        return f"I found the {section_label} section on the BIEK website, but I could not clearly list its items from the current coverage map."

    labels = [clean_answer_text(str(row.get("label") or "")) for row in children if str(row.get("label") or "").strip()]
    labels = [label for label in labels if label]
    if not labels:
        return f"I found the {section_label} section on the BIEK website, but I could not clearly list its items from the current coverage map."

    if len(labels) == 1:
        joined = labels[0]
    elif len(labels) == 2:
        joined = f"{labels[0]} and {labels[1]}"
    else:
        joined = ", ".join(labels[:-1]) + f", and {labels[-1]}"

    return f"Under {section_label}, BIEK lists {joined}."


def is_biek_latest_style_question(question: str, profile: dict[str, Any]) -> bool:
    intent = profile.get("biek_time_intent") or parse_biek_time_intent(question)
    if intent.get("explicit_year") is not None:
        return False
    return bool(intent.get("has_recency_intent")) or _contains_any(
        question.lower(),
        (
            "last date",
            "fee schedule",
            "date sheet",
            "datesheet",
            "notification",
            "notifications",
        ),
    )


def is_biek_datesheet_question(question: str) -> bool:
    return _contains_any(question.lower(), ("date sheet", "datesheet", "exam schedule"))


def is_biek_notification_question(question: str) -> bool:
    return _contains_any(question.lower(), ("notification", "notifications", "notice", "notices"))


def is_biek_model_paper_question(question: str) -> bool:
    low = question.lower()
    return "model paper" in low or ("model" in low and "paper" in low)


def is_biek_esheet_question(question: str) -> bool:
    return _contains_any(
        question.lower(),
        (
            "e-sheet",
            "esheet",
            "omr sheet",
            "answer sheet sample",
            "e-marking sample",
        ),
    )


def is_biek_timeless_form_question(question: str) -> bool:
    return _contains_any(
        question.lower(),
        (
            "migration form",
            "certificate form",
            "verification form",
            "duplicate marksheet",
            "duplicate document",
            "provisional certificate",
            "where can i download",
        ),
    )


def is_biek_identity_question(question: str) -> bool:
    return _contains_any(
        question.lower(),
        (
            "chairman",
            "board members",
            "board member",
            "committee",
            "committees",
            "who runs",
            "who heads",
            "who leads",
            "who is on the board",
            "who is on biek board",
            "who runs biek",
        ),
    )


def build_biek_identity_safe_answer() -> str:
    return "This detail is not clearly confirmed in the available BIEK information. Please verify it on the official BIEK website."


def has_strong_biek_identity_evidence(hits: list[dict[str, Any]]) -> bool:
    if not hits:
        return False

    positive_tokens = (
        "chairman",
        "board members",
        "bmembers",
        "committee",
        "committees",
    )
    mismatch_tokens = (
        "tender",
        "procurement",
        "contract award",
        "bid evaluation",
        "evaluation report",
        "result gazette",
        "result declaration",
        "notification",
        "supply of",
        "secretary board",
        "pay order",
        "bidding document",
    )

    for hit in hits[:4]:
        title = str(hit.get("title") or "").lower()
        source_url = str(hit.get("source_url") or hit.get("url") or "").lower()
        doc_id = str(hit.get("doc_id") or "").lower()
        document_type = str(hit.get("document_type") or "").lower()
        section = str(hit.get("section") or "").lower()
        identity_surface = " ".join(
            [
                title,
                source_url,
                doc_id,
                document_type,
                section,
            ]
        )
        blob = identity_surface + " " + str(hit.get("text") or "")[:600].lower()

        if any(token in blob for token in mismatch_tokens):
            continue

        if any(token in identity_surface for token in positive_tokens):
            return True

    return False


def biek_question_explicit_year(question: str, profile: dict[str, Any]) -> int | None:
    intent = profile.get("biek_time_intent") or parse_biek_time_intent(question)
    value = intent.get("explicit_year")
    return int(value) if isinstance(value, int) else None


def biek_relevant_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [hit for hit in hits if is_biek_time_sensitive_hit(hit)]


def latest_supported_biek_year(hits: list[dict[str, Any]]) -> int | None:
    years = [extract_biek_hit_year(hit) for hit in hits]
    years = [year for year in years if year is not None]
    return max(years) if years else None


def is_biek_evidence_stale_for_question(
    question: str,
    hits: list[dict[str, Any]],
    profile: dict[str, Any],
) -> bool:
    if biek_question_explicit_year(question, profile) is not None:
        return False
    if is_biek_timeless_form_question(question):
        return False
    if not is_biek_latest_style_question(question, profile):
        return False

    relevant_hits = biek_relevant_hits(hits) or hits
    latest_year = latest_supported_biek_year(relevant_hits)
    if latest_year is None:
        return True
    return latest_year < min(biek_supported_current_years())


def biek_primary_link(hit: dict[str, Any]) -> str:
    url = str(hit.get("url") or hit.get("source_url") or "").strip()
    return url


def biek_hits_for_year(hits: list[dict[str, Any]], year: int) -> list[dict[str, Any]]:
    return [hit for hit in hits if extract_biek_hit_year(hit) == year]


def build_biek_stale_safe_answer(
    question: str,
    hits: list[dict[str, Any]],
    profile: dict[str, Any],
) -> str:
    relevant_hits = biek_relevant_hits(hits) or hits
    latest_year = latest_supported_biek_year(relevant_hits)
    if latest_year is None:
        return (
            "The latest available BIEK information for this topic is not clearly confirmed in the current evidence."
        )
    return (
        f"I could only find older BIEK information for this topic, mainly from {latest_year}. "
        "The latest detail is not clearly confirmed in the available 2025-2026 BIEK information."
    )


def build_biek_datesheet_answer(
    question: str,
    hits: list[dict[str, Any]],
    profile: dict[str, Any],
) -> str:
    explicit_year = biek_question_explicit_year(question, profile)
    datesheet_hits = [
        hit
        for hit in hits
        if str(hit.get("document_type") or "").lower() == "datesheet"
        or str(hit.get("section") or "").lower() == "datesheet"
    ]
    if explicit_year is not None:
        matching = biek_hits_for_year(datesheet_hits, explicit_year)
        if matching:
            link = biek_primary_link(matching[0])
            if link:
                return f"I found a BIEK date sheet for {explicit_year} here: {link}"
            return f"I found a BIEK date sheet for {explicit_year}."
        return ""

    current_hits = []
    for year in biek_supported_current_years():
        current_hits = biek_hits_for_year(datesheet_hits, year)
        if current_hits:
            link = biek_primary_link(current_hits[0])
            if link:
                return f"The latest available BIEK date sheet I found is for {year}. You can view it here: {link}"
            return f"The latest available BIEK date sheet I found is for {year}."
    return ""


def summarize_biek_notification_titles(hits: list[dict[str, Any]]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        title = clean_answer_text(str(hit.get("title") or ""))
        if not title:
            continue
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(title)
    return cleaned[:3]


def build_biek_notification_summary_answer(
    question: str,
    hits: list[dict[str, Any]],
    profile: dict[str, Any],
) -> str:
    explicit_year = biek_question_explicit_year(question, profile)
    notification_hits = [
        hit
        for hit in hits
        if str(hit.get("document_type") or "").lower() == "notification"
        or str(hit.get("section") or "").lower() == "notifications"
    ]
    if explicit_year is not None:
        notification_hits = biek_hits_for_year(notification_hits, explicit_year)
    else:
        preferred: list[dict[str, Any]] = []
        for year in biek_supported_current_years():
            preferred.extend(biek_hits_for_year(notification_hits, year))
        if preferred:
            notification_hits = preferred

    if not notification_hits:
        return ""

    years = []
    for hit in notification_hits:
        year = extract_biek_hit_year(hit)
        if year is not None and year not in years:
            years.append(year)
    titles = summarize_biek_notification_titles(notification_hits)
    if not titles:
        return ""

    if explicit_year is not None:
        return f"For {explicit_year}, I found BIEK notifications including {', '.join(titles[:2])}."

    if years:
        if len(years) == 1:
            year_text = str(years[0])
        else:
            year_text = " and ".join(str(year) for year in years[:2])
        return (
            f"I found BIEK notifications for this topic from {year_text}, including "
            f"{', '.join(titles[:2])}."
        )
    return f"I found BIEK notifications including {', '.join(titles[:2])}."


def build_biek_model_paper_answer(
    question: str,
    hits: list[dict[str, Any]],
    profile: dict[str, Any],
) -> str:
    model_hits = [
        hit
        for hit in hits
        if str(hit.get("document_type") or "").lower() == "model_paper"
        or str(hit.get("section") or "").lower() == "model_paper"
    ]
    if not model_hits:
        return ""

    explicit_year = biek_question_explicit_year(question, profile)
    if explicit_year is not None:
        matching = biek_hits_for_year(model_hits, explicit_year)
        if matching:
            link = biek_primary_link(matching[0])
            if link:
                return f"Yes. The BIEK model paper for {explicit_year} is available here: {link}"
            return f"Yes. I found a BIEK model paper entry for {explicit_year}."
        return ""

    preferred = sorted(
        model_hits,
        key=lambda hit: (
            extract_biek_hit_year(hit) or 0,
            float(hit.get("score") or 0.0),
        ),
        reverse=True,
    )
    best = preferred[0]
    year = extract_biek_hit_year(best)
    link = biek_primary_link(best)
    if link and year is not None:
        return f"Yes. The latest BIEK model paper I found is for {year}: {link}"
    if link:
        return f"Yes. A BIEK model paper is available here: {link}"
    if year is not None:
        return f"Yes. I found a BIEK model paper entry for {year}."
    return "Yes. I found a BIEK model paper entry."


def build_biek_esheet_answer(
    question: str,
    hits: list[dict[str, Any]],
    profile: dict[str, Any],
) -> str:
    esheet_hits = [
        hit
        for hit in hits
        if str(hit.get("document_type") or "").lower() == "exam_material"
        or "e-sheet" in str(hit.get("title") or "").lower()
        or "e-sheet" in str(hit.get("source_url") or "").lower()
        or "omr" in str(hit.get("title") or "").lower()
    ]
    if not esheet_hits:
        return ""

    preferred = sorted(
        esheet_hits,
        key=lambda hit: (
            extract_biek_hit_year(hit) or 0,
            float(hit.get("score") or 0.0),
        ),
        reverse=True,
    )
    links: list[str] = []
    for hit in preferred:
        link = biek_primary_link(hit)
        if link and link not in links:
            links.append(link)
        if len(links) >= 2:
            break

    year = extract_biek_hit_year(preferred[0])
    if len(links) >= 2 and year is not None:
        return (
            f"Yes. BIEK has E-Sheet sample files for {year}. "
            f"You can use these links: {links[0]} and {links[1]}"
        )
    if links:
        return f"Yes. BIEK has an E-Sheet sample file here: {links[0]}"
    if year is not None:
        return f"Yes. I found BIEK E-Sheet sample material for {year}."
    return "Yes. I found BIEK E-Sheet sample material."


def is_biek_help_question(question: str) -> bool:
    return _contains_any(
        question.lower(),
        (
            "what can you help me with",
            "what can u help me with",
            "can you tell me what you can help me with",
            "can u tell me what u can help me with",
            "what can you do",
            "what do you do",
            "how can you help",
            "how can u help",
        ),
    )


def is_biek_generic_help_question(question: str) -> bool:
    low = question.lower()
    return is_biek_help_question(question) or _contains_any(
        low,
        (
            "what kind of information can i ask",
            "what kind of information can i ask about",
            "can you help me find forms",
            "can u help me find forms",
            "can you help me find model papers",
            "can u help me find model papers",
            "can you help me check biek results",
            "can u help me check biek results",
            "what can i ask about",
        ),
    )


def is_biek_contact_question(question: str) -> bool:
    low = question.lower()
    return "biek" in low and _contains_any(
        low,
        (
            "contact",
            "contact number",
            "phone",
            "email address",
            "email",
            "how can i contact",
            "how do i contact",
            "reach",
        ),
    )


def is_biek_form_or_scheme_question(question: str) -> bool:
    return _contains_any(
        question.lower(),
        (
            "certificate form",
            "verification certificate",
            "verification marksheet",
            "verification migration",
            "verification form",
            "provisional certification form",
            "migration form",
            "scrutiny form",
            "duplicate marksheet",
            "duplicate enrolment card",
            "duplicate computerized admit card",
            "duplicate registration card",
            "duplicate manual admit card",
            "proforma of special chance",
            "improvement of division",
            "registration of all groups",
            "examination forms",
            "permission forms",
            "scheme of studies",
            "scheme and model paper",
        ),
    )


def build_biek_help_answer() -> str:
    return (
        "I can help with BIEK notifications, forms, datesheets, results-related information, "
        "fee vouchers, and affiliated college details based on the BIEK knowledge base."
    )


def build_biek_better_help_answer(question: str) -> str:
    low = question.lower()
    if "forms" in low:
        link = biek_nav_row_link("Download Forms")
        return (
            "Yes. I can help you find BIEK forms and downloads such as certificate, migration, "
            "verification, duplicate, registration, examination, and permission forms."
            + (f" Start here: {link}" if link else "")
        )
    if "model paper" in low or "model papers" in low:
        link = biek_nav_row_link("Model Paper 2026") or biek_nav_row_link("Scheme & Model Paper 2023")
        return (
            "Yes. I can help you find BIEK model papers, scheme and model paper resources, "
            "and related examination materials from the website."
            + (f" A useful starting link is: {link}" if link else "")
        )
    if "result" in low or "results" in low:
        link = biek_nav_row_link("Recent Result Declaration")
        return (
            "Yes. I can help with BIEK results-related questions, including guiding you to the "
            "results area or checking supported roll-number-based result lookups."
            + (f" Results page: {link}" if link else "")
        )
    if "what kind of information" in low or "what can i ask about" in low or is_biek_help_question(question):
        link = biek_nav_row_link("Download Forms") or biek_nav_row_link("Date Sheet")
        return (
            "I can help with BIEK forms, date sheets, notifications, model papers, results-related questions, "
            "authorized banks, and affiliated-college information."
            + (f" A good place to start is: {link}" if link else "")
        )
    return (
        "I can help with BIEK forms, date sheets, notifications, model papers, results-related questions, "
        "authorized banks, and affiliated-college information."
    )


def build_biek_contact_safe_answer(hits: list[dict[str, Any]]) -> str:
    contact_row = find_biek_nav_row_by_label("Contact us")
    if contact_row is not None:
        docs = biek_content_docs()
        contact_doc = resolve_biek_nav_primary_doc(contact_row, docs)
        if contact_doc is not None:
            text = biek_content_text(contact_doc)
            phones, emails = extract_biek_contact_details(text)
            url = str(contact_row.get("url") or contact_doc.get("source_url") or "").strip()
            if phones and emails:
                return (
                    f"BIEK contact details include phone numbers {', '.join(phones[:4])} "
                    f"and email {', '.join(emails[:2])}. You can view the contact page here: {url}"
                )
            if phones:
                return (
                    f"BIEK contact numbers include {', '.join(phones[:4])}. "
                    f"You can view the contact page here: {url}"
                )
            if emails:
                return (
                    f"The available BIEK contact email is {', '.join(emails[:2])}. "
                    f"You can view the contact page here: {url}"
                )

    combined_text = "\n".join(normalized_hit_text(hit) for hit in hits)
    phones = []
    for match in re.findall(r"\b\d{7,}\b", combined_text):
        if match not in phones:
            phones.append(match)
    email_match = EMAIL_RE.search(combined_text)

    if phones and email_match:
        return (
            f"BIEK contact details in the available information include phone numbers {', '.join(phones[:4])} "
            f"and email {email_match.group(0)}."
        )
    if phones:
        return f"BIEK contact numbers in the available information include {', '.join(phones[:4])}."
    if email_match:
        return f"The available BIEK contact email is {email_match.group(0)}."
    return "This detail is not clearly confirmed in the available BIEK information. Please verify it on the official BIEK website."


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


def build_pricing_unknown_answer(hits: list[dict[str, Any]]) -> str:
    contact_answer = build_contact_answer(hits)
    if contact_answer:
        return (
            "Public pricing is not confirmed in the available information. "
            f"For current pricing, contact Synapse Tech directly. {contact_answer}"
        )
    return (
        "Public pricing is not confirmed in the available information. "
        "Please contact Synapse Tech directly for current pricing."
    )


def build_high_risk_unknown_answer(question: str, hits: list[dict[str, Any]]) -> str:
    category = high_risk_category(question)
    contact_answer = build_contact_answer(hits)

    if category == "pricing":
        return build_pricing_unknown_answer(hits)
    if category == "guarantee":
        return (
            "A fixed outcome or percentage improvement is not confirmed in the available information. "
            "Synapse Tech can discuss expected impact for your scope, but the assistant should not guarantee results."
        )
    if category == "sla":
        return (
            "Public SLA numbers or exact uptime commitments are not confirmed in the available information. "
            "Please verify official SLA terms with Synapse Tech."
        )
    if category == "roadmap":
        return (
            "Roadmap dates, launch dates, or release milestones are not confirmed in the available information. "
            "Please verify timelines directly with Synapse Tech."
        )
    if category == "certification_compliance":
        return (
            "Security certifications or compliance standards are not explicitly confirmed in the available information. "
            "Please verify official certification or compliance claims with Synapse Tech."
        )
    if category == "integration":
        if is_general_integration_list_question(question):
            return (
                "A single exhaustive public integration list is not clearly confirmed in the available information. "
                "Available pages mention tools or platforms in specific contexts, so please verify supported integrations with Synapse Tech."
            )
        return (
            "That specific integration or public API documentation detail is not confirmed in the available information. "
            "Please verify supported integrations or API documentation with Synapse Tech."
        )
    if category == "technical_detail":
        return (
            "Exact model architecture or internal technical implementation details are not confirmed in the available information. "
            "Please verify technical specifications directly with Synapse Tech."
        )
    if category == "customer_reference":
        return (
            "Public customer names, logos, references, testimonials, or case studies are not confirmed in the available information. "
            "Please verify approved customer references directly with Synapse Tech."
        )
    if category == "benchmark":
        return (
            "Exact benchmark numbers or performance results are not confirmed in the available information. "
            "Please verify benchmark data directly with Synapse Tech."
        )
    if category == "legal":
        return (
            "Legal or contractual terms are not confirmed in the available information. "
            "Please obtain official terms directly from Synapse Tech."
        )
    if contact_answer:
        return f"{DEFAULT_FALLBACK_RESPONSE} {contact_answer}"
    return DEFAULT_FALLBACK_RESPONSE


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
    configured = RUNTIME_POLICIES.get("voice_agent", {})
    if isinstance(configured, dict) and configured.get("answer"):
        return str(configured["answer"])
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
        return RUNTIME_POLICIES.get("custom_software", {}).get(
            "answer",
            DEFAULT_RUNTIME_POLICIES["custom_software"]["answer"],
        )
    return ""


def build_private_deployment_answer(hits: list[dict[str, Any]]) -> str:
    blob = " ".join(normalized_hit_text(hit) for hit in hits).lower()
    if "opira" in blob and "offline" in blob and ("private" in blob or "on-prem" in blob):
        return RUNTIME_POLICIES.get("private_infrastructure", {}).get(
            "answer",
            DEFAULT_RUNTIME_POLICIES["private_infrastructure"]["answer"],
        )
    return ""


def build_workflow_automation_answer(hits: list[dict[str, Any]]) -> str:
    blob = " ".join(normalized_hit_text(hit) for hit in hits).lower()
    if "workflow automation" in blob and all(term in blob for term in ("n8n", "airbyte", "airtable")):
        return RUNTIME_POLICIES.get("workflow_automation", {}).get(
            "answer",
            DEFAULT_RUNTIME_POLICIES["workflow_automation"]["answer"],
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
    aliases = profile.get("product_aliases") or []
    if aliases:
        return PRODUCT_NAME_NORMALIZATION.get(str(aliases[0]).lower(), title_prefix(str(aliases[0])))
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
    card = PRODUCT_CARDS.get(title)
    if isinstance(card, dict) and card.get("detail_answer"):
        return str(card["detail_answer"])

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

    title = ""
    for alias in profile.get("product_aliases") or []:
        expected = PRODUCT_NAME_NORMALIZATION.get(str(alias).lower())
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
    if is_biek_domain(profile) and is_biek_form_or_scheme_question(question):
        return "generic"

    if is_high_risk_question(question):
        return "high_risk_unknown"

    if ops_guidance_intent(question):
        return "ops_guidance"

    if is_purchase_or_get_started_question(question):
        return "purchase"
    if is_direct_contact_question(question):
        return "contact"
    if is_private_deployment_question(question):
        return "private_infrastructure"
    if profile.get("single_product_recommendation") or (
        is_single_product_recommendation_question(question)
        and profile.get("metadata_routes")
    ):
        return "service_guidance"
    if is_custom_software_question(question):
        return "service_guidance"
    if is_workflow_automation_question(question):
        return "service_guidance"
    if is_voice_agent_question(question):
        return "voice_agent"
    if profile.get("company_overview") or is_company_overview_or_synopsis_question(question):
        return "company_overview"

    model_intent = profile.get("model_intent") or "generic"
    model_risk = profile.get("model_risk") or "normal"
    if model_risk == "high_risk" or model_intent == "high_risk_unknown":
        return "high_risk_unknown"
    if model_intent in {
        "product_catalog",
        "product_detail",
        "service_catalog",
        "service_guidance",
        "private_infrastructure",
        "purchase",
        "industry",
        "company_overview",
    }:
        return model_intent
    if model_intent == "contact" and is_direct_contact_question(question):
        return "contact"

    intent = profile.get("intent")
    if intent == "contact" and is_direct_contact_question(question):
        return "contact"
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
        return build_high_risk_unknown_answer(question, hits)

    if policy == "ops_guidance":
        return build_ops_guidance_answer(ops_guidance_intent(question))

    if policy == "contact":
        return build_contact_answer(hits)

    if policy == "purchase":
        return build_purchase_answer(question, hits, profile)

    if policy == "product_catalog":
        if is_customer_support_product_question(question):
            return ""
        if wants_product_descriptions(question):
            return ""
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

    if policy == "company_overview":
        return build_instruction_fallback(question)

    if policy == "voice_agent":
        return build_voice_agent_answer(hits)

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
    classifier_model: str | None = None,
    page_types: set[str],
    top_k: int,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    total_started = time.perf_counter()
    timings: dict[str, float] = {}
    stage_started = time.perf_counter()
    index, rows, manifest = load_kb_resources(index_path, metadata_path, manifest_path)
    timings["kb_load_s"] = round(time.perf_counter() - stage_started, 4)
    model = embed_model or manifest.get("embedding_model")
    if not model:
        raise ValueError(
            "No embedding model available. Use --embed-model or ensure the manifest exists."
        )

    planner_model = rewrite_model or classifier_model or DEFAULT_REWRITE_MODEL
    cache_key = (
        question.strip(),
        str(index_path.resolve()),
        str(metadata_path.resolve()),
        str(manifest_path.resolve()),
        model,
        planner_model,
        rewrite_model or "",
        classifier_model or "",
        RAG_COMBINED_PLANNING,
        tuple(sorted(page_types)),
        top_k,
    )
    cached_retrieval = _RETRIEVAL_CACHE.get(cache_key)
    if cached_retrieval is not None:
        hits, cached_model, cached_profile = cached_retrieval
        profile = copy.deepcopy(cached_profile)
        cached_timings = dict(profile.get("timings") or {})
        cached_timings["cache_hit"] = True
        cached_timings["retrieval_total_s"] = round(time.perf_counter() - total_started, 4)
        profile["timings"] = cached_timings
        return copy.deepcopy(hits), cached_model, profile

    profile = query_kb.query_profile(question)
    profile["knowledge_domain"] = infer_retrieval_domain(index_path, metadata_path, manifest_path)
    if is_biek_knowledge_domain(profile["knowledge_domain"]):
        profile["biek_time_intent"] = parse_biek_time_intent(question)
    stage_started = time.perf_counter()
    plan_cache_key = (question.strip(), planner_model, RAG_COMBINED_PLANNING)
    cached_plan = _QUERY_PLAN_CACHE.get(plan_cache_key)
    if cached_plan is not None:
        rewrite, classification = copy.deepcopy(cached_plan)
        timings["planning_cache_hit"] = True
    else:
        if RAG_COMBINED_PLANNING:
            rewrite, classification = plan_query_with_model(
                question=question,
                ollama_url=ollama_url,
                model=planner_model,
                knowledge_domain=profile.get("knowledge_domain") or "synapse",
            )
        else:
            rewrite = rewrite_query_with_model(
                question=question,
                ollama_url=ollama_url,
                model=rewrite_model or DEFAULT_REWRITE_MODEL,
                knowledge_domain=profile.get("knowledge_domain") or "synapse",
            )
            search_query_for_classification = (
                rewrite.get("search_query") or profile.get("search_query") or question
            )
            classification = classify_question_with_model(
                question=question,
                search_query=search_query_for_classification,
                ollama_url=ollama_url,
                model=classifier_model or DEFAULT_CLASSIFIER_MODEL,
                knowledge_domain=profile.get("knowledge_domain") or "synapse",
            )
        _QUERY_PLAN_CACHE[plan_cache_key] = (
            copy.deepcopy(rewrite),
            copy.deepcopy(classification),
        )
    timings["planning_s"] = round(time.perf_counter() - stage_started, 4)
    profile["rewrite"] = rewrite
    profile["intent_hint"] = rewrite.get("intent_hint")
    search_query = rewrite.get("search_query") or profile.get("search_query") or question
    apply_model_classification_to_profile(profile, classification)
    profile["company_overview"] = is_company_overview_or_synopsis_question(question)
    profile["product_comparison"] = is_product_comparison_question(question)
    profile["single_product_recommendation"] = is_single_product_recommendation_question(question)
    profile["comparison_products"] = product_names_in_question(question)
    profile["purchase_intent"] = is_purchase_or_get_started_question(question)
    profile["pricing_intent"] = is_pricing_or_quote_question(question)
    profile["metadata_routes"] = metadata_routes_for_question(question)
    profile["answer_policy"] = determine_answer_policy(question, profile)
    expansion = route_query_expansion(profile)
    if expansion:
        existing_tokens = set(query_kb.tokenize(search_query))
        expansion_tokens = [
            token for token in expansion.split() if token.lower() not in existing_tokens
        ]
        if expansion_tokens:
            search_query = f"{search_query} {' '.join(expansion_tokens)}"
    profile["routed_search_query"] = search_query
    stage_started = time.perf_counter()
    embedding_cache_key = (ollama_url, model, search_query)
    cached_query_vector = _EMBEDDING_CACHE.get(embedding_cache_key)
    if cached_query_vector is not None:
        query_vector = cached_query_vector.copy()
        timings["embedding_cache_hit"] = True
    else:
        query_vector = query_kb.embed_query(ollama_url, model, search_query)
        _EMBEDDING_CACHE[embedding_cache_key] = query_vector.copy()
    timings["embedding_s"] = round(time.perf_counter() - stage_started, 4)
    retrieval_top_k = top_k
    if profile.get("single_product_recommendation"):
        retrieval_top_k = max(top_k, 10)
    elif profile.get("answer_policy") in {"product_catalog", "service_catalog", "industry"}:
        retrieval_top_k = max(top_k, 10)
    elif profile.get("intent") in SUMMARY_INTENT_CONFIG:
        retrieval_top_k = max(top_k, 10)
    elif profile.get("company_overview"):
        retrieval_top_k = max(top_k, 8)
    elif profile.get("entity_terms"):
        retrieval_top_k = max(top_k, 8)
    stage_started = time.perf_counter()
    hits = query_kb.search_index(
        index,
        query_vector,
        rows,
        page_types,
        retrieval_top_k,
        profile,
    )
    timings["faiss_search_s"] = round(time.perf_counter() - stage_started, 4)
    raw_ranked_hits = sorted(
        hits,
        key=lambda hit: float(hit.get("raw_score") or 0.0),
        reverse=True,
    )
    top_similarity = (
        float(raw_ranked_hits[0].get("raw_score") or 0.0)
        if raw_ranked_hits else 0.0
    )
    second_similarity = (
        float(raw_ranked_hits[1].get("raw_score") or 0.0)
        if len(raw_ranked_hits) > 1 else 0.0
    )
    profile["retrieval_confidence"] = {
        "top_similarity": round(top_similarity, 4),
        "second_similarity": round(second_similarity, 4),
        "score_gap": round(top_similarity - second_similarity, 4),
        "top_rerank_score": round(float(hits[0].get("score") or 0.0), 4) if hits else 0.0,
        "top_doc_ids": [str(hit.get("doc_id") or "") for hit in hits[:3]],
        "top_chunk_ids": [str(hit.get("chunk_id") or "") for hit in hits[:3]],
    }
    log_kb_debug(
        {
            "question": question,
            "stage": "retrieval",
            "search_query": search_query,
            "routed_search_query": profile.get("routed_search_query"),
            "intent_hint": profile.get("intent_hint"),
            "model_classification": profile.get("model_classification"),
            "answer_policy": profile.get("answer_policy"),
            "retrieval_confidence": profile.get("retrieval_confidence"),
            "metadata_routes": profile.get("metadata_routes", []),
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
    stage_started = time.perf_counter()
    if profile.get("company_overview"):
        hits = augment_hits_with_about_us(hits, rows, max_chunks=4)
    if is_product_comparison_question(question):
        hits = augment_hits_for_product_comparison(
            hits,
            rows,
            product_names_in_question(question),
        )
    answer_policy = profile.get("answer_policy")
    if answer_policy in {"purchase", "contact"} or is_pricing_or_quote_question(question):
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
    if answer_policy == "private_infrastructure" or is_private_deployment_question(question):
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
    metadata_doc_ids = {
        doc_id
        for route in profile.get("metadata_routes", [])
        for doc_id in route.get("doc_ids", [])
    }
    if metadata_doc_ids:
        hits = augment_hits_with_matching_rows(
            hits,
            rows,
            matches=lambda row: row.get("doc_id") in metadata_doc_ids,
            max_chunks=4,
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
    hits = augment_hits_with_context_pack(
        hits,
        rows,
        question=question,
        profile=profile,
    )
    pre_cleanup_count = len(hits)
    hits = rerank_hits_for_generation(question, hits, profile)
    hits = dedupe_similar_hits(hits)
    profile["retrieval_cleanup"] = {
        "pre_cleanup_hits": pre_cleanup_count,
        "post_cleanup_hits": len(hits),
        "dedupe_removed": max(0, pre_cleanup_count - len(hits)),
        "top_generation_doc_ids": [str(hit.get("doc_id") or "") for hit in hits[:5]],
    }
    if is_biek_knowledge_domain(profile.get("knowledge_domain")):
        profile["retrieval_cleanup"]["biek_year_rerank"] = {
            "active": True,
            "question_intent": profile.get("biek_time_intent"),
            "top_hit_years": [
                {
                    "chunk_id": str(hit.get("chunk_id") or ""),
                    "title": str(hit.get("title") or ""),
                    "hit_year": extract_biek_hit_year(hit),
                    "year_debug": hit.get("biek_year_rerank"),
                }
                for hit in hits[:5]
            ],
        }
    timings["retrieval_postprocess_s"] = round(time.perf_counter() - stage_started, 4)
    timings["retrieval_total_s"] = round(time.perf_counter() - total_started, 4)
    profile["timings"] = timings
    _RETRIEVAL_CACHE[cache_key] = (copy.deepcopy(hits), model, copy.deepcopy(profile))
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
        "classification": profile.get("model_classification") or {},
        "answer_policy": profile.get("answer_policy") or determine_answer_policy(question, profile),
        "timings": profile.get("timings", {}),
        "observability": profile.get("observability") or build_answer_observability(
            question=question,
            answer=answer,
            profile=profile,
            hits=hits,
            route="unknown",
            used_template=False,
            used_refusal=answer == DEFAULT_FALLBACK_RESPONSE,
        ),
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


def source_doc_ids(hits: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    doc_ids: list[str] = []
    for hit in hits:
        doc_id = str(hit.get("doc_id") or "").strip()
        if doc_id and doc_id not in seen:
            doc_ids.append(doc_id)
            seen.add(doc_id)
    return doc_ids


def build_answer_observability(
    *,
    question: str,
    answer: str,
    profile: dict[str, Any],
    hits: list[dict[str, Any]],
    route: str,
    used_template: bool,
    used_refusal: bool,
) -> dict[str, Any]:
    rewrite = profile.get("rewrite") or {}
    classification = profile.get("model_classification") or {}
    answer_policy = profile.get("answer_policy") or determine_answer_policy(question, profile)
    refusal = used_refusal or answer == DEFAULT_FALLBACK_RESPONSE or "not confirmed" in answer.lower()
    return {
        "question": question,
        "intent": answer_policy or classification.get("intent") or profile.get("intent") or "generic",
        "confidence": classification.get("confidence", 0.0),
        "route": route,
        "rewrite_query": rewrite.get("search_query") or profile.get("search_query") or question,
        "sources": source_doc_ids(hits),
        "metadata_routes": profile.get("metadata_routes", []),
        "retrieval_confidence": profile.get("retrieval_confidence"),
        "used_template": used_template,
        "used_refusal": refusal,
        "generation_experiment": RAG_GENERATE_ORDINARY_ANSWERS,
        "fast_rag": RAG_FAST_MODE,
        "ollama_keep_alive": OLLAMA_KEEP_ALIVE,
        "biek_answer_guard": profile.get("biek_answer_guard"),
        "context_compression": profile.get("context_compression"),
        "prompt_metrics": profile.get("prompt_metrics"),
        "model_profile": profile.get("runtime_model_profile"),
        "timings": profile.get("timings", {}),
    }


def observe_answer(
    *,
    question: str,
    answer: str,
    profile: dict[str, Any],
    hits: list[dict[str, Any]],
    route: str,
    used_template: bool = False,
    used_refusal: bool = False,
) -> str:
    final_answer = finalize_kb_answer(question, answer)
    event = build_answer_observability(
        question=question,
        answer=final_answer,
        profile=profile,
        hits=hits,
        route=route,
        used_template=used_template,
        used_refusal=used_refusal,
    )
    profile["observability"] = event
    log_kb_debug({"stage": "answer_observability", **event})
    return final_answer


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
    replacements = {
        "Opira Ai": "Opira AI",
        "opira ai": "Opira AI",
        "Coversaction Ai": "Coversaction AI",
        "coversaction ai": "Coversaction AI",
        "Conversaction Ai": "Coversaction AI",
        "Conversaction AI": "Coversaction AI",
        "coversoga AI": "Coversaction AI",
        "coversoga ai": "Coversaction AI",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    cleaned = re.sub(
        r"\b(?:our|Synapse(?: Tech(?: Inc\.)?)?'?s?)\s+products?\s+(?:like|include|includes)\s+(n8n|Make\.com|Airbyte|Airtable)",
        r"tools include \1",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bproducts?\s+(?:like|include|includes)\s+(n8n|Make\.com|Airbyte|Airtable)",
        r"tools include \1",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.replace("automates every form", "can automate forms")
    cleaned = cleaned.replace("automate every form", "automate forms")
    cleaned = cleaned.replace("policy query & breach check", "policy queries and breach checks")
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
    biek_relevant_document_types = {
        "notification",
        "datesheet",
        "model_paper",
        "exam_material",
        "scheme_of_studies",
        "form",
        "press_release",
        "results_document",
        "affiliation_list",
        "contact_info",
    }
    has_biek_relevant_doc = any(
        str(hit.get("document_type") or "").lower() in biek_relevant_document_types
        or str(hit.get("section") or "").lower() in {
            "notifications",
            "datesheet",
            "model_paper",
            "forms",
            "press_release",
            "results",
            "contact",
            "general",
        }
        for hit in hits[:4]
    )

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

    if is_biek_domain(profile) and low_risk and has_biek_relevant_doc:
        if best_combined >= 1 and top_score >= 0.35:
            return True
        if is_biek_model_paper_question(question) and top_score >= 0.3:
            return True
        if is_biek_esheet_question(question) and top_score >= 0.3:
            return True
        if is_biek_datesheet_question(question) and top_score >= 0.3:
            return True
        if is_biek_notification_question(question) and top_score >= 0.3:
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
    model_profile = get_model_profile(model)
    profile["runtime_model_profile"] = {
        "model": model,
        "name": model_profile["name"],
        "top_k": model_profile["top_k"],
        "context_k": model_profile["context_k"],
        "temperature": model_profile["temperature"],
        "num_predict": model_profile["num_predict"],
    }
    if not context_hits:
        log_kb_debug({"question": question, "stage": "empty_context", "fallback": True})
        return observe_answer(
            question=question,
            answer=DEFAULT_FALLBACK_RESPONSE,
            profile=profile,
            hits=[],
            route="empty_kb_fallback",
            used_refusal=True,
        )

    intent = profile.get("intent")
    answer_policy = determine_answer_policy(question, profile)
    generate_ordinary_answer = (
        RAG_GENERATE_ORDINARY_ANSWERS
        and answer_policy not in SAFETY_DETERMINISTIC_POLICIES
    )

    if is_biek_domain(profile):
        if is_biek_generic_help_question(question):
            profile["biek_answer_guard"] = {
                "route": "biek_help_direct",
                "stale_suppression": False,
            }
            return observe_answer(
                question=question,
                answer=build_biek_better_help_answer(question),
                profile=profile,
                hits=context_hits,
                route="biek_help_direct",
                used_template=True,
            )
        biek_direct_answer = ""
        if is_biek_datesheet_question(question):
            biek_direct_answer = build_biek_datesheet_answer(question, context_hits, profile)
            if biek_direct_answer:
                profile["biek_answer_guard"] = {
                    "route": "biek_datesheet_direct",
                    "stale_suppression": False,
                }
                return observe_answer(
                    question=question,
                    answer=biek_direct_answer,
                    profile=profile,
                    hits=context_hits,
                    route="biek_datesheet_direct",
                    used_template=True,
                )
        if is_biek_notification_question(question):
            biek_direct_answer = build_biek_notification_summary_answer(question, context_hits, profile)
            if biek_direct_answer:
                profile["biek_answer_guard"] = {
                    "route": "biek_notification_summary",
                    "stale_suppression": False,
                }
                return observe_answer(
                    question=question,
                    answer=biek_direct_answer,
                    profile=profile,
                    hits=context_hits,
                    route="biek_notification_summary",
                    used_template=True,
                )
        if is_biek_model_paper_question(question):
            biek_direct_answer = build_biek_model_paper_answer(question, context_hits, profile)
            if biek_direct_answer:
                profile["biek_answer_guard"] = {
                    "route": "biek_model_paper_direct",
                    "stale_suppression": False,
                }
                return observe_answer(
                    question=question,
                    answer=biek_direct_answer,
                    profile=profile,
                    hits=context_hits,
                    route="biek_model_paper_direct",
                    used_template=True,
                )
        if is_biek_esheet_question(question):
            biek_direct_answer = build_biek_esheet_answer(question, context_hits, profile)
            if biek_direct_answer:
                profile["biek_answer_guard"] = {
                    "route": "biek_esheet_direct",
                    "stale_suppression": False,
                }
                return observe_answer(
                    question=question,
                    answer=biek_direct_answer,
                    profile=profile,
                    hits=context_hits,
                    route="biek_esheet_direct",
                    used_template=True,
                )
        nav_coverage_rows = load_biek_nav_coverage()
        if is_biek_nav_section_question(question):
            matched_section_row = match_biek_nav_section(question, nav_coverage_rows)
            if matched_section_row:
                profile["biek_answer_guard"] = {
                    "route": "biek_nav_section_summary",
                    "stale_suppression": False,
                    "row_id": matched_section_row.get("id"),
                    "row_label": matched_section_row.get("label"),
                }
                return observe_answer(
                    question=question,
                    answer=build_biek_nav_section_summary(
                        str(matched_section_row.get("label") or ""),
                        nav_coverage_rows,
                    ),
                    profile=profile,
                    hits=context_hits,
                    route="biek_nav_section_summary",
                    used_template=True,
                )
        coverage_match = classify_biek_coverage_match(question, nav_coverage_rows)
        if coverage_match.get("applies") and coverage_match.get("match_type") == "exact_match":
            matched_row = coverage_match.get("best_row") or {}
            target_scoped_answer = generate_biek_target_scoped_answer(
                question=question,
                row=matched_row,
                coverage_rows=nav_coverage_rows,
                profile=profile,
                ollama_url=ollama_url,
                model=model,
                temperature=temperature,
                num_predict=num_predict,
            )
            profile["biek_answer_guard"] = {
                "route": "biek_coverage_exact_content",
                "stale_suppression": False,
                "match_type": "exact_match",
                "row_id": matched_row.get("id"),
                "row_label": matched_row.get("label"),
                "row_url": matched_row.get("url"),
                "query_tokens": coverage_match.get("query_tokens"),
                "evidence_pack_count": len((profile.get("biek_target_evidence") or {}).get("sources", [])),
            }
            return observe_answer(
                question=question,
                answer=target_scoped_answer or build_biek_content_backed_answer(question, matched_row, nav_coverage_rows),
                profile=profile,
                hits=context_hits,
                route="biek_coverage_exact_content",
                used_template=not bool(target_scoped_answer),
                used_refusal=not bool(target_scoped_answer and target_scoped_answer.strip()),
            )
        if is_biek_contact_question(question):
            contact_answer = build_biek_contact_safe_answer(context_hits)
            profile["biek_answer_guard"] = {
                "route": "biek_contact_safe",
                "stale_suppression": False,
            }
            return observe_answer(
                question=question,
                answer=contact_answer,
                profile=profile,
                hits=context_hits,
                route="biek_contact_safe",
                used_template=True,
                used_refusal="not clearly confirmed" in contact_answer.lower(),
            )
        if is_biek_identity_question(question) and not has_strong_biek_identity_evidence(context_hits):
            profile["biek_answer_guard"] = {
                "route": "biek_identity_safe",
                "stale_suppression": False,
            }
            safe_answer = build_biek_identity_safe_answer()
            return observe_answer(
                question=question,
                answer=safe_answer,
                profile=profile,
                hits=context_hits,
                route="biek_identity_safe",
                used_template=True,
                used_refusal=True,
            )
        if coverage_match.get("applies") and coverage_match.get("match_type") == "close_match":
            matched_rows = coverage_match.get("rows") or []
            best_row = coverage_match.get("best_row") or {}
            profile["biek_answer_guard"] = {
                "route": "biek_coverage_close_match",
                "stale_suppression": False,
                "match_type": "close_match",
                "row_id": best_row.get("id"),
                "row_label": best_row.get("label"),
                "query_tokens": coverage_match.get("query_tokens"),
            }
            return observe_answer(
                question=question,
                answer=build_biek_close_match_answer(question, matched_rows),
                profile=profile,
                hits=context_hits,
                route="biek_coverage_close_match",
                used_template=True,
            )
        if coverage_match.get("applies") and coverage_match.get("match_type") == "no_match":
            profile["biek_answer_guard"] = {
                "route": "biek_coverage_no_match",
                "stale_suppression": False,
                "match_type": "no_match",
                "query_tokens": coverage_match.get("query_tokens"),
            }
            return observe_answer(
                question=question,
                answer=build_biek_no_match_answer(),
                profile=profile,
                hits=context_hits,
                route="biek_coverage_no_match",
                used_template=True,
                used_refusal=True,
            )
        if is_biek_evidence_stale_for_question(question, context_hits, profile):
            stale_answer = build_biek_stale_safe_answer(question, context_hits, profile)
            profile["biek_answer_guard"] = {
                "route": "biek_stale_suppression",
                "stale_suppression": True,
                "latest_supported_year": latest_supported_biek_year(biek_relevant_hits(context_hits) or context_hits),
                "question_intent": profile.get("biek_time_intent"),
            }
            return observe_answer(
                question=question,
                answer=stale_answer,
                profile=profile,
                hits=context_hits,
                route="biek_stale_suppression",
                used_template=True,
                used_refusal=True,
            )

    if (
        not generate_ordinary_answer
        and not DISABLE_ROUTE_SPECIFIC_ANSWERS
        and intent in SUMMARY_INTENT_CONFIG
        and answer_policy in {"product_catalog", "service_catalog", "industry"}
        and answer_policy in HARD_DETERMINISTIC_POLICIES
        and allow_deterministic_summary(question, intent)
    ):
        summary_answer = build_summary_answer(intent, context_hits)
        if summary_answer:
            return observe_answer(
                question=question,
                answer=summary_answer,
                profile=profile,
                hits=context_hits,
                route="template_summary",
                used_template=True,
            )

    if (
        not generate_ordinary_answer
        and not DISABLE_ROUTE_SPECIFIC_ANSWERS
        and not DISABLE_NONCRITICAL_DIRECT_ANSWERS
        and answer_policy in HARD_DETERMINISTIC_POLICIES
    ):
        policy_answer = build_policy_answer(answer_policy, question, context_hits, profile)
        if policy_answer:
            return observe_answer(
                question=question,
                answer=clean_answer_text(policy_answer),
                profile=profile,
                hits=context_hits,
                route="hard_policy",
                used_template=True,
                used_refusal=answer_policy == "high_risk_unknown",
            )

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
        return observe_answer(
            question=question,
            answer=DEFAULT_FALLBACK_RESPONSE,
            profile=profile,
            hits=context_hits,
            route="support_refusal",
            used_refusal=True,
        )

    if not generate_ordinary_answer and not DISABLE_ROUTE_SPECIFIC_ANSWERS:
        runtime_answer = ""
        if answer_policy == "product_catalog" and wants_product_descriptions(question):
            runtime_answer = build_product_catalog_detail_answer(context_hits)
        elif is_workflow_automation_question(question):
            runtime_answer = build_workflow_automation_answer(context_hits)
        if runtime_answer:
            return observe_answer(
                question=question,
                answer=clean_answer_text(runtime_answer),
                profile=profile,
                hits=context_hits,
                route="runtime_template",
                used_template=True,
            )

    user_prompt = build_user_prompt(
        question,
        context_hits,
        profile,
        natural_output=generate_ordinary_answer,
        model=model,
    )
    predict_cap = max(num_predict, 160) if profile.get("company_overview") else num_predict
    usage = profile.setdefault("usage", {})
    answer_system_prompt = profile_system_prompt(
        NATURAL_ANSWER_SYSTEM_PROMPT if generate_ordinary_answer else SYSTEM_PROMPT,
        model,
        "answer_instructions",
    )
    retry_system_prompt = profile_system_prompt(
        SYSTEM_PROMPT,
        model,
        "answer_instructions",
    )
    raw = ollama_chat(
        ollama_url=ollama_url,
        model=model,
        messages=[
            {
                "role": "system",
                "content": answer_system_prompt,
            },
            {"role": "user", "content": user_prompt},
        ],
        temperature=max(temperature, 0.3) if generate_ordinary_answer else temperature,
        num_predict=predict_cap,
        usage_sink=usage,
        usage_label="answer_generation",
        json_format=not generate_ordinary_answer,
    )
    payload = safe_parse_json(raw)
    answer = (
        raw.strip()
        if generate_ordinary_answer
        else normalize_answer_value(payload.get("answer")) if isinstance(payload, dict) else ""
    )
    answer = clean_answer_text(answer)

    answer_policy = determine_answer_policy(question, profile)
    product_catalog_needs_model = answer_policy == "product_catalog" and (
        wants_product_descriptions(question) or is_customer_support_product_question(question)
    )
    low_risk_model_answer = (
        (answer_policy not in HARD_DETERMINISTIC_POLICIES or product_catalog_needs_model)
        and not is_high_risk_question(question)
    )
    should_retry_too_cautious = (
        not answer
        or answer == DEFAULT_FALLBACK_RESPONSE
    ) and (profile.get("company_overview") or low_risk_model_answer)

    if not RAG_FAST_MODE and should_retry_too_cautious:
        blob = " ".join(f"{h.get('title', '')} {h.get('text', '')}" for h in context_hits).lower()
        has_grounding = any(
            token in blob
            for token in (
                "synapse",
                "agentic bot",
                "opira",
                "workflow automation",
                "custom web",
                "mobile app",
                "private cloud",
                "offline",
            )
        )
        if has_grounding:
            retry_user = (
                user_prompt
                + "\n\nRetry: The previous JSON was invalid or too cautious. If the evidence contains the "
                "named product, service, or capability, answer from that evidence instead of using the "
                "not-confirmed reply. Write 2-4 factual sentences using only the evidence above. "
                'Return JSON {"answer":"<text>","supported":true}. '
                "Do not use the not-confirmed reply when the excerpts clearly describe the requested item."
            )
            raw_retry = ollama_chat(
                ollama_url=ollama_url,
                model=model,
                messages=[
                    {"role": "system", "content": retry_system_prompt},
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

    if not RAG_FAST_MODE and workflow_answer_too_shallow(question, answer):
        workflow_retry_user = (
            user_prompt
            + "\n\nRetry: The previous answer was too shallow because it mostly listed tools. "
            "Answer the user's workflow automation question by explaining what Synapse Tech does "
            "for the business using only the evidence above. Your answer must include: "
            "1) connecting APIs, databases, SaaS tools, or data pipelines; "
            "2) reducing manual handoffs, copy-paste work, or repetitive tasks; "
            "3) running workflows end-to-end with orchestration, logging, or visibility. "
            "Mention tools like n8n, Make.com, Airbyte, or Airtable only as supporting examples, "
            "not as the whole answer. Return JSON {\"answer\":\"<text>\",\"supported\":true}."
        )
        raw_workflow_retry = ollama_chat(
            ollama_url=ollama_url,
            model=model,
            messages=[
                {"role": "system", "content": retry_system_prompt},
                {"role": "user", "content": workflow_retry_user},
            ],
            temperature=min(0.2, temperature + 0.05),
            num_predict=max(predict_cap, 180),
            usage_sink=usage,
            usage_label="workflow_answer_retry",
        )
        payload_workflow_retry = safe_parse_json(raw_workflow_retry)
        retry_answer = (
            normalize_answer_value(payload_workflow_retry.get("answer"))
            if isinstance(payload_workflow_retry, dict)
            else ""
        )
        retry_answer = clean_answer_text(retry_answer)
        if retry_answer and not workflow_answer_too_shallow(question, retry_answer):
            answer = retry_answer
        elif not generate_ordinary_answer:
            grounded_answer = build_workflow_automation_answer(context_hits)
            if grounded_answer:
                answer = grounded_answer

    if not RAG_FAST_MODE and product_overview_answer_too_shallow(question, answer):
        product_retry_user = (
            user_prompt
            + "\n\nRetry: The previous answer was too generic. The user asked about Synapse Tech products, "
            "so name the actual products from the evidence and add a short grounded phrase for each. "
            "Include Agentic Bot, iRecruit One, Opira AI, Coversaction AI, and Cyber Security Automation "
            "if they appear in the evidence. Return JSON {\"answer\":\"<text>\",\"supported\":true}."
        )
        raw_product_retry = ollama_chat(
            ollama_url=ollama_url,
            model=model,
            messages=[
                {"role": "system", "content": retry_system_prompt},
                {"role": "user", "content": product_retry_user},
            ],
            temperature=min(0.2, temperature + 0.05),
            num_predict=max(predict_cap, 180),
            usage_sink=usage,
            usage_label="product_overview_retry",
        )
        payload_product_retry = safe_parse_json(raw_product_retry)
        retry_answer = (
            normalize_answer_value(payload_product_retry.get("answer"))
            if isinstance(payload_product_retry, dict)
            else ""
        )
        retry_answer = clean_answer_text(retry_answer)
        if retry_answer and not product_overview_answer_too_shallow(question, retry_answer):
            answer = retry_answer
        elif not generate_ordinary_answer:
            grounded_answer = build_product_catalog_detail_answer(context_hits)
            if grounded_answer:
                answer = grounded_answer

    if not RAG_FAST_MODE and product_detail_answer_needs_cleanup(question, answer, profile):
        product_retry_user = (
            user_prompt
            + "\n\nRetry: The previous product-detail answer used awkward or over-broad wording. "
            "Use the structured product card for the product name and approved short wording, then "
            "generate a natural 1-2 sentence answer from the evidence. Do not copy raw page headings, "
            "do not say 'automates every form', and do not mention internal evidence. "
            "Do not use absolute claims like eliminate vulnerabilities, guarantee, or without risk. "
            "Return JSON {\"answer\":\"<text>\",\"supported\":true}."
        )
        raw_product_detail_retry = ollama_chat(
            ollama_url=ollama_url,
            model=model,
            messages=[
                {"role": "system", "content": retry_system_prompt},
                {"role": "user", "content": product_retry_user},
            ],
            temperature=min(0.2, temperature + 0.05),
            num_predict=max(predict_cap, 160),
            usage_sink=usage,
            usage_label="product_detail_retry",
        )
        payload_product_detail_retry = safe_parse_json(raw_product_detail_retry)
        retry_answer = (
            normalize_answer_value(payload_product_detail_retry.get("answer"))
            if isinstance(payload_product_detail_retry, dict)
            else ""
        )
        retry_answer = clean_answer_text(retry_answer)
        if retry_answer and not product_detail_answer_needs_cleanup(question, retry_answer, profile):
            answer = retry_answer
        elif not generate_ordinary_answer:
            grounded_answer = build_entity_answer(question, context_hits, profile)
            if grounded_answer:
                answer = clean_answer_text(grounded_answer)

    comparison_issues = comparison_answer_issues(question, answer)
    instruction_issues = instruction_following_issues(question, answer)
    if not RAG_FAST_MODE and (comparison_issues or instruction_issues):
        expected_sentences = requested_sentence_count(question)
        constraints: list[str] = []
        if comparison_issues:
            constraints.append(
                f"Explicitly cover both products: {join_list(product_names_in_question(question))}."
            )
        if expected_sentences is not None:
            constraints.append(f"Write exactly {expected_sentences} complete sentences.")
        if requests_non_marketing_tone(question):
            constraints.append(
                "Use neutral factual language and remove slogans or promotional phrases."
            )
        constraint_text = " ".join(constraints)
        constraint_retry_user = (
            user_prompt
            + "\n\nRetry: The previous answer did not follow the user's requested answer shape. "
            + constraint_text
            + " Use only the evidence above. Return JSON {\"answer\":\"<text>\",\"supported\":true}."
        )
        raw_constraint_retry = ollama_chat(
            ollama_url=ollama_url,
            model=model,
            messages=[
                {"role": "system", "content": retry_system_prompt},
                {"role": "user", "content": constraint_retry_user},
            ],
            temperature=0.0,
            num_predict=max(predict_cap, 180),
            usage_sink=usage,
            usage_label="constraint_retry",
        )
        payload_constraint_retry = safe_parse_json(raw_constraint_retry)
        retry_answer = (
            normalize_answer_value(payload_constraint_retry.get("answer"))
            if isinstance(payload_constraint_retry, dict)
            else ""
        )
        retry_answer = clean_answer_text(retry_answer)
        if (
            retry_answer
            and not comparison_answer_issues(question, retry_answer)
            and not instruction_following_issues(question, retry_answer)
        ):
            answer = retry_answer
        elif comparison_issues and not generate_ordinary_answer:
            fallback = build_product_comparison_fallback(question)
            if fallback:
                answer = fallback
        elif instruction_issues and not generate_ordinary_answer:
            fallback = build_instruction_fallback(question)
            if fallback:
                answer = fallback

    if not generate_ordinary_answer and should_force_instruction_fallback(question, answer, profile):
        fallback = build_instruction_fallback(question)
        if fallback:
            answer = fallback

    if not generate_ordinary_answer and workflow_answer_too_shallow(question, answer):
        grounded_answer = build_workflow_automation_answer(context_hits)
        if grounded_answer:
            answer = grounded_answer

    if (
        not generate_ordinary_answer
        and is_private_deployment_question(question)
        and "not all" not in answer.lower()
    ):
        grounded_answer = build_private_deployment_answer(context_hits)
        if grounded_answer:
            return observe_answer(
                question=question,
                answer=grounded_answer,
                profile=profile,
                hits=context_hits,
                route="runtime_template",
                used_template=True,
            )

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
        return observe_answer(
            question=question,
            answer=DEFAULT_FALLBACK_RESPONSE,
            profile=profile,
            hits=context_hits,
            route="empty_generation_fallback",
            used_refusal=True,
        )
    return observe_answer(
        question=question,
        answer=answer,
        profile=profile,
        hits=context_hits,
        route="rag_generation",
    )


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
        classifier_model=args.classifier_model,
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
