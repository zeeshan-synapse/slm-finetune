"""Local Streamlit UI for testing Synapse model/RAG configurations.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import html
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
CHAT_DIR = PROJECT_ROOT / "chat"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in (CHAT_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from guardrail_stage1 import (  # noqa: E402
    GENERATOR_SYSTEM_PROMPT,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_URL,
    is_small_talk_question,
    ollama_chat,
    policy_intent,
    policy_response,
    safe_parse_json,
    small_talk_response,
)
from kb_answer import kb_grounded_answer_with_meta  # noqa: E402
import answer_with_kb as aw  # noqa: E402
try:
    from biek_results_lookup import result_lookup_answer  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - optional local BIEK result index feature
    def result_lookup_answer(question: str) -> str | None:
        return None


MODEL_OPTIONS: dict[str, dict[str, Any]] = {
    "Synapse Qwen 2.5 1.5B V1": {
        "fine_tuned_model": "synapse-1.5b-v1",
        "base_model": "qwen2.5:1.5b-instruct",
        "fine_tune_exists": True,
    },
    "Synapse Qwen 2.5 1.5B V2": {
        "fine_tuned_model": "synapse-1.5b-v2",
        "base_model": "qwen2.5:1.5b-instruct",
        "fine_tune_exists": True,
    },
    "Synapse Qwen 2.5 3B": {
        "fine_tuned_model": "synapse-qwen2.5-3b-v1",
        "base_model": "qwen2.5:3b",
        "fine_tune_exists": True,
    },
    "Synapse Llama V1 3B": {
        "fine_tuned_model": "synapse-llama3-v1",
        "base_model": "synapse-llama3-v1-base",
        "fine_tune_exists": True,
    },
    "Synapse Llama V1 8B": {
        "fine_tuned_model": None,
        "base_model": "llama3:latest",
        "fine_tune_exists": False,
    },
    "Synapse Gemma 3 4B V1": {
        "fine_tuned_model": "synapse-gemma3-4b-v1",
        "base_model": "gemma3:4b",
        "fine_tune_exists": True,
    },
    "Qwen2.5 7B": {
        "fine_tuned_model": None,
        "base_model": "qwen2.5:7b",
        "fine_tune_exists": False,
    },
}
DEFAULT_MODEL_LABEL = "Synapse Qwen 2.5 1.5B V2"
KNOWLEDGE_SOURCE_OPTIONS = {
    "Synapse KB": "synapse",
    "BIEK KB": "biek",
    "BIEK Full KB": "biek_full",
}

ANSWER_MODES = {
    "Base": "base_plain",
    "Base + RAG": "base_rag",
    "Fine-tuned": "fine_tuned_plain",
    "Fine-tuned + RAG": "fine_tuned_rag",
}
BATCH_QUESTION_PRESETS = {
    "synapse": (
        "What does Synapse Tech do?\n"
        "What products does Synapse Tech offer?\n"
        "Which Synapse product is best for recruitment automation?\n"
        "How does iRecruit One improve the hiring process?\n"
        "Tell me about Agentic Bot and its main capabilities.\n"
        "Which product would help automate customer support conversations?\n"
        "What is Coversaction AI used for?\n"
        "Can Opira AI operate offline or on private infrastructure?\n"
        "How does Synapse Tech help businesses automate workflows?\n"
        "Do you build custom web and mobile applications?\n"
        "What voice agent and call automation services do you provide?\n"
        "Which industries does Synapse Tech serve?\n"
        "How can I choose the right Synapse solution for my business?\n"
        "How can I purchase a Synapse Tech product or contact your team?\n"
        "Do you publish pricing, SLA guarantees, or security certifications?"
    ),
    "biek": (
        "What is the history of the Board?\n"
        "Who is the chairman of BIEK?\n"
        "Who are the board members of BIEK?\n"
        "What committees does BIEK have?\n"
        "What were the last BIEK results?\n"
        "How can I contact BIEK?\n"
        "What circulars and notifications are available on the BIEK website?\n"
        "What press releases are available on the BIEK website?\n"
        "What is the recent result declaration page for?\n"
        "What does the BIEK date sheet page contain?\n"
        "What are the authorized banks for BIEK?\n"
        "What statistics does BIEK provide?\n"
        "Do you have the sample E-Sheet of 30 pages for Annual 2026 E-Marking?\n"
        "Do you have the sample E-Sheet of 22 pages for Annual 2026 E-Marking?\n"
        "What is the scheme of studies for higher secondary certificates?\n"
        "Do you have Scheme and Model Paper 2023?\n"
        "Is Model Paper 2026 available?\n"
        "Do you have the Science General Mathematics Part-II model paper?\n"
        "Do you have MCQs on the BIEK website?\n"
        "Do you have the certificate form?\n"
        "Do you have the scrutiny form?\n"
        "Do you have the provisional certification form?\n"
        "Do you have the migration form?\n"
        "Do you have the registration form for commerce?\n"
        "Do you have the registration form for humanities?\n"
        "Do you have the verification certificate form?\n"
        "Do you have the verification provisional certificate form?\n"
        "Do you have the verification marksheet form?\n"
        "Do you have the verification migration form?\n"
        "Do you have the cancellation of enrolment form?\n"
        "Do you have the cancellation of registration form?\n"
        "Do you have the duplicate marksheet form?\n"
        "Do you have the duplicate enrolment card form?\n"
        "Do you have the duplicate computerized admit card form?\n"
        "Do you have the duplicate registration card form?\n"
        "Do you have the duplicate manual admit card form?\n"
        "Do you have the proforma of special chance?\n"
        "Do you have the improvement of division form?\n"
        "Do you have the registration of all groups form?\n"
        "Do you have the examination forms page?\n"
        "Do you have the permission forms page?\n"
        "What affiliated colleges information is available on the BIEK website?\n"
        "What is the IOC proforma for affiliation?\n"
        "What can I find under About Us on the BIEK website?\n"
        "What can I find under Misc on the BIEK website?\n"
        "What can I find under Examinations on the BIEK website?\n"
        "What can I find under Download Forms on the BIEK website?\n"
        "What can I find under Recognition on the BIEK website?\n"
        "What can you help me with regarding BIEK?\n"
        "What kind of information can I ask about on the BIEK website?\n"
        "Can you help me find forms on the BIEK website?\n"
        "Can you help me find model papers on the BIEK website?\n"
        "Can you help me check BIEK results?\n"
        "Give me the result for roll number 800926.\n"
        "Give me the result for roll number 808259.\n"
        "Show me the 2025 supplementary commerce regular result for roll number 800926.\n"
        "Show me the 2025 supplementary commerce regular result for roll number 808259.\n"
        "I want my BIEK result for roll number 800926.\n"
        "Can you check the BIEK result for roll number 808259?"
    ),
}
MODEL_PARAMETER_SIZES = {
    "qwen2.5:1.5b-instruct": "1.5B",
    "qwen-base:latest": "1.5B",
    "synapse-1.5b-v1": "1.5B",
    "synapse-1.5b-v2": "1.5B",
    "synapse-llama3-v1-base": "3B",
    "synapse-llama3-v1": "3B",
    "llama3:latest": "8B",
    "gemma3:4b": "4B",
    "synapse-gemma3-4b-v1": "4B",
    "qwen2.5:3b": "3B",
    "qwen2.5:7b": "7B",
}
MODEL_OPTION_DISPLAY_NAMES = {
    "Synapse 1.5B V1": "Synapse Qwen 2.5 1.5B V1",
    "Synapse 1.5B V2": "Synapse Qwen 2.5 1.5B V2",
    "Synapse Llama 3B V1": "Synapse Llama V1 3B",
    "Qwen2.5 3B": "Synapse Qwen 2.5 3B",
}
MODEL_TAG_DISPLAY_NAMES = {
    "synapse-llama3-v1-base": "Synapse Llama V1 3B (base)",
    "synapse-llama3-v1": "Synapse Llama V1 3B",
    "llama3:latest": "Synapse Llama V1 8B (llama3:latest)",
    "qwen2.5:1.5b-instruct": "Synapse Qwen 2.5 1.5B",
    "synapse-1.5b-v1": "Synapse Qwen 2.5 1.5B V1",
    "synapse-1.5b-v2": "Synapse Qwen 2.5 1.5B V2",
    "qwen2.5:3b": "Synapse Qwen 2.5 3B (base)",
    "synapse-qwen2.5-3b-v1": "Synapse Qwen 2.5 3B",
}


def resolved_model_display(
    model_label: str,
    mode_label: str,
    generation_model: str | None,
) -> str:
    if generation_model in MODEL_TAG_DISPLAY_NAMES:
        return MODEL_TAG_DISPLAY_NAMES[generation_model]
    display_label = model_label.replace("Synapse Llama 3B V1", "Synapse Llama V1")
    size = MODEL_PARAMETER_SIZES.get(generation_model or "", "")
    size_suffix = f" {size}" if size and size.lower() not in display_label.lower() else ""
    if "fine-tuned" in mode_label.lower():
        return f"{display_label}{size_suffix}"
    return f"{display_label}{size_suffix} ({generation_model or 'unknown'})"


def model_family(model_key: str, display_name: str) -> str:
    searchable = f"{model_key} {display_name}".lower()
    if "llama" in searchable:
        return "Llama"
    if "gemma" in searchable:
        return "Gemma"
    return "Qwen"


V2_BUCKETS_PATH = PROJECT_ROOT / "eval" / "prompts" / "v2_50_buckets.json"
DEMO_15_GOLD_PATH = PROJECT_ROOT / "eval" / "prompts" / "demo_15_gold.json"
BIEK_GOLD_PATH = PROJECT_ROOT / "eval" / "prompts" / "biek_gold_v1.json"
BIEK_BATCH_GOLD_PATH = PROJECT_ROOT / "eval" / "prompts" / "biek_batch_gold_v1.json"
BATCH_HISTORY_PATH = PROJECT_ROOT / "eval" / "results" / "batch_eval_history.json"
REFUSAL_MARKERS = (
    "not confirmed",
    "not publicly listed",
    "not publicly confirmed",
    "not listed",
    "not available",
    "not specified",
    "please verify",
)
PROMPT_LEAK_PATTERNS = (
    "answer according to",
    "return json",
    "system prompt",
    "developer message",
    "rewrite query",
    "according to the prompt",
)
BIEK_TEMPLATE_OPENERS = (
    "i found",
    "biek has",
    "you can view",
    "you can find",
    "the page is here",
    "the document is here",
    "this page contains",
    "the page says",
    "i could not clearly confirm a direct",
)
BIEK_GENERIC_DRIFT_PATTERNS = (
    "historical and cultural institution",
    "proof of enrollment",
    "educational program",
    "serves as proof of eligibility",
    "institution's policies and regulations",
    "students and educators alike",
)
BIEK_DOMAIN_LEAK_PATTERNS = (
    "synapse tech",
    "products, services, capabilities",
    "deployment options",
    "tech company",
)
ROUTER_SYSTEM_PROMPT = """You are a scope router for the Synapse Tech chatbot.

Your job is to classify the user's message by both scope and intent validity.

The chatbot may answer:
- questions directly about Synapse Tech Inc. and its offerings
- questions closely related to Synapse Tech's business domain

The chatbot must refuse unrelated general questions.

Scope categories:

1. direct_synapse
Use this when the user's message is primarily about Synapse Tech Inc., its website content, company information, products, services, industries served, deployment options, capabilities, contact paths, quote/pricing availability, integrations if publicly confirmed, security/compliance if publicly confirmed, or choosing which Synapse offering fits a business need.

2. adjacent_in_scope
Use this when the user's message is not primarily about Synapse Tech itself, but is clearly within the surrounding domain of AI business software, enterprise automation, workflow automation, recruitment automation, customer support automation, AI voice agents, business chatbots, private AI deployment, offline LLMs, RAG systems, cloud AI infrastructure, operational intelligence, or selecting AI/automation approaches for a business use case.

3. out_of_scope
Use this when the user's message is unrelated to Synapse Tech and unrelated to the surrounding AI/business automation domain. This includes coding help, debugging, script explanation, homework, medicine, health advice, restaurants, food recommendations, fashion, hair dye, sports, entertainment, personal advice, travel, and general facts outside the allowed domain.

Decision rules:

- Classify by meaning, not by exact keywords.
- If the user is asking about Synapse Tech directly, classify as direct_synapse.
- If the user refers to "your service", "your product", "you", "you guys", or similar language and the message is about AI, automation, enterprise software, customer support, recruitment, deployment, or business workflows, classify as direct_synapse.
- If the user is asking which Synapse product or service fits a business need, classify as direct_synapse even if no product name is mentioned.
- If the user is asking a general domain question about AI, automation, enterprise workflows, customer support automation, recruitment automation, RAG, private AI, or similar business-tech topics without primarily asking about Synapse itself, classify as adjacent_in_scope.
- If the user asks whether a named Synapse product can run offline, on-premises, behind a firewall, on private infrastructure, or in a private cloud, classify it as direct_synapse with valid intent.
- If the user is asking for programming help, code explanation, debugging, medical information, restaurant advice, fashion advice, sports, travel, or other unrelated topics, classify as out_of_scope.
- Judge whether the requested action or intent is valid and meaningful in the detected domain.
- If the message mentions a Synapse product or domain topic but the requested action is nonsensical, impossible, or incoherent for that product/topic, mark intent_validity as invalid.
- Examples of invalid intent: asking to eat a software product, wear a cloud service, or perform some other request that does not make sense for the referenced business software/topic.
- If the message mixes topics, classify based on the main user intent.
- If unsure between direct_synapse and adjacent_in_scope, choose adjacent_in_scope.
- If unsure whether the topic belongs to the Synapse / AI business automation domain at all, choose out_of_scope.
- Do not answer the user's question.
- Return only valid JSON.
- Do not include markdown.
- Do not include any text before or after the JSON.

Return exactly this JSON schema:
{"scope":"direct_synapse|adjacent_in_scope|out_of_scope","intent_validity":"valid|invalid","entity":"","confidence":0.0,"reason":"short explanation"}"""
VALID_SCOPE_LABELS = {"direct_synapse", "adjacent_in_scope", "out_of_scope"}
VALID_INTENT_LABELS = {"valid", "invalid"}
PRODUCT_NAME_ALIASES = {
    "irecruit one": "iRecruit One",
    "irecruit": "iRecruit One",
    "agentic bot": "Agentic Bot",
    "opira ai": "Opira AI",
    "opira": "Opira AI",
    "coversaction ai": "Coversaction AI",
    "coversaction": "Coversaction AI",
    "conversaction ai": "Coversaction AI",
    "conversaction": "Coversaction AI",
    "cyber security automation": "Cyber Security Automation",
}
CONTACT_CALL_TO_ACTION = (
    "contact Synapse Tech directly at info@synapsetechinc.com or chat with the team "
    "Monday to Saturday (8 AM - 8 PM)"
)
DEPLOYMENT_TERMS = (
    "offline",
    "on-prem",
    "on prem",
    "on-premise",
    "on premise",
    "on-premises",
    "on premises",
    "private infrastructure",
    "private cloud",
    "behind your firewall",
    "behind the firewall",
    "air-gapped",
    "air gapped",
)


def ollama_plain_answer(
    *,
    model: str,
    question: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "messages": [
                {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        },
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    return {
        "answer": data.get("message", {}).get("content", "").strip(),
        "generation_model": model,
        "usage": {
            "plain_generation": {
                "model": model,
                "input_tokens": data.get("prompt_eval_count"),
                "output_tokens": data.get("eval_count"),
                "total_tokens": (data.get("prompt_eval_count") or 0)
                + (data.get("eval_count") or 0),
            }
        },
    }


def is_biek_domain(knowledge_domain: str) -> bool:
    return (knowledge_domain or "").strip().lower().startswith("biek")


def domain_assistant_label(knowledge_domain: str) -> str:
    if is_biek_domain(knowledge_domain):
        return "BIEK assistant"
    return "Synapse chatbot assistant"


def domain_help_text(knowledge_domain: str) -> str:
    if is_biek_domain(knowledge_domain):
        return (
            "I can help with BIEK notifications, forms, datesheets, results-related information, "
            "affiliated college details, and other questions grounded in the BIEK knowledge base."
        )
    return (
        "I can help with Synapse Tech's products, services, capabilities, deployment options, "
        "and related AI and business automation questions."
    )


def small_talk_response_for_domain(question: str, knowledge_domain: str) -> str:
    low_q = question.strip().lower()
    if is_biek_domain(knowledge_domain):
        if low_q in {"thanks", "thank you"}:
            return "You're welcome. If you have a BIEK question, I can help."
        if low_q in {"who are you", "what can you do"}:
            return (
                "I am the BIEK assistant. I can help with notifications, forms, datesheets, "
                "results-related information, and other BIEK knowledge-base questions."
            )
        return "Hi! How can I help you with BIEK today?"
    return small_talk_response(question)


def quick_bypass(question: str, knowledge_domain: str) -> dict[str, Any] | None:
    if is_small_talk_question(question):
        return {
            "answer": small_talk_response_for_domain(question, knowledge_domain),
            "generation_model": None,
            "usage": {"quick_bypass": True},
            "observability": {
                "intent": "small_talk",
                "route": "quick_bypass",
                "sources": [],
                "used_refusal": False,
            },
        }
    intent = policy_intent(question)
    if intent is not None:
        return {
            "answer": policy_response(intent),
            "generation_model": None,
            "usage": {"quick_bypass": True},
            "observability": {
                "intent": intent,
                "route": "policy_bypass",
                "sources": [],
                "used_refusal": False,
            },
        }
    return None


def canonical_product_name(text: str) -> str | None:
    low = normalize_text(text)
    for alias, canonical in PRODUCT_NAME_ALIASES.items():
        if alias in low:
            return canonical
    return None


def has_follow_up_reference(question: str) -> bool:
    low = normalize_text(question)
    return bool(re.search(r"\b(that|this|it|that product|this product)\b", low))


def is_purchase_question(question: str) -> bool:
    low = normalize_text(question)
    return contains_any(
        low,
        (
            "purchase",
            "buy",
            "acquire",
            "get started",
            "contact",
            "reach out",
            "share details",
            "more details",
            "details please",
            "interested in",
            "want that product",
            "want this product",
        ),
    )


def referenced_product_from_messages(messages: list[dict[str, Any]]) -> str | None:
    for item in reversed(messages):
        for field in ("question", "answer"):
            value = str(item.get(field) or "")
            product = canonical_product_name(value)
            if product:
                return product
    return None


def purchase_response(question: str, messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not is_purchase_question(question):
        return None

    product = canonical_product_name(question)
    if not product and has_follow_up_reference(question):
        product = referenced_product_from_messages(messages)

    if product:
        answer = (
            f"To get started with {product}, {CONTACT_CALL_TO_ACTION}. "
            f"Tell them you're interested in {product} and share what you want it to handle."
        )
    else:
        answer = (
            f"To get started, {CONTACT_CALL_TO_ACTION}. "
            "If you already know which product you want, mention it when you reach out."
        )

    return {
        "answer": answer,
        "generation_model": None,
        "usage": {"quick_bypass": True},
        "observability": {
            "intent": "purchase",
            "route": "purchase_bypass",
            "sources": ["contact-us"] if product is None else ["contact-us", product],
            "used_refusal": False,
        },
    }


def selected_model_for_mode(model_config: dict[str, Any], answer_mode: str) -> str | None:
    if answer_mode.startswith("fine_tuned"):
        if not model_config.get("fine_tune_exists"):
            return None
        return model_config.get("fine_tuned_model")
    return model_config["base_model"]


def is_valid_product_deployment_question(question: str) -> bool:
    product = canonical_product_name(question)
    if not product:
        return False
    return contains_any(normalize_text(question), DEPLOYMENT_TERMS)


def classify_scope(question: str, classifier_model: str) -> dict[str, Any]:
    raw = ollama_chat(
        classifier_model,
        [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
        num_predict=180,
    )
    payload = safe_parse_json(raw)
    scope = str(payload.get("scope") or "").strip()
    if scope not in VALID_SCOPE_LABELS:
        return {
            "scope": "out_of_scope",
            "intent_validity": "invalid",
            "entity": "",
            "confidence": 0.0,
            "reason": "The classifier output was invalid, so the request is treated as out of scope.",
        }
    intent_validity = str(payload.get("intent_validity") or "").strip().lower()
    if intent_validity not in VALID_INTENT_LABELS:
        intent_validity = "invalid" if scope == "out_of_scope" else "valid"
    confidence = payload.get("confidence", 0.0)
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 0.0
    return {
        "scope": scope,
        "intent_validity": intent_validity,
        "entity": str(payload.get("entity") or "").strip(),
        "confidence": max(0.0, min(confidence_value, 1.0)),
        "reason": str(payload.get("reason") or "").strip(),
    }


def out_of_scope_response(
    question: str,
    scope_result: dict[str, Any],
    knowledge_domain: str,
) -> dict[str, Any]:
    return {
        "answer": f"I don't know about that topic. I'm a {domain_assistant_label(knowledge_domain)}. {domain_help_text(knowledge_domain)}",
        "generation_model": None,
        "usage": {"scope_router": scope_result},
        "observability": {
            "intent": "out_of_scope",
            "route": "scope_router_refusal",
            "sources": [],
            "used_refusal": True,
            "scope_router": scope_result,
        },
    }


def invalid_intent_response(
    scope_result: dict[str, Any],
    knowledge_domain: str,
) -> dict[str, Any]:
    return {
        "answer": f"I don't know about this topic. I'm a {domain_assistant_label(knowledge_domain)}. {domain_help_text(knowledge_domain)}",
        "generation_model": None,
        "usage": {"scope_router": scope_result},
        "observability": {
            "intent": "invalid_intent",
            "route": "scope_router_invalid_intent",
            "sources": [],
            "used_refusal": True,
            "scope_router": scope_result,
        },
    }


def biek_result_lookup_response(question: str) -> dict[str, Any] | None:
    answer = result_lookup_answer(question)
    if answer is None:
        return None
    return {
        "answer": answer,
        "generation_model": None,
        "usage": {},
        "observability": {
            "intent": "result_lookup",
            "route": "biek_result_lookup",
            "sources": [],
            "used_refusal": False,
        },
    }


def answer_question(
    *,
    question: str,
    messages: list[dict[str, Any]],
    model_config: dict[str, Any],
    answer_mode: str,
    rag_generated: bool,
    combined_planning: bool,
    fast_rag: bool,
    knowledge_domain: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    aw.RAG_GENERATE_ORDINARY_ANSWERS = rag_generated
    aw.RAG_COMBINED_PLANNING = combined_planning
    aw.RAG_FAST_MODE = fast_rag
    normalized_knowledge_domain = (knowledge_domain or "synapse").strip().lower() or "synapse"
    base_model = model_config["base_model"]
    fine_tuned_model = model_config.get("fine_tuned_model")
    selected_model = selected_model_for_mode(model_config, answer_mode)
    rag_mode = answer_mode in {"base_rag", "fine_tuned_rag"}

    if is_biek_domain(normalized_knowledge_domain):
        result_lookup = biek_result_lookup_response(question)
        if result_lookup is not None:
            return result_lookup

    if answer_mode == "base_plain":
        return ollama_plain_answer(
            model=base_model,
            question=question,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if answer_mode == "fine_tuned_plain":
        if not model_config.get("fine_tune_exists"):
            return {
                "answer": f"Fine-tuned version for {base_model} does not exist in this project.",
                "generation_model": None,
                "usage": {},
            }
        return ollama_plain_answer(
            model=fine_tuned_model,
            question=question,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    bypass = quick_bypass(question, normalized_knowledge_domain)
    if bypass is not None:
        return bypass

    if rag_mode and normalized_knowledge_domain != "synapse":
        scope_result = None
    else:
        purchase_bypass = purchase_response(question, messages)
        if purchase_bypass is not None:
            return purchase_bypass

        if rag_mode and is_valid_product_deployment_question(question):
            scope_result = {
                "scope": "direct_synapse",
                "intent_validity": "valid",
                "entity": canonical_product_name(question) or "",
                "confidence": 1.0,
                "reason": "This is a valid deployment question about a named Synapse product.",
            }
        elif rag_mode and selected_model is not None:
            scope_result = classify_scope(question, selected_model)
        else:
            scope_result = None

    if scope_result is not None:
        if scope_result["intent_validity"] == "invalid":
            return invalid_intent_response(scope_result, normalized_knowledge_domain)
        if scope_result["scope"] == "out_of_scope":
            return out_of_scope_response(question, scope_result, normalized_knowledge_domain)

    if answer_mode == "base_rag":
        return kb_grounded_answer_with_meta(
            question,
            knowledge_domain=knowledge_domain,
            generation_model=base_model,
            rewrite_model=base_model,
            classifier_model=base_model,
            temperature=temperature,
            num_predict=max_tokens,
        )

    if answer_mode == "fine_tuned_rag":
        if not model_config.get("fine_tune_exists"):
            return {
                "answer": f"Fine-tuned version for {base_model} does not exist in this project.",
                "generation_model": None,
                "usage": {},
            }
        return kb_grounded_answer_with_meta(
            question,
            knowledge_domain=knowledge_domain,
            generation_model=fine_tuned_model,
            rewrite_model=fine_tuned_model,
            classifier_model=fine_tuned_model,
            temperature=temperature,
            num_predict=max_tokens,
        )

    raise ValueError(f"Unsupported answer mode: {answer_mode}")


def source_rows(result: dict[str, Any]) -> list[str]:
    observability = result.get("observability") or {}
    sources = observability.get("sources") or []
    return [str(source) for source in sources if source]


def retrieval_confidence(result: dict[str, Any]) -> dict[str, Any]:
    observability = result.get("observability") or {}
    confidence = observability.get("retrieval_confidence") or {}
    return confidence if isinstance(confidence, dict) else {}


def prompt_metrics(result: dict[str, Any]) -> dict[str, Any]:
    observability = result.get("observability") or {}
    metrics = observability.get("prompt_metrics") or {}
    return metrics if isinstance(metrics, dict) else {}


def expected_source_ids(question: str) -> list[str]:
    low = question.lower()
    if "agentic" in low:
        return ["product-agentic-bot"]
    if contains_any(low, ("irecruit", "recruit", "hiring", "resume", "candidate", "cv")):
        return ["product-irecruit-one"]
    if contains_any(low, ("coversaction", "conversaction", "customer support", "support conversation", "chatbot")):
        return ["product-coversaction-ai"]
    if contains_any(low, ("opira", "offline", "private infrastructure", "on-prem", "private cloud")):
        return ["product-opira-ai"]
    if "workflow" in low and "automat" in low:
        return ["services-automation"]
    if contains_any(low, ("custom web", "custom mobile", "custom software", "web and mobile")):
        return ["services-custom-development"]
    if contains_any(low, ("voice agent", "call automation")):
        return ["services-voice-agent"]
    if contains_any(low, ("contact", "purchase", "buy", "get started")):
        return ["contact-us"]
    if "industr" in low:
        return ["industries"]
    if contains_any(low, ("what products", "list products", "products does synapse")):
        return ["product"]
    if contains_any(low, ("what does synapse", "summarize synapse", "explain synapse")):
        return ["about-us"]
    return []


def render_message(item: dict[str, Any], *, debug_enabled: bool, show_sources: bool) -> None:
    with st.chat_message("user"):
        st.write(item["question"])

    with st.chat_message("assistant"):
        st.write(item["answer"])
        generator_display = resolved_model_display(
            item["model_label"],
            item["mode_label"],
            item.get("generation_model"),
        )
        st.caption(
            f"Mode: {item['mode_label']} | Model: {item['model_label']} | "
            f"Knowledge: {item.get('knowledge_source_label', 'Synapse KB')} | "
            f"Generator: {generator_display if item.get('generation_model') else 'quick bypass'} | "
            f"Time: {item.get('elapsed_s', 0.0):.2f}s"
        )

        sources = source_rows(item.get("raw", {}))
        if show_sources and sources:
            with st.expander(f"Sources ({len(sources)})"):
                for source in sources:
                    st.code(source, language=None)

        if debug_enabled:
            with st.expander("Debug metadata"):
                st.json(item.get("raw", {}))


def run_generation_job(job: dict[str, Any]) -> None:
    started = time.perf_counter()
    try:
        result = answer_question(
            question=job["question"],
            messages=job["messages"],
            model_config=job["model_config"],
            answer_mode=job["answer_mode"],
            rag_generated=job["rag_generated"],
            combined_planning=job["combined_planning"],
            fast_rag=job["fast_rag"],
            knowledge_domain=job["knowledge_domain"],
            temperature=job["temperature"],
            max_tokens=job["max_tokens"],
        )
    except Exception as exc:
        result = {
            "answer": f"Error: {exc}",
            "generation_model": None,
            "usage": {},
        }
    elapsed_s = time.perf_counter() - started
    result["elapsed_s"] = round(elapsed_s, 3)
    job["result"] = result
    job["elapsed_s"] = elapsed_s
    job["done"] = True


def start_generation_job(
    *,
    question: str,
    messages: list[dict[str, Any]],
    model_config: dict[str, Any],
    mode_label: str,
    answer_mode: str,
    model_label: str,
    rag_generated: bool,
    combined_planning: bool,
    fast_rag: bool,
    knowledge_domain: str,
    knowledge_source_label: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    job = {
        "id": str(time.time_ns()),
        "question": question,
        "messages": [dict(item) for item in messages],
        "model_config": dict(model_config),
        "mode_label": mode_label,
        "answer_mode": answer_mode,
        "model_label": model_label,
        "rag_generated": rag_generated,
        "combined_planning": combined_planning,
        "fast_rag": fast_rag,
        "knowledge_domain": knowledge_domain,
        "knowledge_source_label": knowledge_source_label,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "started_at": time.perf_counter(),
        "done": False,
        "cancelled": False,
        "result": None,
    }
    thread = threading.Thread(target=run_generation_job, args=(job,), daemon=True)
    job["thread"] = thread
    thread.start()
    return job


def render_active_job(debug_enabled: bool, show_sources: bool) -> None:
    job = st.session_state.get("active_job")
    if not job:
        return

    if job.get("done"):
        st.session_state.active_job = None
        if not job.get("cancelled"):
            result = job.get("result") or {}
            item = {
                "question": job["question"],
                "answer": result.get("answer", ""),
                "mode_label": job["mode_label"],
                "model_label": job["model_label"],
                "knowledge_source_label": job.get("knowledge_source_label", "Synapse KB"),
                "generation_model": result.get("generation_model"),
                "elapsed_s": job.get("elapsed_s", 0.0),
                "raw": result,
            }
            st.session_state.messages.append(item)
        st.rerun()

    with st.chat_message("user"):
        st.write(job["question"])

    with st.chat_message("assistant"):
        elapsed = time.perf_counter() - job["started_at"]
        st.write(f"Generating answer... {elapsed:.1f}s")
        if st.button("Stop generating", key=f"stop_{job['id']}"):
            job["cancelled"] = True
            st.session_state.active_job = None
            st.rerun()

    time.sleep(0.5)
    st.rerun()


@st.cache_data(show_spinner=False)
def load_gold_manifest_for_domain(knowledge_domain: str) -> dict[str, dict[str, Any]]:
    paths: list[Path]
    if is_biek_domain(knowledge_domain):
        paths = [BIEK_BATCH_GOLD_PATH, BIEK_GOLD_PATH]
    else:
        paths = [DEMO_15_GOLD_PATH]

    manifest: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("cases") or []
        for row in rows:
            question = row.get("question")
            if not question:
                continue
            manifest[normalize_text(question)] = row
    return manifest


@st.cache_data(show_spinner=False)
def load_v2_eval_manifest() -> dict[str, dict[str, Any]]:
    if not V2_BUCKETS_PATH.exists():
        return {}
    payload = json.loads(V2_BUCKETS_PATH.read_text(encoding="utf-8"))
    rows = payload.get("prompt_buckets") or []
    return {normalize_text(row.get("prompt", "")): row for row in rows if row.get("prompt")}


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def contains_any(text: str, needles: tuple[str, ...] | list[str]) -> bool:
    low = text.lower()
    return any(needle.lower() in low for needle in needles)


def is_refusal_answer(answer: str) -> bool:
    return contains_any(answer, REFUSAL_MARKERS)


def word_count(text: str) -> int:
    return len([token for token in re.split(r"\s+", text.strip()) if token])


def sentence_count(text: str) -> int:
    return len([part for part in re.split(r"[.!?]+\s+|\n+", text.strip()) if part.strip()])


def has_link_like_text(text: str) -> bool:
    low = text.lower()
    return (
        "http://" in low
        or "https://" in low
        or "www.biek.edu.pk" in low
        or "biek.edu.pk/" in low
        or ".pdf" in low
    )


def has_repetition(answer: str) -> bool:
    normalized = normalize_text(answer)
    if not normalized:
        return False
    sentences = [
        normalize_text(part)
        for part in re.split(r"[.!?]\s+|\n+", answer)
        if normalize_text(part)
    ]
    seen: set[str] = set()
    for sentence in sentences:
        if len(sentence) > 35 and sentence in seen:
            return True
        seen.add(sentence)
    words = normalized.split()
    if len(words) >= 24:
        first = " ".join(words[:12])
        rest = " ".join(words[12:])
        return first in rest
    return False


def has_prompt_leak(answer: str) -> bool:
    return contains_any(answer, PROMPT_LEAK_PATTERNS)


def has_bad_contact_claim(answer: str) -> bool:
    low = answer.lower()
    if "info@synapsetechinc.com" in low:
        return False
    if (
        "info@biek.edu.pk" in low
        or "contactus.asp" in low
        or "99260211" in low
        or "99260212" in low
        or "99260213" in low
    ):
        return False
    return (
        "@" in answer
        or "add to cart" in low
        or "checkout" in low
        or bool(re.search(r"\b(?:\+?\d[\d(). -]{7,}\d)\b", answer))
    )


def gold_case_issues(case: dict[str, Any], answer: str) -> list[str]:
    issues: list[str] = []
    low_a = answer.lower()
    if case.get("require_refusal", False) and not is_refusal_answer(answer):
        issues.append("missing_required_refusal")
    if not case.get("allowed_refusal", False) and not case.get("require_refusal", False) and is_refusal_answer(answer):
        issues.append("over_refusal")

    for needle in (case.get("must_include") or []) + (case.get("contains_all") or []):
        if needle.lower() not in low_a:
            issues.append(f"missing_required:{needle}")

    for key in ("must_include_any", "must_include_any_group_2", "contains_any"):
        needles = case.get(key) or []
        if needles and not contains_any(answer, needles):
            issues.append(f"missing_any:{'|'.join(needles)}")

    for needle in case.get("must_not_include") or []:
        if needle.lower() in low_a:
            issues.append(f"forbidden_claim:{needle}")

    return issues


def biek_target_terms(case: dict[str, Any]) -> list[str]:
    if case.get("target_terms"):
        return [str(item) for item in case.get("target_terms") if str(item).strip()]
    generic = {
        "biek",
        "official biek website",
        "not clearly confirmed",
        ".pdf",
        "page",
        "form",
        "forms",
        "resource",
        "document",
        "results",
        "result",
    }
    candidates = []
    for value in case.get("contains_any") or []:
        item = str(value).strip()
        if not item:
            continue
        low = item.lower()
        if low in generic or low.startswith("http"):
            continue
        if "%" in item:
            continue
        candidates.append(item)
    return candidates[:4]


def biek_question_prefers_link(question: str) -> bool:
    low = question.lower()
    return contains_any(
        low,
        (
            "download",
            "link",
            "page",
            "do you have",
            "where can i",
            "where do i",
            "form",
            "model paper",
            "scheme",
            "e-sheet",
            "mcq",
        ),
    )


def biek_question_prefers_summary(question: str) -> bool:
    low = question.lower()
    return contains_any(
        low,
        (
            "what is",
            "what are",
            "what does",
            "tell me about",
            "can u tell me about",
            "can you tell me about",
            "how can i contact",
            "what can i find under",
        ),
    )


def is_biek_ambiguous_result_question(question: str, answer: str) -> bool:
    low_q = question.lower()
    low_a = answer.lower()
    return (
        "roll number" in low_q
        and contains_any(low_q, ("result", "results"))
        and contains_any(
            low_a,
            (
                "multiple biek result records",
                "multiple result records",
                "add the year",
                "add a narrower combination",
                "narrow it to one exact result",
            ),
        )
    )


def is_biek_result_question(question: str) -> bool:
    low = question.lower()
    return "roll number" in low and contains_any(low, ("result", "results"))


def is_biek_download_forms_summary(question: str) -> bool:
    low = question.lower()
    return "download forms" in low and contains_any(low, ("what can i find under", "what can you find under"))


def has_biek_download_forms_listing(answer: str) -> bool:
    low = answer.lower()
    markers = (
        "certificate form",
        "scrutiny form",
        "provisional certification form",
        "migration form",
        "registration form",
        "verification",
        "duplicate",
        "examination forms",
        "permission forms",
    )
    hits = sum(1 for marker in markers if marker in low)
    return hits >= 4


def is_biek_form_resource_question(question: str) -> bool:
    low = question.lower()
    return contains_any(low, ("form", "forms", "pdf", "download link", "resource"))


def biek_quality_scores(question: str, answer: str, case: dict[str, Any], issues: list[str]) -> dict[str, Any]:
    low_a = answer.lower()
    refusal = is_refusal_answer(answer)
    wc = word_count(answer)
    sc = sentence_count(answer)
    target_terms = biek_target_terms(case)
    target_hit = any(term.lower() in low_a for term in target_terms) if target_terms else False
    link_present = has_link_like_text(answer)
    asks_link = biek_question_prefers_link(question)
    asks_summary = biek_question_prefers_summary(question)
    result_question = is_biek_result_question(question)
    ambiguous_result = is_biek_ambiguous_result_question(question, answer)
    download_forms_summary = is_biek_download_forms_summary(question)

    natural = 2
    if contains_any(low_a, BIEK_TEMPLATE_OPENERS):
        natural -= 1
    if (wc > 120 or sc > 5) and not ambiguous_result:
        natural -= 1
    if has_repetition(answer):
        natural -= 1
    if re.search(r"^\s*(?:\d+\.|-|\*)\s", answer.strip()):
        natural -= 1
    if ambiguous_result:
        natural = max(natural, 1)
    natural = max(0, natural)

    grounding = 2
    if refusal:
        grounding = 1
    if not refusal and target_terms and not target_hit:
        grounding -= 1
    if contains_any(low_a, BIEK_GENERIC_DRIFT_PATTERNS):
        grounding = 0
    grounding = max(0, grounding)

    summary = 2
    if asks_summary:
        if (wc > 110 or sc > 5) and not ambiguous_result:
            summary -= 1
        if answer.count("http") >= 2 or answer.count(".pdf") >= 2:
            summary -= 1
    else:
        if wc > 140 and not ambiguous_result:
            summary -= 1
    if contains_any(low_a, ("includes the following steps", "step-1", "step 1")) and "step" not in question.lower():
        summary = 0
    if ambiguous_result:
        summary = max(summary, 1)
    summary = max(0, summary)

    link_usefulness = 2
    if asks_link:
        if not link_present and not refusal:
            link_usefulness = 0
        elif not link_present and refusal:
            link_usefulness = 1
    else:
        if (answer.count("http") >= 2 or answer.count(".pdf") >= 2) and not result_question:
            link_usefulness = 0
        elif link_present:
            link_usefulness = 1
    if result_question and link_present:
        link_usefulness = max(link_usefulness, 1)
    if ambiguous_result and link_present:
        link_usefulness = max(link_usefulness, 2)
    if download_forms_summary and has_biek_download_forms_listing(answer):
        link_usefulness = max(link_usefulness, 1)
    if "missing_any" in " ".join(issues) and asks_link and not link_present:
        link_usefulness = 0

    safety = 2
    if has_prompt_leak(answer) or contains_any(low_a, BIEK_DOMAIN_LEAK_PATTERNS):
        safety = 0
    elif contains_any(low_a, BIEK_GENERIC_DRIFT_PATTERNS) or has_bad_contact_claim(answer):
        safety = min(safety, 1)
    if any(issue.startswith("forbidden_claim:") for issue in issues):
        safety = 0

    total = natural + grounding + summary + link_usefulness + safety
    quality_issues: list[str] = []
    if natural <= 0:
        quality_issues.append("unnatural_answer")
    if grounding <= 0:
        quality_issues.append("weak_target_grounding")
    if summary <= 0:
        quality_issues.append("answer_dump_not_summary")
    if link_usefulness <= 0:
        quality_issues.append("missing_or_unhelpful_linking")
    if safety <= 0:
        quality_issues.append("hallucination_or_domain_leak_risk")

    return {
        "natural": natural,
        "grounding": grounding,
        "summary": summary,
        "link_usefulness": link_usefulness,
        "safety": safety,
        "total": total,
        "target_terms": target_terms,
        "quality_issues": quality_issues,
    }


def evaluate_answer(
    question: str,
    answer: str,
    raw: dict[str, Any],
    knowledge_domain: str,
) -> dict[str, Any]:
    gold_manifest = load_gold_manifest_for_domain(knowledge_domain)
    gold_case = gold_manifest.get(normalize_text(question), {})
    manifest = load_v2_eval_manifest()
    meta = manifest.get(normalize_text(question), {})
    expected_behavior = gold_case.get("expected_behavior") or meta.get("expected_behavior", "")
    bucket = meta.get("bucket", "CUSTOM")
    low_q = question.lower()
    low_a = answer.lower()
    issues: list[str] = []
    verdict = "✅ Pass"

    if gold_case:
        bucket = "BIEK_GOLD" if is_biek_domain(knowledge_domain) else "DEMO_15_GOLD"
        issues.extend(gold_case_issues(gold_case, answer))

    if not answer.strip():
        issues.append("empty_answer")
    if has_prompt_leak(answer):
        issues.append("prompt_leak")
    if has_repetition(answer):
        issues.append("repetition")

    recruitment_question = contains_any(
        low_q,
        ("recruit", "recruitment", "hiring", "hire", "resume", "cv", "candidate"),
    ) and contains_any(low_q, ("product", "help", "best", "automation", "recommend"))
    if recruitment_question:
        if "irecruit" not in low_a:
            issues.append("wrong_product_recommendation:expected_iRecruit_One")
        if contains_any(low_a, ("cyber security automation", "coversaction", "opira")) and "irecruit" not in low_a:
            issues.append("wrong_product_recommendation")

    support_question = contains_any(
        low_q,
        ("customer support", "support conversation", "support conversations", "chatbot", "customer conversations"),
    )
    if support_question and "coversaction" not in low_a and "conversaction" not in low_a:
        issues.append("wrong_product_recommendation:expected_Coversaction_AI")

    private_question = contains_any(
        low_q,
        ("offline", "private infrastructure", "on-prem", "on premises", "private cloud", "behind firewall"),
    )
    if private_question and "opira" not in low_a:
        issues.append("wrong_product_recommendation:expected_Opira_AI")

    high_risk_question = contains_any(
        low_q,
        (
            "price",
            "pricing",
            "cost",
            "sla",
            "certification",
            "certifications",
            "compliance",
            "legal",
            "guarantee",
            "roadmap",
            "salesforce",
            "api documentation",
        ),
    )
    if is_biek_domain(knowledge_domain) and is_biek_form_resource_question(question):
        form_only_certification = contains_any(low_q, ("certification", "certificate")) and not contains_any(
            low_q,
            ("price", "pricing", "cost", "sla", "compliance", "legal", "guarantee", "roadmap", "salesforce"),
        )
        if form_only_certification:
            high_risk_question = False
    if high_risk_question:
        invented_specific = bool(re.search(r"\$\s?\d+|\b\d+\s?(?:%|percent)\b", answer))
        if invented_specific and not contains_any(low_q, ("50 percent", "50%")):
            issues.append("unsupported_specific_claim")
        if not is_refusal_answer(answer) and contains_any(
            low_q,
            ("price", "pricing", "cost", "sla", "certification", "compliance", "legal", "roadmap", "salesforce"),
        ):
            issues.append("under_refusal")

    contact_question = contains_any(low_q, ("contact", "purchase", "buy", "get started"))
    if contact_question and has_bad_contact_claim(answer):
        issues.append("bad_contact_claim")

    if bucket == "TRULY_UNKNOWN" and not is_refusal_answer(answer):
        issues.append("missing_unknown_caveat")
    if bucket == "POLICY_BEHAVIOR" and len(answer.split()) > 90:
        issues.append("too_verbose_policy_answer")

    observability = raw.get("observability") or {}
    if observability.get("used_refusal") and bucket in {"KNOWN_FROM_SITE", "GENERAL_OPS_REASONING"}:
        issues.append("possible_over_refusal")

    quality = None
    if is_biek_domain(knowledge_domain) and gold_case:
        quality = biek_quality_scores(question, answer, gold_case, issues)
        issues.extend(item for item in quality["quality_issues"] if item not in issues)

    if quality is not None:
        if issues or quality["total"] <= 5:
            verdict = "❌ Fail"
        elif quality["total"] <= 7 or bucket == "CUSTOM" or not expected_behavior:
            verdict = "⚠️ Partial"
        else:
            verdict = "✅ Pass"
    else:
        if issues:
            verdict = "❌ Fail"
        elif bucket == "CUSTOM" or not expected_behavior:
            verdict = "⚠️ Partial"

    return {
        "verdict": verdict,
        "issue": ", ".join(dict.fromkeys(issues)) if issues else "",
        "bucket": bucket,
        "expected_behavior": expected_behavior,
        "is_refusal": is_refusal_answer(answer),
        "quality": quality or {},
    }


def run_batch_eval(
    *,
    questions: list[str],
    model_config: dict[str, Any],
    mode_label: str,
    model_label: str,
    rag_generated: bool,
    combined_planning: bool,
    fast_rag: bool,
    knowledge_domain: str,
    temperature: float,
    max_tokens: int,
    progress_slot: Any | None = None,
    status_slot: Any | None = None,
    status_prefix: str = "",
) -> list[dict[str, Any]]:
    def progress_callback(index: int, total: int, question: str) -> None:
        prefix = f"{status_prefix} " if status_prefix else ""
        status.write(f"{prefix}Running {index}/{total}: {question}")
        progress.progress(index / total)

    rows: list[dict[str, Any]] = []
    progress = progress_slot or st.progress(0)
    status = status_slot or st.empty()
    rows, _ = run_batch_eval_core(
        questions=questions,
        model_config=model_config,
        mode_label=mode_label,
        model_label=model_label,
        rag_generated=rag_generated,
        combined_planning=combined_planning,
        fast_rag=fast_rag,
        knowledge_domain=knowledge_domain,
        temperature=temperature,
        max_tokens=max_tokens,
        progress_callback=progress_callback,
    )
    status.empty()
    return rows


def run_batch_eval_core(
    *,
    questions: list[str],
    model_config: dict[str, Any],
    mode_label: str,
    model_label: str,
    rag_generated: bool,
    combined_planning: bool,
    fast_rag: bool,
    knowledge_domain: str,
    temperature: float,
    max_tokens: int,
    progress_callback: Any | None = None,
    should_stop: Any | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    total = len(questions)
    stopped_early = False
    for index, question in enumerate(questions, start=1):
        if should_stop and should_stop():
            stopped_early = True
            break
        if progress_callback is not None:
            progress_callback(index, total, question)
        started = time.perf_counter()
        try:
            raw = answer_question(
                question=question,
                messages=[],
                model_config=model_config,
                answer_mode=ANSWER_MODES[mode_label],
                rag_generated=rag_generated,
                combined_planning=combined_planning,
                fast_rag=fast_rag,
                knowledge_domain=knowledge_domain,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raw = {
                "answer": f"Error: {exc}",
                "generation_model": None,
                "usage": {},
            }
        elapsed_s = time.perf_counter() - started
        raw["elapsed_s"] = round(elapsed_s, 3)
        answer = raw.get("answer", "")
        evaluation = evaluate_answer(question, answer, raw, knowledge_domain)
        confidence = retrieval_confidence(raw)
        prompt = prompt_metrics(raw)
        usage = raw.get("usage") or {}
        answer_usage = (
            usage.get("answer_generation")
            or usage.get("answer_retry")
            or usage.get("biek_target_answer")
            or usage.get("biek_canonical_answer")
            or usage.get("plain_generation")
            or {}
        )
        prompt_input_tokens = answer_usage.get("input_tokens")
        prompt_output_tokens = answer_usage.get("output_tokens")
        prompt_total_tokens = answer_usage.get("total_tokens")
        if prompt_total_tokens is None and (
            prompt_input_tokens is not None or prompt_output_tokens is not None
        ):
            prompt_total_tokens = (prompt_input_tokens or 0) + (prompt_output_tokens or 0)
        top_doc_ids = [str(doc_id) for doc_id in confidence.get("top_doc_ids") or []]
        expected_docs = expected_source_ids(question)
        expected_top1 = (
            top_doc_ids[0] in expected_docs
            if expected_docs and top_doc_ids else None
        )
        expected_top3 = (
            any(doc_id in expected_docs for doc_id in top_doc_ids[:3])
            if expected_docs and top_doc_ids else None
        )
        rows.append(
            {
                "question": question,
                "answer": answer,
                "verdict": evaluation["verdict"],
                "issue": evaluation["issue"],
                "bucket": evaluation["bucket"],
                "expected_behavior": evaluation["expected_behavior"],
                "quality_total": (evaluation.get("quality") or {}).get("total"),
                "quality_natural": (evaluation.get("quality") or {}).get("natural"),
                "quality_grounding": (evaluation.get("quality") or {}).get("grounding"),
                "quality_summary": (evaluation.get("quality") or {}).get("summary"),
                "quality_link_usefulness": (evaluation.get("quality") or {}).get("link_usefulness"),
                "quality_safety": (evaluation.get("quality") or {}).get("safety"),
                "latency_s": round(elapsed_s, 2),
                "generation_model": raw.get("generation_model"),
                "generation_model_display": resolved_model_display(
                    model_label,
                    mode_label,
                    raw.get("generation_model"),
                ) if raw.get("generation_model") else "quick bypass",
                "sources": ", ".join(source_rows(raw)),
                "selected_context_chunks": prompt.get("selected_context_chunks"),
                "user_prompt_chars": prompt.get("user_prompt_chars"),
                "evidence_chars": prompt.get("evidence_chars"),
                "structured_context_chars": prompt.get("structured_context_chars"),
                "input_tokens": prompt_input_tokens,
                "output_tokens": prompt_output_tokens,
                "total_tokens": prompt_total_tokens,
                "prompt_input_tokens": prompt_input_tokens,
                "prompt_output_tokens": prompt_output_tokens,
                "top_similarity": confidence.get("top_similarity"),
                "second_similarity": confidence.get("second_similarity"),
                "score_gap": confidence.get("score_gap"),
                "top_retrieved_docs": ", ".join(top_doc_ids),
                "expected_sources": ", ".join(expected_docs),
                "expected_source_top1": expected_top1,
                "expected_source_top3": expected_top3,
                "is_refusal": evaluation["is_refusal"],
                "debug": raw,
            }
        )
    return rows, stopped_early


def selected_plan_runs(
    *,
    model_labels: list[str],
    mode_labels: list[str],
) -> tuple[list[tuple[str, dict[str, Any], str]], list[str]]:
    runs: list[tuple[str, dict[str, Any], str]] = []
    skipped: list[str] = []
    for current_model_label in model_labels:
        model_config = MODEL_OPTIONS[current_model_label]
        for current_mode_label in mode_labels:
            if current_mode_label.startswith("Fine-tuned") and not model_config.get("fine_tune_exists"):
                skipped.append(f"{current_model_label} + {current_mode_label}")
                continue
            runs.append((current_model_label, model_config, current_mode_label))
    return runs, skipped


def run_automation_job(job: dict[str, Any]) -> None:
    try:
        total_runs = len(job["planned_runs"])
        for run_index, (queued_model_label, queued_config, queued_mode_label) in enumerate(job["planned_runs"], start=1):
            if job.get("stop_requested"):
                break
            job["current_run_index"] = run_index
            job["current_model_label"] = queued_model_label
            job["current_mode_label"] = queued_mode_label
            job["current_question_index"] = 0
            job["rows"] = []

            def progress_callback(index: int, total: int, question: str) -> None:
                job["current_question_index"] = index
                job["question_total"] = total
                job["current_question"] = question

            rows, stopped_early = run_batch_eval_core(
                questions=job["questions"],
                model_config=queued_config,
                mode_label=queued_mode_label,
                model_label=queued_model_label,
                rag_generated=job["rag_generated"],
                combined_planning=job["combined_planning"],
                fast_rag=job["fast_rag"],
                knowledge_domain=job["knowledge_domain"],
                temperature=job["temperature"],
                max_tokens=job["max_tokens"],
                progress_callback=progress_callback,
                should_stop=lambda: bool(job.get("stop_requested")),
            )
            job["rows"] = rows
            if rows:
                summary = summarize_batch(rows)
                history_entry = build_batch_history_entry(
                    rows=rows,
                    summary=summary,
                    model_label=queued_model_label,
                    mode_label=queued_mode_label,
                    rag_style=job["rag_style"],
                    combined_planning=job["combined_planning"],
                    fast_rag=job["fast_rag"],
                    temperature=job["temperature"],
                    max_tokens=job["max_tokens"],
                    retrieval_change_label=str(job.get("retrieval_change_label") or ""),
                )
                add_batch_history_entry(history_entry)
                job["suite_rows"].append(
                    {
                        "model": MODEL_OPTION_DISPLAY_NAMES.get(queued_model_label, queued_model_label),
                        "mode": queued_mode_label,
                        "score": summary["score"],
                        "pass": summary["pass"],
                        "fail": summary["fail"],
                        "partial": summary["partial"],
                        "avg_latency_s": summary["avg_latency_s"],
                        "history_id": history_entry["id"],
                    }
                )
            job["completed_runs"] = len(job["suite_rows"])
            if stopped_early:
                break
        job["done"] = True
        job["stopped"] = bool(job.get("stop_requested"))
    except Exception as exc:
        job["error"] = str(exc)
        job["done"] = True


def start_automation_job(
    *,
    planned_runs: list[tuple[str, dict[str, Any], str]],
    skipped_runs: list[str],
    questions: list[str],
    rag_style: str,
    combined_planning: bool,
    fast_rag: bool,
    knowledge_domain: str,
    knowledge_source_label: str,
    temperature: float,
    max_tokens: int,
    retrieval_change_label: str,
) -> dict[str, Any]:
    job = {
        "id": str(time.time_ns()),
        "planned_runs": [(label, dict(config), mode) for label, config, mode in planned_runs],
        "skipped_runs": list(skipped_runs),
        "questions": list(questions),
        "rag_style": rag_style,
        "rag_generated": rag_style == "Model-generated",
        "combined_planning": combined_planning,
        "fast_rag": fast_rag,
        "knowledge_domain": knowledge_domain,
        "knowledge_source_label": knowledge_source_label,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "retrieval_change_label": retrieval_change_label.strip(),
        "started_at": time.perf_counter(),
        "current_run_index": 0,
        "current_model_label": None,
        "current_mode_label": None,
        "current_question_index": 0,
        "question_total": len(questions),
        "current_question": "",
        "rows": [],
        "suite_rows": [],
        "completed_runs": 0,
        "stop_requested": False,
        "stopped": False,
        "done": False,
        "error": None,
    }
    thread = threading.Thread(target=run_automation_job, args=(job,), daemon=True)
    job["thread"] = thread
    thread.start()
    return job


def run_manual_batch_job(job: dict[str, Any]) -> None:
    try:
        def progress_callback(index: int, total: int, question: str) -> None:
            job["current_question_index"] = index
            job["question_total"] = total
            job["current_question"] = question

        rows, stopped_early = run_batch_eval_core(
            questions=job["questions"],
            model_config=job["model_config"],
            mode_label=job["mode_label"],
            model_label=job["model_label"],
            rag_generated=job["rag_generated"],
            combined_planning=job["combined_planning"],
            fast_rag=job["fast_rag"],
            knowledge_domain=job["knowledge_domain"],
            temperature=job["temperature"],
            max_tokens=job["max_tokens"],
            progress_callback=progress_callback,
            should_stop=lambda: bool(job.get("stop_requested")),
        )
        job["rows"] = rows
        job["done"] = True
        job["stopped"] = stopped_early or bool(job.get("stop_requested"))
    except Exception as exc:
        job["error"] = str(exc)
        job["done"] = True


def start_manual_batch_job(
    *,
    questions: list[str],
    model_config: dict[str, Any],
    mode_label: str,
    model_label: str,
    rag_style: str,
    combined_planning: bool,
    fast_rag: bool,
    knowledge_domain: str,
    knowledge_source_label: str,
    temperature: float,
    max_tokens: int,
    retrieval_change_label: str,
) -> dict[str, Any]:
    job = {
        "id": str(time.time_ns()),
        "questions": list(questions),
        "model_config": dict(model_config),
        "mode_label": mode_label,
        "model_label": model_label,
        "rag_style": rag_style,
        "rag_generated": rag_style == "Model-generated",
        "combined_planning": combined_planning,
        "fast_rag": fast_rag,
        "knowledge_domain": knowledge_domain,
        "knowledge_source_label": knowledge_source_label,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "retrieval_change_label": retrieval_change_label.strip(),
        "started_at": time.perf_counter(),
        "current_question_index": 0,
        "question_total": len(questions),
        "current_question": "",
        "rows": [],
        "stop_requested": False,
        "stopped": False,
        "done": False,
        "error": None,
    }
    thread = threading.Thread(target=run_manual_batch_job, args=(job,), daemon=True)
    job["thread"] = thread
    thread.start()
    return job


def summarize_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    pass_count = sum(1 for row in rows if row["verdict"].startswith("✅"))
    fail_count = sum(1 for row in rows if row["verdict"].startswith("❌"))
    partial_count = sum(1 for row in rows if row["verdict"].startswith("⚠️"))
    refusal_count = sum(1 for row in rows if row.get("is_refusal"))
    issue_counts: dict[str, int] = {}
    for row in rows:
        for issue in [part.strip() for part in row.get("issue", "").split(",") if part.strip()]:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    avg_latency = sum(float(row["latency_s"]) for row in rows) / total if total else 0.0
    input_token_rows = [row for row in rows if row.get("input_tokens") is not None]
    output_token_rows = [row for row in rows if row.get("output_tokens") is not None]
    total_token_rows = [row for row in rows if row.get("total_tokens") is not None]
    prompt_char_rows = [row for row in rows if row.get("user_prompt_chars") is not None]
    prompt_chunk_rows = [row for row in rows if row.get("selected_context_chunks") is not None]
    score = ((pass_count + (partial_count * 0.5)) / total * 100) if total else 0.0
    confidence_rows = [row for row in rows if row.get("top_similarity") is not None]
    avg_top_similarity = (
        sum(float(row["top_similarity"]) for row in confidence_rows) / len(confidence_rows)
        if confidence_rows else 0.0
    )
    avg_score_gap = (
        sum(float(row.get("score_gap") or 0.0) for row in confidence_rows) / len(confidence_rows)
        if confidence_rows else 0.0
    )
    pass_confidence = [
        float(row["top_similarity"])
        for row in confidence_rows
        if row["verdict"].startswith("✅")
    ]
    fail_confidence = [
        float(row["top_similarity"])
        for row in confidence_rows
        if row["verdict"].startswith("❌")
    ]
    expected_rows = [row for row in rows if row.get("expected_source_top1") is not None]
    expected_top1_count = sum(1 for row in expected_rows if row.get("expected_source_top1"))
    expected_top3_count = sum(1 for row in expected_rows if row.get("expected_source_top3"))
    high_confidence_failures = sum(
        1
        for row in confidence_rows
        if row["verdict"].startswith("❌") and float(row["top_similarity"]) >= 0.75
    )
    low_confidence_questions = sum(
        1 for row in confidence_rows if float(row["top_similarity"]) < 0.55
    )
    return {
        "total": total,
        "pass": pass_count,
        "fail": fail_count,
        "partial": partial_count,
        "avg_latency_s": round(avg_latency, 2),
        "refusal_count": refusal_count,
        "score": round(score, 1),
        "issue_counts": issue_counts,
        "prompt_metrics": {
            "avg_input_tokens": (
                round(
                    sum(float(row["input_tokens"]) for row in input_token_rows) / len(input_token_rows),
                    1,
                )
                if input_token_rows else None
            ),
            "avg_output_tokens": (
                round(
                    sum(float(row["output_tokens"]) for row in output_token_rows) / len(output_token_rows),
                    1,
                )
                if output_token_rows else None
            ),
            "avg_total_tokens": (
                round(
                    sum(float(row["total_tokens"]) for row in total_token_rows) / len(total_token_rows),
                    1,
                )
                if total_token_rows else None
            ),
            "avg_prompt_chars": (
                round(
                    sum(float(row["user_prompt_chars"]) for row in prompt_char_rows) / len(prompt_char_rows),
                    1,
                )
                if prompt_char_rows else None
            ),
            "avg_selected_context_chunks": (
                round(
                    sum(float(row["selected_context_chunks"]) for row in prompt_chunk_rows) / len(prompt_chunk_rows),
                    2,
                )
                if prompt_chunk_rows else None
            ),
        },
        "retrieval_confidence": {
            "question_count": len(confidence_rows),
            "avg_top_similarity": round(avg_top_similarity, 4),
            "avg_score_gap": round(avg_score_gap, 4),
            "pass_avg_similarity": round(sum(pass_confidence) / len(pass_confidence), 4) if pass_confidence else None,
            "fail_avg_similarity": round(sum(fail_confidence) / len(fail_confidence), 4) if fail_confidence else None,
            "expected_source_questions": len(expected_rows),
            "expected_source_top1_count": expected_top1_count,
            "expected_source_top1_pct": round(expected_top1_count / len(expected_rows) * 100, 1) if expected_rows else None,
            "expected_source_top3_count": expected_top3_count,
            "expected_source_top3_pct": round(expected_top3_count / len(expected_rows) * 100, 1) if expected_rows else None,
            "high_confidence_failures_075": high_confidence_failures,
            "low_confidence_questions_055": low_confidence_questions,
        },
    }


def score_style(score: float) -> tuple[str, str]:
    if score >= 85:
        return "#00beac", "Excellent"
    if score >= 70:
        return "#22c55e", "Good"
    if score >= 55:
        return "#f59e0b", "Needs work"
    return "#dc2626", "Poor"


def load_batch_history() -> list[dict[str, Any]]:
    if not BATCH_HISTORY_PATH.exists():
        return []
    try:
        payload = json.loads(BATCH_HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    return payload if isinstance(payload, list) else []


def batch_history_file_status() -> str:
    relative_path = BATCH_HISTORY_PATH.relative_to(PROJECT_ROOT)
    if not BATCH_HISTORY_PATH.exists():
        return f"Latest eval result file: `{relative_path}` is not created yet."
    modified_at = datetime.fromtimestamp(BATCH_HISTORY_PATH.stat().st_mtime).strftime("%b %d %H:%M")
    return f"Latest eval result file: `{relative_path}` · modified {modified_at}."


def save_batch_history(history: list[dict[str, Any]]) -> None:
    BATCH_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    BATCH_HISTORY_PATH.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def add_batch_history_entry(entry: dict[str, Any]) -> None:
    history = load_batch_history()
    history.insert(0, entry)
    save_batch_history(history[:50])


def delete_batch_history_entry(entry_id: str) -> None:
    history = [entry for entry in load_batch_history() if entry.get("id") != entry_id]
    save_batch_history(history)


def history_day_key(entry: dict[str, Any]) -> str:
    created_at = str(entry.get("created_at") or "").strip()
    if len(created_at) >= 10:
        return created_at[:10]
    return "Unknown date"


def group_history_by_day(history: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in history:
        grouped.setdefault(history_day_key(entry), []).append(entry)
    ordered_keys = sorted(grouped.keys(), reverse=True)
    return [(day_key, grouped[day_key]) for day_key in ordered_keys]


def history_day_label(day_key: str) -> str:
    if day_key == "Unknown date":
        return day_key
    try:
        day_value = datetime.strptime(day_key, "%Y-%m-%d").date()
    except ValueError:
        return day_key
    today = datetime.now().date()
    if day_value == today:
        return f"Today ({day_key})"
    if day_value == today.fromordinal(today.toordinal() - 1):
        return f"Yesterday ({day_key})"
    return day_value.strftime("%b %d, %Y")


def delete_batch_history_day(day_key: str) -> None:
    history = [entry for entry in load_batch_history() if history_day_key(entry) != day_key]
    save_batch_history(history)


def build_batch_history_entry(
    *,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    model_label: str,
    mode_label: str,
    rag_style: str,
    combined_planning: bool,
    fast_rag: bool,
    temperature: float,
    max_tokens: int,
    retrieval_change_label: str = "",
) -> dict[str, Any]:
    return {
        "id": str(time.time_ns()),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_label": model_label,
        "mode_label": mode_label,
        "rag_style": rag_style,
        "combined_planning": combined_planning,
        "fast_rag": fast_rag,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "retrieval_change_label": retrieval_change_label.strip(),
        "summary": summary,
        "results": rows,
    }


def rows_to_csv(rows: list[dict[str, Any]]) -> str:
    import csv
    import io

    output = io.StringIO()
    fields = [
        "question",
        "answer",
        "verdict",
        "issue",
        "bucket",
        "expected_behavior",
        "quality_total",
        "quality_natural",
        "quality_grounding",
        "quality_summary",
        "quality_link_usefulness",
        "quality_safety",
        "latency_s",
        "generation_model",
        "generation_model_display",
        "sources",
        "top_similarity",
        "second_similarity",
        "score_gap",
        "top_retrieved_docs",
        "expected_sources",
        "expected_source_top1",
        "expected_source_top3",
        "is_refusal",
        "selected_context_chunks",
        "user_prompt_chars",
        "evidence_chars",
        "structured_context_chars",
        "prompt_input_tokens",
        "prompt_output_tokens",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return output.getvalue()


def render_history_card(entry: dict[str, Any]) -> None:
    summary = entry.get("summary") or {}
    confidence = summary.get("retrieval_confidence") or {}
    score = float(summary.get("score") or 0.0)
    color, label = score_style(score)
    entry_id = str(entry.get("id") or "")
    rag_label = str(entry.get("rag_style") or "Unknown RAG")
    model_key = history_model_key(entry)
    generation_model = None if " | " in model_key else model_key
    model_display = resolved_model_display(
        str(entry.get("model_label") or "Unknown model"),
        str(entry.get("mode_label") or "Unknown mode"),
        generation_model,
    )
    title = f"{model_display} | {rag_label}"
    config_line = (
        f"Combined planning: {'on' if entry.get('combined_planning') else 'off'} | "
        f"Fast RAG: {'on' if entry.get('fast_rag') else 'off'} | "
        f"Temp: {entry.get('temperature')} | Max tokens: {entry.get('max_tokens')}"
    )
    retrieval_change_label = str(entry.get("retrieval_change_label") or "").strip()
    prompt_summary = summary.get("prompt_metrics") or {}
    with st.container(border=True):
        left, right = st.columns([5, 1.5])
        with left:
            st.markdown(f"**{title}**")
            if entry.get("_overall_model_rank"):
                overall = entry.get("_overall_model_metrics") or {}
                consistency = overall.get("consistency")
                st.caption(
                    f"Overall model rank #{entry['_overall_model_rank']} · "
                    f"composite {entry.get('_overall_model_score', 0):.1f} · "
                    f"avg quality {overall.get('average_quality', 0):.1f} · "
                    f"strict accuracy {overall.get('strict_accuracy', 0):.1f}% · "
                    f"consistency {f'{consistency:.1f}' if consistency is not None else 'insufficient'}"
                )
            st.caption(entry.get("created_at", ""))
            st.caption(config_line)
            if retrieval_change_label:
                st.caption(f"Retrieval change: {retrieval_change_label}")
            st.markdown(
                f"Total `{summary.get('total', 0)}` · Pass `{summary.get('pass', 0)}` · "
                f"Partial `{summary.get('partial', 0)}` · Fail `{summary.get('fail', 0)}` · "
                f"Avg `{summary.get('avg_latency_s', 0)}s` · Refusals `{summary.get('refusal_count', 0)}`"
            )
            if confidence.get("question_count"):
                st.caption(
                    f"Retrieval: avg similarity {confidence.get('avg_top_similarity')} · "
                    f"avg gap {confidence.get('avg_score_gap')} · "
                    f"expected source top-3 {confidence.get('expected_source_top3_pct')}%"
                )
            if prompt_summary:
                st.caption(
                    f"Prompt: avg input {prompt_summary.get('avg_input_tokens')} · "
                    f"avg output {prompt_summary.get('avg_output_tokens')} · "
                    f"avg total {prompt_summary.get('avg_total_tokens')} · "
                    f"avg prompt chars {prompt_summary.get('avg_prompt_chars')} · "
                    f"avg context chunks {prompt_summary.get('avg_selected_context_chunks')}"
                )
        with right:
            st.markdown(
                f"""
                <div style="
                    background:{color};
                    color:white;
                    border-radius:999px;
                    padding:8px 12px;
                    font-weight:800;
                    text-align:center;
                    margin-bottom:10px;
                    white-space:nowrap;
                ">{score:.1f}% {label}</div>
                """,
                unsafe_allow_html=True,
            )
            st.download_button(
                "JSON",
                data=json.dumps(entry, ensure_ascii=False, indent=2),
                file_name=f"batch_eval_{entry_id}.json",
                mime="application/json",
                key=f"download_history_json_{entry_id}",
                use_container_width=True,
            )
            if st.button("Delete", key=f"delete_history_{entry_id}", use_container_width=True):
                delete_batch_history_entry(entry_id)
                st.rerun()

        issue_counts = summary.get("issue_counts") or {}
        if issue_counts:
            st.caption(
                "Issues: "
                + ", ".join(f"{issue}: {count}" for issue, count in issue_counts.items())
            )

        with st.expander("View run details"):
            detail_rows = [
                {key: value for key, value in row.items() if key != "debug"}
                for row in entry.get("results", [])
            ]
            st.dataframe(detail_rows, use_container_width=True, hide_index=True)


def history_report_rows(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in history:
        summary = entry.get("summary") or {}
        total = int(summary.get("total") or 0)
        correct = int(summary.get("pass") or 0)
        partial = int(summary.get("partial") or 0)
        wrong = int(summary.get("fail") or 0)
        issue_counts = summary.get("issue_counts") or {}
        confidence = summary.get("retrieval_confidence") or {}
        prompt_summary = summary.get("prompt_metrics") or {}
        rows.append(
            {
                "created_at": entry.get("created_at", ""),
                "model": entry.get("model_label", ""),
                "mode": entry.get("mode_label", ""),
                "retrieval_change_label": entry.get("retrieval_change_label", ""),
                "rag_behavior": entry.get("rag_style", ""),
                "combined_planning": "on" if entry.get("combined_planning") else "off",
                "fast_rag": "on" if entry.get("fast_rag") else "off",
                "temperature": entry.get("temperature"),
                "max_tokens": entry.get("max_tokens"),
                "score": float(summary.get("score") or 0.0),
                "total": total,
                "correct": correct,
                "partial": partial,
                "wrong": wrong,
                "accuracy_pct": round(correct / total * 100, 1) if total else 0.0,
                "non_fail_pct": round((correct + partial) / total * 100, 1) if total else 0.0,
                "avg_latency_s": float(summary.get("avg_latency_s") or 0.0),
                "refusals": int(summary.get("refusal_count") or 0),
                "avg_top_similarity": confidence.get("avg_top_similarity"),
                "avg_score_gap": confidence.get("avg_score_gap"),
                "pass_avg_similarity": confidence.get("pass_avg_similarity"),
                "fail_avg_similarity": confidence.get("fail_avg_similarity"),
                "expected_source_top1_pct": confidence.get("expected_source_top1_pct"),
                "expected_source_top3_pct": confidence.get("expected_source_top3_pct"),
                "high_confidence_failures_075": confidence.get("high_confidence_failures_075", 0),
                "low_confidence_questions_055": confidence.get("low_confidence_questions_055", 0),
                "avg_input_tokens": prompt_summary.get("avg_input_tokens"),
                "avg_output_tokens": prompt_summary.get("avg_output_tokens"),
                "avg_total_tokens": prompt_summary.get("avg_total_tokens"),
                "avg_prompt_chars": prompt_summary.get("avg_prompt_chars"),
                "avg_selected_context_chunks": prompt_summary.get("avg_selected_context_chunks"),
                "top_issues": ", ".join(
                    f"{issue}:{count}"
                    for issue, count in sorted(
                        issue_counts.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:4]
                ),
            }
        )
    return rows


def pick_report_row(rows: list[dict[str, Any]], key: str, *, highest: bool = True) -> dict[str, Any] | None:
    if not rows:
        return None
    return sorted(rows, key=lambda row: float(row.get(key) or 0.0), reverse=highest)[0]


def report_metric(label: str, row: dict[str, Any] | None, key: str, suffix: str = "") -> None:
    if not row:
        st.metric(label, "N/A")
        return
    st.metric(label, f"{row.get(key)}{suffix}")
    model_col, info_col = st.columns([0.86, 0.14], vertical_alignment="center")
    with model_col:
        st.caption(f"{row.get('model')} | {row.get('mode')} | {row.get('rag_behavior')}")
        if row.get("retrieval_change_label"):
            st.caption(f"Retrieval change: {row.get('retrieval_change_label')}")
    with info_col:
        with st.popover("ⓘ", help="Show the complete metrics for this evaluation run"):
            st.markdown(f"**{row.get('model') or 'Unknown model'}**")
            st.caption(
                f"{row.get('mode') or 'Unknown mode'} | "
                f"{row.get('rag_behavior') or 'Unknown RAG behavior'}"
            )
            st.markdown(
                f"""
- **Score:** {row.get('score', 0)}%
- **Strict accuracy:** {row.get('accuracy_pct', 0)}%
- **Non-fail rate:** {row.get('non_fail_pct', 0)}%
- **Average time:** {row.get('avg_latency_s', 0)}s
- **Results:** {row.get('correct', 0)} pass, {row.get('partial', 0)} partial, {row.get('wrong', 0)} fail
- **Refusals:** {row.get('refusals', 0)}
- **Questions:** {row.get('total', 0)}
- **Average retrieval similarity:** {row.get('avg_top_similarity')}
- **Average retrieval gap:** {row.get('avg_score_gap')}
- **Expected source top-1:** {row.get('expected_source_top1_pct')}%
- **Expected source top-3:** {row.get('expected_source_top3_pct')}%
- **Average input tokens:** {row.get('avg_input_tokens')}
- **Average output tokens:** {row.get('avg_output_tokens')}
- **Average total tokens:** {row.get('avg_total_tokens')}
- **High-confidence failures (≥0.75):** {row.get('high_confidence_failures_075', 0)}
- **Low-confidence questions (<0.55):** {row.get('low_confidence_questions_055', 0)}
- **Combined planning:** {row.get('combined_planning', 'off')}
- **Fast RAG:** {row.get('fast_rag', 'off')}
- **Temperature:** {row.get('temperature')}
- **Max tokens:** {row.get('max_tokens')}
                """
            )
            if row.get("top_issues"):
                st.caption(f"Top issues: {row['top_issues']}")


def rows_to_csv_report(rows: list[dict[str, Any]]) -> str:
    import csv
    import io

    output = io.StringIO()
    fields = [
        "created_at",
        "model",
        "mode",
        "retrieval_change_label",
        "rag_behavior",
        "combined_planning",
        "fast_rag",
        "temperature",
        "max_tokens",
        "score",
        "total",
        "correct",
        "partial",
        "wrong",
        "accuracy_pct",
        "non_fail_pct",
        "avg_latency_s",
        "refusals",
        "avg_top_similarity",
        "avg_score_gap",
        "pass_avg_similarity",
        "fail_avg_similarity",
        "expected_source_top1_pct",
        "expected_source_top3_pct",
        "high_confidence_failures_075",
        "low_confidence_questions_055",
        "avg_input_tokens",
        "avg_output_tokens",
        "avg_total_tokens",
        "avg_prompt_chars",
        "avg_selected_context_chunks",
        "top_issues",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return output.getvalue()


def render_history_report(history: list[dict[str, Any]]) -> None:
    rows = history_report_rows(history)
    if not rows:
        st.info("No history available for report.")
        return

    st.markdown("### Report")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        report_metric("Highest score", pick_report_row(rows, "score"), "score", "%")
    with c2:
        report_metric("Fastest avg time", pick_report_row(rows, "avg_latency_s", highest=False), "avg_latency_s", "s")
    with c3:
        report_metric("Most correct", pick_report_row(rows, "correct"), "correct")
    with c4:
        report_metric("Lowest score", pick_report_row(rows, "score", highest=False), "score", "%")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        report_metric("Most wrong", pick_report_row(rows, "wrong"), "wrong")
    with c6:
        report_metric("Best strict accuracy", pick_report_row(rows, "accuracy_pct"), "accuracy_pct", "%")
    with c7:
        report_metric("Best non-fail rate", pick_report_row(rows, "non_fail_pct"), "non_fail_pct", "%")
    with c8:
        report_metric("Most refusals", pick_report_row(rows, "refusals"), "refusals")

    confidence_rows = [row for row in rows if row.get("avg_top_similarity") is not None]
    if confidence_rows:
        source_accuracy_rows = [
            row for row in confidence_rows
            if row.get("expected_source_top1_pct") is not None
        ]
        c9, c10, c11, c12 = st.columns(4)
        with c9:
            report_metric(
                "Highest avg similarity",
                pick_report_row(confidence_rows, "avg_top_similarity"),
                "avg_top_similarity",
            )
        with c10:
            report_metric(
                "Best source top-1",
                pick_report_row(source_accuracy_rows, "expected_source_top1_pct"),
                "expected_source_top1_pct",
                "%",
            )
        with c11:
            report_metric(
                "Best source top-3",
                pick_report_row(source_accuracy_rows, "expected_source_top3_pct"),
                "expected_source_top3_pct",
                "%",
            )
        with c12:
            report_metric(
                "Fewest confident failures",
                pick_report_row(confidence_rows, "high_confidence_failures_075", highest=False),
                "high_confidence_failures_075",
            )

    st.dataframe(rows, use_container_width=True, hide_index=True)

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_count": len(rows),
        "rows": rows,
    }
    st.download_button(
        "Download report JSON",
        data=json.dumps(payload, ensure_ascii=False, indent=2),
        file_name="batch_eval_history_report.json",
        mime="application/json",
    )
    st.download_button(
        "Download report CSV",
        data=rows_to_csv_report(rows),
        file_name="batch_eval_history_report.csv",
        mime="text/csv",
    )


def history_model_key(entry: dict[str, Any]) -> str:
    counts: dict[str, int] = {}
    for row in entry.get("results") or []:
        model = str(row.get("generation_model") or "").strip()
        if model:
            counts[model] = counts.get(model, 0) + 1
    if counts:
        return max(counts, key=counts.get)
    return f"{entry.get('model_label', 'Unknown model')} | {entry.get('mode_label', 'Unknown mode')}"


def history_model_filter_label(entry: dict[str, Any]) -> str:
    model_key = history_model_key(entry)
    if model_key in {"synapse-llama3-v1", "synapse-llama3-v1-base"}:
        return "Synapse Llama V1 3B"
    if model_key == "llama3:latest":
        return "Synapse Llama V1 8B"
    if model_key == "synapse-1.5b-v1":
        return "Synapse Qwen 2.5 1.5B V1"
    if model_key == "synapse-1.5b-v2":
        return "Synapse Qwen 2.5 1.5B V2"
    if model_key in {"qwen2.5:3b", "synapse-qwen2.5-3b-v1"}:
        return "Synapse Qwen 2.5 3B"

    saved_label = str(entry.get("model_label") or "Unknown model")
    return MODEL_OPTION_DISPLAY_NAMES.get(saved_label, saved_label)


def rank_history_by_overall_model(
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in history:
        grouped.setdefault(history_model_key(entry), []).append(entry)

    aggregates: dict[str, dict[str, Any]] = {}
    for model_key, entries in grouped.items():
        run_scores: list[float] = []
        latencies: list[float] = []
        all_rows: list[dict[str, Any]] = []
        source_top3_count = 0
        source_question_count = 0
        comparable_runs: dict[tuple[str, ...], list[float]] = {}
        for entry in entries:
            summary = entry.get("summary") or {}
            confidence = summary.get("retrieval_confidence") or {}
            score = float(summary.get("score") or 0.0)
            run_scores.append(score)
            rows = list(entry.get("results") or [])
            all_rows.extend(rows)
            signature = tuple(normalize_text(str(row.get("question") or "")) for row in rows)
            if signature:
                comparable_runs.setdefault(signature, []).append(score)
            source_top3_count += int(confidence.get("expected_source_top3_count") or 0)
            source_question_count += int(confidence.get("expected_source_questions") or 0)
            latency = float(summary.get("avg_latency_s") or 0.0)
            if latency > 0:
                latencies.append(latency)

        total_questions = len(all_rows)
        passed_questions = sum(
            1 for row in all_rows if str(row.get("verdict") or "").startswith("✅")
        )
        refusal_markers = (
            "over_refusal",
            "missing_required_refusal",
            "under_refusal",
            "possible_over_refusal",
            "missing_unknown_caveat",
        )
        safety_markers = (
            "prompt_leak",
            "unsupported_specific_claim",
            "bad_contact_claim",
            "repetition",
            "empty_answer",
        )
        refusal_errors = sum(
            1
            for row in all_rows
            if any(marker in str(row.get("issue") or "") for marker in refusal_markers)
        )
        safety_errors = sum(
            1
            for row in all_rows
            if any(marker in str(row.get("issue") or "") for marker in safety_markers)
        )
        repeated_scores = max(comparable_runs.values(), key=len, default=[])
        consistency_score: float | None = None
        if len(repeated_scores) >= 2:
            cohort_mean = sum(repeated_scores) / len(repeated_scores)
            variance = sum((score - cohort_mean) ** 2 for score in repeated_scores) / len(repeated_scores)
            consistency_score = max(0.0, 100.0 - (variance ** 0.5 * 2.0))

        aggregates[model_key] = {
            "average_quality": sum(run_scores) / len(run_scores) if run_scores else 0.0,
            "strict_accuracy": passed_questions / total_questions * 100.0 if total_questions else 0.0,
            "source_top3_accuracy": (
                source_top3_count / source_question_count * 100.0
                if source_question_count else None
            ),
            "consistency": consistency_score,
            "consistency_runs": len(repeated_scores),
            "average_latency": sum(latencies) / len(latencies) if latencies else 0.0,
            "refusal_correctness": (
                (1.0 - refusal_errors / total_questions) * 100.0 if total_questions else 0.0
            ),
            "safety_error_control": (
                (1.0 - safety_errors / total_questions) * 100.0 if total_questions else 0.0
            ),
            "run_count": len(entries),
        }

    valid_latencies = [
        metrics["average_latency"]
        for metrics in aggregates.values()
        if metrics["average_latency"] > 0
    ]
    fastest_overall = min(valid_latencies) if valid_latencies else 0.0
    for metrics in aggregates.values():
        latency = metrics["average_latency"]
        speed_score = min(100.0, fastest_overall / latency * 100.0) if latency else 0.0
        metrics["speed_score"] = speed_score
        source_score = metrics["source_top3_accuracy"]
        consistency_score = metrics["consistency"]
        # Unknown metrics receive a neutral score instead of a free perfect score.
        metrics["overall_score"] = (
            metrics["average_quality"] * 0.35
            + metrics["strict_accuracy"] * 0.20
            + (source_score if source_score is not None else 50.0) * 0.15
            + (consistency_score if consistency_score is not None else 50.0) * 0.10
            + speed_score * 0.10
            + metrics["refusal_correctness"] * 0.05
            + metrics["safety_error_control"] * 0.05
        )

    ordered_models = sorted(
        aggregates,
        key=lambda key: (
            aggregates[key]["overall_score"],
            aggregates[key]["average_quality"],
            -aggregates[key]["average_latency"],
        ),
        reverse=True,
    )
    ranks = {model_key: index for index, model_key in enumerate(ordered_models, start=1)}
    ranked_history: list[dict[str, Any]] = []
    for entry in history:
        copied = dict(entry)
        model_key = history_model_key(entry)
        copied["_overall_model_rank"] = ranks[model_key]
        copied["_overall_model_score"] = aggregates[model_key]["overall_score"]
        copied["_overall_model_metrics"] = aggregates[model_key]
        ranked_history.append(copied)
    ranked_history.sort(
        key=lambda entry: (
            int(entry["_overall_model_rank"]),
            -float((entry.get("summary") or {}).get("score") or 0.0),
        )
    )
    return ranked_history


def overall_model_metrics(history: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for entry in rank_history_by_overall_model(history):
        model_key = history_model_key(entry)
        if model_key not in metrics:
            metrics[model_key] = dict(entry.get("_overall_model_metrics") or {})
            metrics[model_key]["rank"] = entry.get("_overall_model_rank")
            metrics[model_key]["overall_score"] = entry.get("_overall_model_score", 0.0)
            metrics[model_key]["display_name"] = resolved_model_display(
                str(entry.get("model_label") or "Unknown model"),
                str(entry.get("mode_label") or "Unknown mode"),
                None if " | " in model_key else model_key,
            )
    return metrics


def history_entry_metrics(entry: dict[str, Any]) -> dict[str, Any]:
    summary = entry.get("summary") or {}
    confidence = summary.get("retrieval_confidence") or {}
    rows = list(entry.get("results") or [])
    total_questions = len(rows)
    refusal_markers = (
        "over_refusal",
        "missing_required_refusal",
        "under_refusal",
        "possible_over_refusal",
        "missing_unknown_caveat",
    )
    safety_markers = (
        "prompt_leak",
        "unsupported_specific_claim",
        "bad_contact_claim",
        "repetition",
        "empty_answer",
    )
    refusal_errors = sum(
        1
        for row in rows
        if any(marker in str(row.get("issue") or "") for marker in refusal_markers)
    )
    safety_errors = sum(
        1
        for row in rows
        if any(marker in str(row.get("issue") or "") for marker in safety_markers)
    )
    model_key = history_model_key(entry)
    display_name = resolved_model_display(
        str(entry.get("model_label") or "Unknown model"),
        str(entry.get("mode_label") or "Unknown mode"),
        None if " | " in model_key else model_key,
    )
    return {
        "overall_score": float(summary.get("score") or 0.0),
        "average_quality": float(summary.get("score") or 0.0),
        "strict_accuracy": (
            float(summary.get("pass") or 0.0) / float(summary.get("total") or 1.0) * 100.0
            if summary.get("total") else 0.0
        ),
        "source_top3_accuracy": confidence.get("expected_source_top3_pct"),
        "consistency": None,
        "average_latency": float(summary.get("avg_latency_s") or 0.0),
        "refusal_correctness": (
            (1.0 - refusal_errors / total_questions) * 100.0 if total_questions else 0.0
        ),
        "safety_error_control": (
            (1.0 - safety_errors / total_questions) * 100.0 if total_questions else 0.0
        ),
        "run_count": 1,
        "display_name": display_name,
        "created_at": str(entry.get("created_at") or ""),
        "mode_label": str(entry.get("mode_label") or ""),
        "rag_style": str(entry.get("rag_style") or ""),
        "retrieval_change_label": str(entry.get("retrieval_change_label") or ""),
    }


def history_entry_option_label(entry: dict[str, Any]) -> str:
    summary = entry.get("summary") or {}
    metrics = history_entry_metrics(entry)
    retrieval_change_label = metrics.get("retrieval_change_label") or ""
    suffix = f" | {retrieval_change_label}" if retrieval_change_label else ""
    return (
        f"{metrics['display_name']} | {metrics['mode_label']} | "
        f"{metrics['created_at']} | {float(summary.get('score') or 0.0):.1f}%{suffix}"
    )


def comparison_value(value: Any, *, suffix: str = "") -> str:
    if value is None:
        return "Insufficient data"
    if isinstance(value, float):
        return f"{value:.1f}{suffix}"
    return f"{value}{suffix}"


def comparison_card_html(
    model: str,
    metrics: dict[str, Any],
    other_metrics: dict[str, Any],
    *,
    winner: bool,
    tie: bool,
) -> str:
    if tie:
        status = "Tied overall"
    elif winner:
        status = "Better overall"
    else:
        status = "Lower overall"
    rows = [
        ("Overall score", "overall_score", "", True),
        ("Average quality", "average_quality", "", True),
        ("Strict accuracy", "strict_accuracy", "%", True),
        ("Source top-3", "source_top3_accuracy", "%", True),
        ("Consistency", "consistency", "", True),
        ("Average latency", "average_latency", "s", False),
        ("Refusal correctness", "refusal_correctness", "%", True),
        ("Safety/error control", "safety_error_control", "%", True),
        ("Saved runs", "run_count", "", None),
    ]
    metric_rows: list[str] = []
    for label, key, suffix, higher_better in rows:
        value = metrics.get(key)
        other_value = other_metrics.get(key)
        row_background = "rgba(148,163,184,.06)"
        row_border = "rgba(148,163,184,.16)"
        if higher_better is not None and value is not None and other_value is not None:
            difference = float(value) - float(other_value)
            if abs(difference) >= 0.0001:
                is_better = difference > 0 if higher_better else difference < 0
                if is_better:
                    row_background = "linear-gradient(90deg, rgba(34,197,94,.18), rgba(16,185,129,.05))"
                    row_border = "rgba(34,197,94,.36)"
                else:
                    row_background = "linear-gradient(90deg, rgba(248,113,113,.17), rgba(239,68,68,.05))"
                    row_border = "rgba(248,113,113,.34)"
        metric_rows.append(
            f'<div style="display:flex;justify-content:space-between;gap:16px;padding:8px 10px;'
            f'margin:5px 0;border:1px solid {row_border};border-radius:9px;background:{row_background};">'
            f'<span style="color:#cbd5e1;">{html.escape(label)}</span>'
            f'<strong>{html.escape(comparison_value(value, suffix=suffix))}</strong></div>'
        )
    return f"""
    <div style="background:rgba(15,23,42,.28);border:1px solid rgba(148,163,184,.24);border-radius:16px;padding:20px;min-height:500px;">
        <div style="font-size:12px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#94a3b8;">{status}</div>
        <div style="font-size:22px;font-weight:800;margin:6px 0 14px;">{html.escape(model)}</div>
        {''.join(metric_rows)}
    </div>
    """


def model_comparison_remarks(
    model_a: str,
    metrics_a: dict[str, Any],
    model_b: str,
    metrics_b: dict[str, Any],
) -> str:
    score_a = float(metrics_a.get("overall_score") or 0.0)
    score_b = float(metrics_b.get("overall_score") or 0.0)
    if abs(score_a - score_b) < 0.05:
        return f"{model_a} and {model_b} are effectively tied overall. Compare latency and failure details before choosing."
    winner, winner_metrics = (model_a, metrics_a) if score_a > score_b else (model_b, metrics_b)
    loser, loser_metrics = (model_b, metrics_b) if score_a > score_b else (model_a, metrics_a)
    leads: list[str] = []
    for label, key in (
        ("average quality", "average_quality"),
        ("strict accuracy", "strict_accuracy"),
        ("source retrieval", "source_top3_accuracy"),
        ("consistency", "consistency"),
        ("refusal handling", "refusal_correctness"),
        ("safety/error control", "safety_error_control"),
    ):
        winner_value = winner_metrics.get(key)
        loser_value = loser_metrics.get(key)
        if winner_value is not None and loser_value is not None and float(winner_value) > float(loser_value):
            leads.append(label)
    winner_latency = float(winner_metrics.get("average_latency") or 0.0)
    loser_latency = float(loser_metrics.get("average_latency") or 0.0)
    speed_note = ""
    if winner_latency and loser_latency and winner_latency > loser_latency:
        speed_note = f" {loser} is faster on average, so quality versus latency remains the main tradeoff."
    lead_text = ", ".join(leads[:3]) if leads else "the combined weighted metrics"
    difference = abs(score_a - score_b)
    return (
        f"{winner} ranks higher overall by {difference:.1f} points, mainly through {lead_text}."
        f"{speed_note}"
    )


def render_model_comparison(history: list[dict[str, Any]]) -> None:
    comparison_mode = st.segmented_control(
        "Comparison scope",
        ["Family aggregate", "Saved runs"],
        default="Family aggregate",
        key="comparison_scope_mode",
        width="stretch",
    )
    if comparison_mode == "Saved runs":
        if not history:
            st.info("Run Batch Eval to add saved runs for comparison.")
            return
        if len(history) == 1:
            selector_a, selector_b = st.columns(2)
            with selector_a:
                st.selectbox(
                    "Run A",
                    [str(history[0].get("id") or "only-run")],
                    disabled=True,
                    format_func=lambda _: history_entry_option_label(history[0]),
                    key="compare_run_a_single",
                )
            with selector_b:
                st.selectbox("Run B", ["No second run"], disabled=True, key="compare_run_b_single")
            st.info("Save a second run in this day block to compare run instances.")
            return
        run_by_id = {str(entry.get("id") or ""): entry for entry in history if entry.get("id")}
        run_ids = list(run_by_id)
        selector_a, selector_b = st.columns(2)
        with selector_a:
            run_a_id = st.selectbox(
                "Run A",
                run_ids,
                index=0,
                key="compare_run_a",
                format_func=lambda run_id: history_entry_option_label(run_by_id[run_id]),
            )
        with selector_b:
            default_run_b_index = 1 if len(run_ids) > 1 else 0
            run_b_id = st.selectbox(
                "Run B",
                run_ids,
                index=default_run_b_index,
                key="compare_run_b",
                format_func=lambda run_id: history_entry_option_label(run_by_id[run_id]),
            )
        if run_a_id == run_b_id:
            st.warning("Select two different saved runs to compare.")
            return
        entry_a = run_by_id[run_a_id]
        entry_b = run_by_id[run_b_id]
        metrics_a = history_entry_metrics(entry_a)
        metrics_b = history_entry_metrics(entry_b)
        display_a = history_entry_option_label(entry_a)
        display_b = history_entry_option_label(entry_b)
        score_a = float(metrics_a.get("overall_score") or 0.0)
        score_b = float(metrics_b.get("overall_score") or 0.0)
        tie = abs(score_a - score_b) < 0.05
        card_a, card_b = st.columns(2)
        with card_a:
            st.markdown(
                comparison_card_html(
                    display_a,
                    metrics_a,
                    metrics_b,
                    winner=score_a > score_b,
                    tie=tie,
                ),
                unsafe_allow_html=True,
            )
        with card_b:
            st.markdown(
                comparison_card_html(
                    display_b,
                    metrics_b,
                    metrics_a,
                    winner=score_b > score_a,
                    tie=tie,
                ),
                unsafe_allow_html=True,
            )
        st.markdown("**Final remarks**")
        st.info(model_comparison_remarks(display_a, metrics_a, display_b, metrics_b))
        return

    metrics_by_model = overall_model_metrics(history)
    models = sorted(
        metrics_by_model,
        key=lambda model: float(metrics_by_model[model].get("overall_score") or 0.0),
        reverse=True,
    )
    if not models:
        selector_a, selector_b = st.columns(2)
        with selector_a:
            st.selectbox("Model A", ["0 models"], disabled=True, key="compare_model_a_empty")
        with selector_b:
            st.selectbox("Model B", ["0 models"], disabled=True, key="compare_model_b_empty")
        st.info("Run Batch Eval to add models to comparison history.")
        return
    if len(models) == 1:
        selector_a, selector_b = st.columns(2)
        with selector_a:
            st.selectbox(
                "Model A",
                models,
                disabled=True,
                key="compare_model_a_single",
                format_func=lambda model: metrics_by_model[model].get("display_name", model),
            )
        with selector_b:
            st.selectbox("Model B", ["No second model"], disabled=True, key="compare_model_b_single")
        st.info("Evaluate a second generation model to enable comparison.")
        return

    family_heading_prefix = "__model_family_heading__:"
    grouped_options: list[str] = []
    first_model_by_heading: dict[str, str] = {}
    for family in ("Qwen", "Llama", "Gemma"):
        family_models = [
            model
            for model in models
            if model_family(
                model,
                str(metrics_by_model[model].get("display_name") or model),
            ) == family
        ]
        if not family_models:
            continue
        heading = f"{family_heading_prefix}{family}"
        grouped_options.append(heading)
        grouped_options.extend(family_models)
        first_model_by_heading[heading] = family_models[0]

    def grouped_model_label(option: str) -> str:
        if option.startswith(family_heading_prefix):
            family = option.removeprefix(family_heading_prefix)
            return f"──────── {family.upper()} ────────"
        display_name = metrics_by_model[option].get("display_name", option)
        return f"    {display_name}"

    def select_first_model_for_heading(widget_key: str) -> None:
        selected = st.session_state.get(widget_key)
        if selected in first_model_by_heading:
            st.session_state[widget_key] = first_model_by_heading[selected]

    selector_a, selector_b = st.columns(2)
    with selector_a:
        model_a = st.selectbox(
            "Model A",
            grouped_options,
            index=grouped_options.index(models[0]),
            key="compare_model_a",
            format_func=grouped_model_label,
            on_change=select_first_model_for_heading,
            args=("compare_model_a",),
        )
    with selector_b:
        default_model_b = models[1] if len(models) > 1 else models[0]
        model_b = st.selectbox(
            "Model B",
            grouped_options,
            index=grouped_options.index(default_model_b),
            key="compare_model_b",
            format_func=grouped_model_label,
            on_change=select_first_model_for_heading,
            args=("compare_model_b",),
        )
    model_a = first_model_by_heading.get(model_a, model_a)
    model_b = first_model_by_heading.get(model_b, model_b)
    if model_a == model_b:
        st.warning("Select two different models to compare.")
        return

    metrics_a = metrics_by_model[model_a]
    metrics_b = metrics_by_model[model_b]
    display_a = str(metrics_a.get("display_name") or model_a)
    display_b = str(metrics_b.get("display_name") or model_b)
    score_a = float(metrics_a.get("overall_score") or 0.0)
    score_b = float(metrics_b.get("overall_score") or 0.0)
    tie = abs(score_a - score_b) < 0.05
    card_a, card_b = st.columns(2)
    with card_a:
        st.markdown(
            comparison_card_html(
                display_a,
                metrics_a,
                metrics_b,
                winner=score_a > score_b,
                tie=tie,
            ),
            unsafe_allow_html=True,
        )
    with card_b:
        st.markdown(
            comparison_card_html(
                display_b,
                metrics_b,
                metrics_a,
                winner=score_b > score_a,
                tie=tie,
            ),
            unsafe_allow_html=True,
        )
    st.markdown("**Final remarks**")
    st.info(model_comparison_remarks(display_a, metrics_a, display_b, metrics_b))


def render_batch_history() -> None:
    history = load_batch_history()
    if not history:
        st.info("No saved batch eval runs yet.")
        return

    grouped_days = group_history_by_day(history)
    day_keys = [day_key for day_key, _ in grouped_days]
    default_day_key = next((day_key for day_key in day_keys if day_key == datetime.now().strftime("%Y-%m-%d")), day_keys[0])
    pending_day_key = st.session_state.pop("history_selected_day_pending", None)
    if pending_day_key in day_keys:
        st.session_state.history_selected_day = pending_day_key
    elif pending_day_key is None and "history_selected_day" not in st.session_state:
        st.session_state.history_selected_day = default_day_key
    st.session_state.setdefault("history_selected_day", default_day_key)
    if st.session_state.history_selected_day not in day_keys:
        st.session_state.history_selected_day = default_day_key

    st.markdown("### History by Day")
    selected_day = st.pills(
        "History day",
        day_keys,
        selection_mode="single",
        default=st.session_state.history_selected_day,
        format_func=history_day_label,
        key="history_selected_day",
        width="stretch",
    )
    selected_day_key = selected_day or st.session_state.history_selected_day or default_day_key
    selected_history = next(
        (entries for day_key, entries in grouped_days if day_key == selected_day_key),
        [],
    )

    report_col, compare_col = st.columns(2)
    with report_col:
        show_report = st.toggle("Show report", value=False)
    st.session_state.setdefault("show_model_comparison", False)
    with compare_col:
        comparison_label = (
            "Close model comparison"
            if st.session_state.show_model_comparison
            else "Compare models"
        )
        if st.button(comparison_label, use_container_width=True):
            st.session_state.show_model_comparison = not st.session_state.show_model_comparison
            st.rerun()
    delete_col, summary_col = st.columns([1.2, 3.8])
    with delete_col:
        if st.button("Delete selected day", use_container_width=True, key=f"delete_history_day_{selected_day_key}"):
            delete_batch_history_day(selected_day_key)
            remaining_history = load_batch_history()
            remaining_days = [day_key for day_key, _ in group_history_by_day(remaining_history)]
            st.session_state.history_selected_day_pending = remaining_days[0] if remaining_days else None
            st.rerun()
    with summary_col:
        st.caption(
            f"{history_day_label(selected_day_key)} · {len(selected_history)} saved runs"
        )
    if show_report:
        render_history_report(selected_history)
        st.divider()
    if st.session_state.show_model_comparison:
        st.markdown("### Model comparison")
        render_model_comparison(selected_history)
        st.divider()

    if not selected_history:
        st.info("No saved runs in the selected day block.")
        return

    history_model_labels = {history_model_filter_label(entry) for entry in selected_history}
    model_options = list(MODEL_OPTIONS)
    model_options.extend(sorted(history_model_labels.difference(model_options)))
    mode_options = sorted({str(entry.get("mode_label") or "Unknown mode") for entry in selected_history})
    rag_options = sorted({str(entry.get("rag_style") or "Unknown RAG") for entry in selected_history})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sort_order = st.selectbox(
            "Sort by answer quality",
            ["Original order", "Highest first", "Lowest first", "Best overall model"],
            key="history_sort_score",
        )
    with c2:
        model_filter = st.selectbox(
            "Model",
            ["All", *model_options],
            key="history_filter_model",
        )
    with c3:
        mode_filter = st.selectbox(
            "Answer mode",
            ["All", *mode_options],
            key="history_filter_mode",
        )
    with c4:
        rag_filter = st.selectbox(
            "RAG behavior",
            ["All", *rag_options],
            key="history_filter_rag",
        )

    filtered_history = []
    for entry in selected_history:
        if model_filter != "All" and history_model_filter_label(entry) != model_filter:
            continue
        if mode_filter != "All" and entry.get("mode_label") != mode_filter:
            continue
        if rag_filter != "All" and entry.get("rag_style") != rag_filter:
            continue
        filtered_history.append(entry)

    if sort_order == "Best overall model":
        filtered_history = rank_history_by_overall_model(filtered_history)
    elif sort_order != "Original order":
        reverse = sort_order == "Highest first"
        filtered_history.sort(
            key=lambda entry: float((entry.get("summary") or {}).get("score") or 0.0),
            reverse=reverse,
        )

    st.caption(
        f"Showing {len(filtered_history)} of {len(selected_history)} saved runs in {history_day_label(selected_day_key)}."
    )
    if not filtered_history:
        st.info("No history cards match the selected filters.")
        return

    for entry in filtered_history:
        render_history_card(entry)


def render_batch_eval_tab(
    *,
    selected_config: dict[str, Any],
    mode_label: str,
    model_label: str,
    rag_style: str,
    combined_planning: bool,
    fast_rag: bool,
    knowledge_domain: str,
    knowledge_source_label: str,
    temperature: float,
    max_tokens: int,
    batch_view: str,
    auto_model_labels: list[str],
    auto_mode_labels: list[str],
    start_automation: bool,
) -> None:
    st.subheader("Batch Eval")
    st.caption("Paste one question per line. The batch uses the current sidebar configuration.")
    st.caption(batch_history_file_status())
    retrieval_change_label = st.text_input(
        "Retrieval change label",
        value=st.session_state.get("retrieval_change_label", ""),
        key="retrieval_change_label",
        help="Label the current retrieval tweak batch so report/history comparisons stay attributable.",
        placeholder="Example: phase-2 top_k=6 -> context_k=3",
    )

    preset_domain = "biek" if is_biek_domain(knowledge_domain) else (
        knowledge_domain if knowledge_domain in BATCH_QUESTION_PRESETS else "synapse"
    )
    selected_default_questions = BATCH_QUESTION_PRESETS[preset_domain]
    if "batch_questions_text" not in st.session_state:
        st.session_state.batch_questions_text = selected_default_questions
    if "batch_questions_domain" not in st.session_state:
        st.session_state.batch_questions_domain = preset_domain
    if st.session_state.get("batch_questions_domain") != preset_domain:
        st.session_state.batch_questions_text = selected_default_questions
        st.session_state.batch_questions_domain = preset_domain

    active_batch_preset = st.session_state.get("batch_questions_domain", preset_domain)
    preset_col1, preset_col2 = st.columns(2)
    with preset_col1:
        if st.button(
            "Load Synapse Batch",
            type="primary" if active_batch_preset == "synapse" else "secondary",
            use_container_width=True,
        ):
            st.session_state.batch_questions_text = BATCH_QUESTION_PRESETS["synapse"]
            st.session_state.batch_questions_domain = "synapse"
            st.rerun()
    with preset_col2:
        if st.button(
            "Load BIEK Batch",
            type="primary" if active_batch_preset == "biek" else "secondary",
            use_container_width=True,
        ):
            st.session_state.batch_questions_text = BATCH_QUESTION_PRESETS["biek"]
            st.session_state.batch_questions_domain = "biek"
            st.rerun()

    batch_text = st.text_area(
        "Questions",
        key="batch_questions_text",
        height=180,
    )
    questions = [line.strip() for line in batch_text.splitlines() if line.strip()]

    planned_runs, skipped_runs = selected_plan_runs(
        model_labels=auto_model_labels,
        mode_labels=auto_mode_labels,
    )

    if batch_view == "Manual":
        st.markdown("#### Current config")
        st.session_state.setdefault("manual_batch_job", None)
        manual_batch_job = st.session_state.get("manual_batch_job")
        is_running_manual_batch = bool(manual_batch_job and not manual_batch_job.get("done"))
        action_col, stop_col = st.columns(2)
        with action_col:
            if st.button(
                "Run Batch Eval",
                type="primary",
                disabled=not questions or is_running_manual_batch,
            ):
                st.session_state.manual_batch_job = start_manual_batch_job(
                    questions=questions,
                    model_config=selected_config,
                    mode_label=mode_label,
                    model_label=model_label,
                    rag_style=rag_style,
                    combined_planning=combined_planning,
                    fast_rag=fast_rag,
                    knowledge_domain=knowledge_domain,
                    knowledge_source_label=knowledge_source_label,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    retrieval_change_label=retrieval_change_label,
                )
                st.rerun()
        with stop_col:
            if is_running_manual_batch and st.button(
                "Stop Batch Eval",
                type="secondary",
                use_container_width=True,
            ):
                manual_batch_job["stop_requested"] = True
                st.rerun()

        manual_batch_job = st.session_state.get("manual_batch_job")
        if manual_batch_job:
            question_total = int(manual_batch_job.get("question_total") or 0)
            current_question_index = int(manual_batch_job.get("current_question_index") or 0)
            st.progress(current_question_index / question_total if question_total else 0.0)
            current_question = manual_batch_job.get("current_question")
            if current_question and not manual_batch_job.get("done"):
                st.caption(
                    f"Current question {current_question_index}/{question_total}: {current_question}"
                )
            if manual_batch_job.get("stop_requested") and not manual_batch_job.get("done"):
                st.warning("Stopping after the current in-flight question finishes.")
            if manual_batch_job.get("error"):
                st.error(f"Batch eval failed: {manual_batch_job['error']}")
            if manual_batch_job.get("done"):
                st.session_state.batch_eval_rows = manual_batch_job.get("rows") or []
                if st.session_state.batch_eval_rows:
                    summary = summarize_batch(st.session_state.batch_eval_rows)
                    history_entry = build_batch_history_entry(
                        rows=st.session_state.batch_eval_rows,
                        summary=summary,
                        model_label=model_label,
                        mode_label=mode_label,
                        rag_style=rag_style,
                        combined_planning=combined_planning,
                        fast_rag=fast_rag,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        retrieval_change_label=str(manual_batch_job.get("retrieval_change_label") or ""),
                    )
                    add_batch_history_entry(history_entry)
                if manual_batch_job.get("stopped"):
                    st.warning("Batch eval stopped.")
                elif st.session_state.batch_eval_rows:
                    st.success("Batch eval saved to history.")
                st.session_state.manual_batch_job = None
                st.rerun()
            time.sleep(0.5)
            st.rerun()
    else:
        st.markdown("#### Automation")
        st.caption("Configure the queue from the sidebar, then start or stop it there.")
        if planned_runs:
            st.caption(f"Planned runs: {len(planned_runs)}")
        if skipped_runs:
            st.warning(
                "Skipping unsupported fine-tuned combinations: "
                + ", ".join(skipped_runs)
            )
        automation_job = st.session_state.get("automation_job")
        if start_automation and questions and planned_runs and not automation_job:
            st.session_state.automation_job = start_automation_job(
                planned_runs=planned_runs,
                skipped_runs=skipped_runs,
                questions=questions,
                rag_style=rag_style,
                combined_planning=combined_planning,
                fast_rag=fast_rag,
                knowledge_domain=knowledge_domain,
                knowledge_source_label=knowledge_source_label,
                temperature=temperature,
                max_tokens=max_tokens,
                retrieval_change_label=retrieval_change_label,
            )
            st.rerun()

        automation_job = st.session_state.get("automation_job")
        if automation_job:
            total_runs = len(automation_job.get("planned_runs") or [])
            completed_runs = int(automation_job.get("completed_runs") or 0)
            current_run_index = int(automation_job.get("current_run_index") or 0)
            question_total = int(automation_job.get("question_total") or 0)
            current_question_index = int(automation_job.get("current_question_index") or 0)
            outer_progress = st.progress(
                completed_runs / total_runs if total_runs else 0.0
            )
            inner_progress = st.progress(
                current_question_index / question_total if question_total else 0.0
            )
            current_model_label = automation_job.get("current_model_label")
            current_mode_label = automation_job.get("current_mode_label")
            if current_model_label and current_mode_label and not automation_job.get("done"):
                st.caption(
                    f"Current run {current_run_index}/{total_runs}: "
                    f"{MODEL_OPTION_DISPLAY_NAMES.get(current_model_label, current_model_label)} | "
                    f"{current_mode_label}"
                )
            current_question = automation_job.get("current_question")
            if current_question and not automation_job.get("done"):
                st.caption(f"Current question {current_question_index}/{question_total}: {current_question}")
            if automation_job.get("stop_requested") and not automation_job.get("done"):
                st.warning("Stopping after the current in-flight question finishes.")
            if automation_job.get("error"):
                st.error(f"Automation failed: {automation_job['error']}")
            if automation_job.get("done"):
                st.session_state.batch_eval_rows = automation_job.get("rows") or []
                st.session_state.batch_eval_suite_runs = automation_job.get("suite_rows") or []
                if automation_job.get("stopped"):
                    st.warning("Automation stopped.")
                else:
                    st.success(
                        f"Saved {len(st.session_state.batch_eval_suite_runs)} batch eval runs to history."
                    )
                st.session_state.automation_job = None
                st.rerun()
            time.sleep(0.5)
            st.rerun()

    suite_runs = st.session_state.get("batch_eval_suite_runs") or []
    if suite_runs:
        st.markdown("#### Latest automated run")
        st.dataframe(suite_runs, use_container_width=True, hide_index=True)

    show_history = st.toggle("Show history", value=True)
    if show_history:
        render_batch_history()
        st.divider()

    rows = st.session_state.get("batch_eval_rows") or []
    if not rows:
        return

    summary = summarize_batch(rows)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total", summary["total"])
    c2.metric("Pass", summary["pass"])
    c3.metric("Fail", summary["fail"])
    c4.metric("Partial", summary["partial"])
    c5.metric("Avg latency", f"{summary['avg_latency_s']}s")
    c6.metric("Refusals", summary["refusal_count"])
    prompt_summary = summary.get("prompt_metrics") or {}
    if any(value is not None for value in prompt_summary.values()):
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Avg input tokens", prompt_summary.get("avg_input_tokens"))
        p2.metric("Avg output tokens", prompt_summary.get("avg_output_tokens"))
        p3.metric("Avg total tokens", prompt_summary.get("avg_total_tokens"))
        p4.metric("Avg prompt chars", prompt_summary.get("avg_prompt_chars"))
        p5.metric("Avg context chunks", prompt_summary.get("avg_selected_context_chunks"))

    confidence = summary.get("retrieval_confidence") or {}
    if confidence.get("question_count"):
        with st.expander("Retrieval confidence", expanded=True):
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Avg top similarity", confidence.get("avg_top_similarity"))
            r2.metric("Avg score gap", confidence.get("avg_score_gap"))
            r3.metric("Expected source top-1", f"{confidence.get('expected_source_top1_pct')}%")
            r4.metric("Expected source top-3", f"{confidence.get('expected_source_top3_pct')}%")
            st.caption(
                f"Pass avg similarity: {confidence.get('pass_avg_similarity')} | "
                f"Fail avg similarity: {confidence.get('fail_avg_similarity')} | "
                f"High-confidence failures (≥0.75): {confidence.get('high_confidence_failures_075')} | "
                f"Low-confidence questions (<0.55): {confidence.get('low_confidence_questions_055')}"
            )
            confidence_rows = [row for row in rows if row.get("top_similarity") is not None]
            lowest_rows = sorted(
                confidence_rows,
                key=lambda row: float(row.get("top_similarity") or 0.0),
            )[:5]
            if lowest_rows:
                st.markdown("**Lowest-confidence questions**")
                st.dataframe(
                    [
                        {
                            "question": row.get("question"),
                            "verdict": row.get("verdict"),
                            "top_similarity": row.get("top_similarity"),
                            "score_gap": row.get("score_gap"),
                            "top_retrieved_docs": row.get("top_retrieved_docs"),
                        }
                        for row in lowest_rows
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            confident_failures = [
                row for row in confidence_rows
                if row.get("verdict", "").startswith("❌")
                and float(row.get("top_similarity") or 0.0) >= 0.75
            ]
            if confident_failures:
                st.markdown("**High-confidence failures**")
                st.dataframe(
                    [
                        {
                            "question": row.get("question"),
                            "issue": row.get("issue"),
                            "top_similarity": row.get("top_similarity"),
                            "score_gap": row.get("score_gap"),
                            "top_retrieved_docs": row.get("top_retrieved_docs"),
                        }
                        for row in confident_failures
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

    if summary["issue_counts"]:
        with st.expander("Issue counts", expanded=True):
            st.json(summary["issue_counts"])

    table_rows = [
        {key: value for key, value in row.items() if key != "debug"}
        for row in rows
    ]
    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    json_payload = json.dumps(
        {
            "summary": summary,
            "model_label": model_label,
            "mode_label": mode_label,
            "rag_style": rag_style,
            "combined_planning": combined_planning,
            "results": rows,
        },
        ensure_ascii=False,
        indent=2,
    )
    st.download_button(
        "Download JSON results",
        data=json_payload,
        file_name="batch_eval_results.json",
        mime="application/json",
    )
    st.download_button(
        "Download CSV results",
        data=rows_to_csv(rows),
        file_name="batch_eval_results.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title="Synapse Local Chat Lab", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {
            padding-bottom: 7rem;
        }
        [data-testid="stChatInput"] {
            position: fixed;
            bottom: 1rem;
            left: 23rem;
            right: 2rem;
            z-index: 1000;
        }
        [data-testid="stChatInput"] textarea {
            max-height: 8rem;
        }
        [data-testid="stPopover"] button {
            width: 2rem !important;
            min-width: 2rem !important;
            height: 2rem !important;
            min-height: 2rem !important;
            padding: 0 !important;
            border: 0 !important;
            border-radius: 50% !important;
            background: transparent !important;
            box-shadow: none !important;
        }
        [data-testid="stPopover"] button:hover {
            background: rgba(56, 189, 248, 0.12) !important;
        }
        [data-testid="stPopover"] button:focus-visible {
            outline: 1px solid #38bdf8 !important;
            outline-offset: 2px;
        }
        [data-testid="stPopover"] button svg {
            display: none !important;
        }
        @media (max-width: 900px) {
            [data-testid="stChatInput"] {
                left: 1rem;
                right: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Synapse Local Chat Lab")
    st.caption("Local-only test UI for Base, Fine-tuned, RAG, and Fine-tuned + RAG configurations.")

    with st.sidebar:
        st.header("Runtime Config")
        batch_view = st.segmented_control(
            "Batch mode",
            ["Manual", "Automation"],
            default="Manual",
            key="sidebar_batch_view_mode",
            width="stretch",
        )
        st.session_state.setdefault("automation_job", None)
        auto_model_labels: list[str] = []
        auto_mode_labels: list[str] = []
        start_automation = False
        if batch_view == "Automation":
            st.header("Batch Automation")
            st.caption("Pick multiple models and answer modes for queued batch runs.")
            auto_model_labels = st.pills(
                "Models to run",
                list(MODEL_OPTIONS),
                selection_mode="multi",
                default=[st.session_state.get("manual_model_label", DEFAULT_MODEL_LABEL)],
                format_func=lambda current_model: MODEL_OPTION_DISPLAY_NAMES.get(current_model, current_model),
                key="batch_eval_auto_models",
                help="These model families will run one after another in Batch Eval automation.",
                width="stretch",
            ) or []
            auto_mode_labels = st.pills(
                "Answer modes to run",
                list(ANSWER_MODES),
                selection_mode="multi",
                default=[st.session_state.get("manual_mode_label", "Base + RAG")],
                key="batch_eval_auto_modes",
                help="Each selected model will run once for each selected answer mode.",
                width="stretch",
            ) or []
            planned_runs, skipped_runs = selected_plan_runs(
                model_labels=auto_model_labels,
                mode_labels=auto_mode_labels,
            )
            if planned_runs:
                st.caption(f"Queued runs: {len(planned_runs)}")
            if skipped_runs:
                st.caption(f"Skipped unsupported combos: {len(skipped_runs)}")
            active_automation = st.session_state.get("automation_job")
            is_running_automation = bool(active_automation and not active_automation.get("done"))
            if is_running_automation:
                if st.button(
                    "Stop Automation",
                    type="secondary",
                    key="sidebar_stop_automation",
                    use_container_width=True,
                ):
                    active_automation["stop_requested"] = True
                    st.rerun()
            else:
                start_automation = st.button(
                    "Start Automation",
                    type="primary",
                    disabled=not planned_runs,
                    key="sidebar_start_automation",
                    use_container_width=True,
                )
            st.divider()

        model_label = st.session_state.get("manual_model_label", DEFAULT_MODEL_LABEL)
        mode_label = st.session_state.get("manual_mode_label", "Base + RAG")
        knowledge_source_label = st.session_state.get("knowledge_source_label", "BIEK KB")
        if batch_view == "Manual":
            model_label = st.selectbox(
                "Model family / selected model",
                list(MODEL_OPTIONS),
                index=list(MODEL_OPTIONS).index(model_label) if model_label in MODEL_OPTIONS else 0,
                format_func=lambda model: MODEL_OPTION_DISPLAY_NAMES.get(model, model),
                key="manual_model_label",
                help=(
                    "Selects the model pair to test. For fine-tuned modes, this uses the Synapse-tuned "
                    "model if it exists. For base modes, it uses the original base model for that family."
                ),
            )
            mode_label = st.selectbox(
                "Answer mode",
                list(ANSWER_MODES),
                index=list(ANSWER_MODES).index(mode_label) if mode_label in ANSWER_MODES else 1,
                key="manual_mode_label",
                help=(
                    "Base = original model only. Base + RAG = original model with KB retrieval. "
                    "Fine-tuned = Synapse model only. Fine-tuned + RAG = Synapse model with KB retrieval."
                ),
            )
        knowledge_source_label = st.selectbox(
            "Knowledge source",
            list(KNOWLEDGE_SOURCE_OPTIONS),
            index=list(KNOWLEDGE_SOURCE_OPTIONS).index(knowledge_source_label)
            if knowledge_source_label in KNOWLEDGE_SOURCE_OPTIONS
            else 0,
            key="knowledge_source_label",
            help="Choose which local knowledge base the RAG modes should query.",
        )
        knowledge_domain = KNOWLEDGE_SOURCE_OPTIONS[knowledge_source_label]
        rag_style = st.radio(
            "RAG behavior",
            ["Deterministic/template", "Model-generated"],
            index=1,
            help=(
                "Deterministic/template uses safer fixed policy answers for known cases. "
                "Model-generated lets the selected LLM write more natural answers from KB evidence, "
                "but it can be slower and may need stronger validation."
            ),
        )
        combined_planning = st.toggle(
            "Combined planning",
            value=False,
            help=(
                "When on, one model call handles both query rewriting and intent classification "
                "before retrieval. This usually reduces latency because the RAG path makes fewer "
                "LLM calls. Turn it off only when comparing against the older separate rewrite + "
                "classifier flow."
            ),
        )
        fast_rag = st.toggle(
            "Fast RAG",
            value=True,
            help=(
                "When on, the RAG path skips extra correction/retry calls after the first answer. "
                "This is faster and useful for latency testing, but weak answers will not get a "
                "second model pass to fix shallow wording, cautious refusals, or instruction issues."
            ),
        )
        temperature = st.slider(
            "Temperature",
            0.0,
            1.0,
            0.2,
            0.05,
            help=(
                "Controls randomness during generation. Lower values are more stable and factual; "
                "higher values sound more varied but can increase drift or hallucination. For RAG "
                "testing, 0.1-0.3 is usually the safer range."
            ),
        )
        max_tokens = st.slider(
            "Max tokens",
            40,
            500,
            180,
            10,
            help=(
                "Maximum answer length the model is allowed to generate. Higher values allow more "
                "detail but increase latency and may cause rambling. Lower values are faster and "
                "more concise, but can cut off useful answers."
            ),
        )
        debug_enabled = st.toggle(
            "Debug",
            value=True,
            help=(
                "Shows raw metadata such as rewrite query, intent/classification, selected route, "
                "token usage, and observability. Use this when diagnosing why an answer was wrong."
            ),
        )
        show_sources = st.toggle(
            "Show retrieved sources",
            value=True,
            help=(
                "Displays the KB document IDs used as evidence. This helps confirm whether retrieval "
                "found the right source documents before blaming the final model."
            ),
        )

        selected_config = MODEL_OPTIONS[model_label]
        if batch_view == "Manual":
            st.divider()
            st.markdown("**Resolved models**")
            st.code(
                json.dumps(
                    {
                        "base_model": resolved_model_display(
                            model_label,
                            "Base",
                            selected_config["base_model"],
                        ),
                        "fine_tuned_model": (
                            resolved_model_display(
                                model_label,
                                "Fine-tuned",
                                selected_config.get("fine_tuned_model"),
                            )
                            if selected_config.get("fine_tuned_model") else None
                        ),
                        "fine_tune_exists": selected_config.get("fine_tune_exists"),
                    },
                    indent=2,
                ),
                language="json",
            )

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    if batch_view == "Automation":
        render_batch_eval_tab(
            selected_config=selected_config,
            mode_label=mode_label,
            model_label=model_label,
            rag_style=rag_style,
            combined_planning=combined_planning,
            fast_rag=fast_rag,
            knowledge_domain=knowledge_domain,
            knowledge_source_label=knowledge_source_label,
            temperature=temperature,
            max_tokens=max_tokens,
            batch_view=batch_view,
            auto_model_labels=auto_model_labels,
            auto_mode_labels=auto_mode_labels,
            start_automation=start_automation,
        )
        return

    chat_tab, batch_tab = st.tabs(["Chat", "Batch Eval"])

    with chat_tab:
        st.session_state.setdefault("messages", [])
        st.session_state.setdefault("active_job", None)

        for item in st.session_state.messages:
            render_message(item, debug_enabled=debug_enabled, show_sources=show_sources)

        if st.session_state.active_job:
            render_active_job(debug_enabled=debug_enabled, show_sources=show_sources)
            return

        prompt_placeholder = (
            "Ask a BIEK question" if is_biek_domain(knowledge_domain) else "Ask a Synapse Tech question"
        )
        question = st.chat_input(prompt_placeholder)
        if question:
            st.session_state.active_job = start_generation_job(
                question=question,
                messages=st.session_state.messages,
                model_config=selected_config,
                mode_label=mode_label,
                answer_mode=ANSWER_MODES[mode_label],
                model_label=model_label,
                rag_generated=rag_style == "Model-generated",
                combined_planning=combined_planning,
                fast_rag=fast_rag,
                knowledge_domain=knowledge_domain,
                knowledge_source_label=knowledge_source_label,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            st.rerun()

    with batch_tab:
        render_batch_eval_tab(
            selected_config=selected_config,
            mode_label=mode_label,
            model_label=model_label,
            rag_style=rag_style,
            combined_planning=combined_planning,
            fast_rag=fast_rag,
            knowledge_domain=knowledge_domain,
            knowledge_source_label=knowledge_source_label,
            temperature=temperature,
            max_tokens=max_tokens,
            batch_view=batch_view,
            auto_model_labels=auto_model_labels,
            auto_mode_labels=auto_mode_labels,
            start_automation=start_automation,
        )


if __name__ == "__main__":
    main()
