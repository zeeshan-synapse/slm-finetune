import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

OLLAMA_URL = "http://localhost:11434"
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")

GENERATOR_MODEL = "synapse-1.5b-v1"
JUDGE_MODEL = "llama3:latest"

GENERATOR_SYSTEM_PROMPT = (
    "You are a factual assistant for Synapse Tech. "
    "Answer clearly and concisely. "
    "If information is unknown, say it is not confirmed."
)

JUDGE_SYSTEM_PROMPT = """You are a strict response quality judge.
Return ONLY valid JSON with this schema:
{
  "pass": boolean,
  "reasons": [string],
  "notes": string
}

Judge the answer for:
- hallucinated specifics (pricing, integrations, SLA/certs/roadmap claims not confirmed),
- artifacts/meta text,
- topic drift (unrelated additions),
- clarity and conciseness.

Critical output rule:
- Output exactly one single-line JSON object and nothing else.
"""

FALLBACK_RESPONSE = (
    "This detail is not confirmed in available information. "
    "Please verify with Synapse Tech."
)
FAILURE_LOG_PATH = Path("logs/guardrail_failures.jsonl")


FORBIDDEN_ARTIFACT_PREFIXES = [
    "answer according to:",
    "follows from:",
    "answer below",
    "according to the article",
]

ARTIFACT_LEAK_PATTERNS = [
    r"answered\s*:",
    r"relevant material found",
    r"answer to the question below exactly",
    r"answer in one sentence",
    r"only lowercase unless quoted",
    r"\b[a-d]\)\s",
    r"\ba:\s.*\bb:\s",
    r"\bquestion:\s",
    r"according to the page",
    r"according to the report",
    r"which statement is correct",
    r"learn more here",
    r"sign up today",
    r"answer questions accurately",
    r"stay grounded",
    r"natural conversational tone",
]


def check_ollama() -> None:
    try:
        requests.get(OLLAMA_URL, timeout=5)
    except Exception:
        print("Ollama is not running. Start with: ollama serve")
        sys.exit(1)


def ollama_chat(
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.1,
    num_predict: int = 120,
    stop: list[str] | None = None,
) -> str:
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "stop": stop or [],
            },
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("message", {}).get("content", "").strip()


def safe_parse_json(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to recover JSON object when the judge wraps output
        # in prose before/after the JSON payload.
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"pass": False, "reasons": ["judge_output_not_json"], "notes": raw}


def validate_judge_contract(verdict: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if not isinstance(verdict, dict):
        return {
            "pass": False,
            "reasons": ["judge_output_not_object"],
            "notes": str(verdict),
        }

    if "pass" not in verdict or not isinstance(verdict["pass"], bool):
        reasons.append("judge_contract_invalid_pass")
    if "reasons" not in verdict or not isinstance(verdict["reasons"], list):
        reasons.append("judge_contract_invalid_reasons")
    if "notes" not in verdict or not isinstance(verdict["notes"], str):
        reasons.append("judge_contract_invalid_notes")

    if "reasons" in verdict and isinstance(verdict["reasons"], list):
        if any(not isinstance(item, str) for item in verdict["reasons"]):
            reasons.append("judge_contract_reasons_not_string_array")

    if reasons:
        return {
            "pass": False,
            "reasons": reasons,
            "notes": json.dumps(verdict, ensure_ascii=False),
        }
    return verdict


def sentence_count(text: str) -> int:
    parts = re.split(r"[.!?]+(?:\s+|$)", text.strip())
    return len([p for p in parts if p.strip()])


def contains_any(text: str, tokens: list[str]) -> bool:
    return any(t in text for t in tokens)


def is_high_risk_question(question: str) -> bool:
    low_q = question.lower()
    return contains_any(
        low_q,
        [
            "pricing",
            "price",
            "cost",
            "quote",
            "integration",
            "roi",
            "guarante",
            "sla",
            "certification",
            "compliance",
            "roadmap",
            "legal",
            "security",
        ],
    )


def is_policy_question(question: str) -> bool:
    low_q = question.lower()
    return contains_any(
        low_q,
        [
            "one-sentence policy",
            "one sentence policy",
            "policy should you follow",
            "should responses include",
            "provide a short policy",
            "when information is missing",
            "when info is missing",
            "should you guess",
            "must avoid",
        ],
    )


def is_comparison_question(question: str) -> bool:
    low_q = question.lower()
    return contains_any(
        low_q,
        [
            "compare",
            "comparison",
            "difference between",
            "vs",
            "versus",
            "choose between",
            "which one should",
        ],
    )


def is_sla_question(question: str) -> bool:
    low_q = question.lower()
    return contains_any(low_q, ["sla", "uptime", "response-time", "response time", "penalty clause"])


def is_roadmap_question(question: str) -> bool:
    low_q = question.lower()
    return contains_any(low_q, ["roadmap", "release date", "milestone", "upcoming feature"])


def is_cert_compliance_question(question: str) -> bool:
    low_q = question.lower()
    return contains_any(low_q, ["certification", "certificate", "compliance", "regulatory approval", "framework"])


def is_legal_question(question: str) -> bool:
    low_q = question.lower()
    return contains_any(low_q, ["legal", "liability", "contractual", "warranty", "indemn", "terms"])


def is_outcome_guarantee_question(question: str) -> bool:
    low_q = question.lower()
    guarantee_signal = contains_any(low_q, ["guarante", "promise"])
    outcome_signal = contains_any(
        low_q,
        [
            "roi",
            "return on investment",
            "return rate",
            "support cost",
            "support costs",
            "cost reduction",
            "productivity",
            "percentage",
            "percent",
            "reduction",
            "outcome",
            "3 months",
            "90 days",
            "deadline",
        ],
    )
    return guarantee_signal and outcome_signal


def mentions_specific_integration_name(text: str) -> bool:
    return contains_any(
        text,
        [
            "salesforce",
            "hubspot",
            "zapier",
            "google drive",
            "workday",
            "sap",
            "crm",
            "api endpoint",
            "rest api",
        ],
    )


def is_general_integration_list_question(question: str) -> bool:
    low_q = question.lower()
    return "integration" in low_q and contains_any(
        low_q,
        [
            "which integrations",
            "what integrations",
            "integration list",
            "integrations are explicitly supported",
            "integrations are explicitly confirmed",
        ],
    )


def is_specific_integration_question(question: str) -> bool:
    low_q = question.lower()
    if mentions_specific_integration_name(low_q):
        return True
    return "integration" in low_q and not is_general_integration_list_question(question)


def normalize_repeated_letters(text: str) -> str:
    return re.sub(r"([a-z])\1{2,}", r"\1", text.lower())


def is_small_talk_question(question: str) -> bool:
    low_q = question.strip().lower()
    normalized_q = normalize_repeated_letters(low_q)
    word_tokens = re.findall(r"[a-z]+", normalized_q)
    token_set = set(word_tokens)

    business_terms = {
        "agentic",
        "automation",
        "bot",
        "buy",
        "contact",
        "cost",
        "custom",
        "kb",
        "opira",
        "pricing",
        "product",
        "products",
        "purchase",
        "service",
        "services",
        "synapse",
        "workflow",
    }
    question_terms = {
        "can",
        "could",
        "do",
        "does",
        "how",
        "list",
        "tell",
        "what",
        "when",
        "where",
        "which",
        "why",
    }
    if token_set & business_terms:
        return False
    if (token_set & question_terms) and not contains_any(
        normalized_q,
        [
            "how are you",
            "who are you",
            "what can you do",
        ],
    ):
        return False

    small_talk_exact = {
        "hi",
        "hello",
        "helo",
        "hey",
        "thanks",
        "thank you",
        "yo",
        "sup",
        "good morning",
        "good afternoon",
        "good evening",
    }
    if normalized_q in small_talk_exact:
        return True
    greeting_tokens = {
        "hi",
        "hello",
        "helo",
        "hey",
        "yo",
        "sup",
        "bro",
        "there",
        "buddy",
        "mate",
    }
    if word_tokens and len(word_tokens) <= 4 and set(word_tokens) <= greeting_tokens:
        return True
    return contains_any(
        normalized_q,
        [
            "how are you",
            "who are you",
            "what can you do",
            "nice to meet you",
        ],
    )


def small_talk_response(question: str) -> str:
    low_q = question.strip().lower()
    if low_q in {"thanks", "thank you"}:
        return "You're welcome. If you have a Synapse Tech question, I can help."
    if low_q in {"who are you", "what can you do"}:
        return (
            "I am the Synapse Tech assistant. I can help with company products, services, "
            "and policy-safe factual guidance."
        )
    return "Hi! How can I help you with Synapse Tech today?"


def policy_intent(question: str) -> str | None:
    low_q = question.strip().lower()

    # Meta: how the assistant should handle unsupported legal/security *claims* (not legal advice).
    if contains_any(
        low_q,
        [
            "unsupported legal requests",
            "unsupported legal request",
            "how should you answer unsupported legal",
            "unsupported security claims",
            "unsupported security claim",
            "how should you answer unsupported security",
        ],
    ):
        return "unsupported_claims_handling"

    if contains_any(
        low_q,
        [
            "benchmark numbers are unavailable",
            "how should you respond if benchmark",
            "respond if benchmark numbers",
        ],
    ):
        return "benchmark_unavailable_policy"

    # Meta: unknown user questions → assistant behavior (not Synapse corpus facts).
    if contains_any(
        low_q,
        [
            "if i ask something unknown",
            "ask something unknown",
            "something unknown, how should you answer",
            "something unknown how should you",
        ],
    ):
        return "assistant_unknown_qna"

    # Rubric: tone, concision, vague users, forbidden patterns in answers.
    if contains_any(
        low_q,
        [
            "answer according to",
            "phrases like answer",
            "repetitive lines",
            "should responses include repetitive",
            "what tone should the assistant",
            "tone should the assistant use",
            "how concise should default responses",
            "concise answer style guideline",
            "give a concise answer style",
            "how should you handle vague user questions",
            "handle vague user questions",
            "what should never appear in final answers",
            "never appear in final answers",
        ],
    ):
        return "assistant_rubric"

    if contains_any(
        low_q,
        [
            "should you guess",
            "guess or state",
            "if a feature is not confirmed",
            "if information is missing",
            "if info is missing",
            "unconfirmed feature",
            "feature information is missing",
            "if feature information is missing",
        ],
    ):
        return "unknown_handling"
    if contains_any(
        low_q,
        [
            "answer:",
            "note:",
            "template/meta",
            "meta phrases",
            "scaffolding text",
            "answer below",
        ],
    ):
        return "meta_text_policy"
    if contains_any(
        low_q,
        [
            "one sentence",
            "default tone",
            "response style",
            "factual accuracy",
            "required response behavior",
        ],
    ):
        return "style_policy"
    if contains_any(
        low_q,
        [
            "three things",
            "must avoid",
            "prohibited response patterns",
            "behaviors the assistant must avoid",
        ],
    ):
        return "avoid_list_policy"
    if contains_any(
        low_q,
        [
            "chain-of-thought",
            "chain of thought",
            "meta commentary",
            "off-chain",
        ],
    ):
        return "cot_policy"
    return None


def policy_response(intent: str) -> str:
    if intent == "unsupported_claims_handling":
        return (
            "Do not provide legal advice or endorse security outcomes that are not documented. "
            "State that the request or claim cannot be confirmed from available information, "
            "stay factual and bounded, and point users to official Synapse contacts for commitments."
        )
    if intent == "benchmark_unavailable_policy":
        return (
            "If benchmark numbers are unavailable, do not invent results. "
            "State what is unavailable; only offer methodology or comparisons that are explicitly grounded."
        )
    if intent == "assistant_unknown_qna":
        return (
            "If a question is outside confirmed Synapse materials, say it is not confirmed in available information. "
            "Do not invent facts, numbers, or commitments; suggest verifying with Synapse Tech for official details."
        )
    if intent == "assistant_rubric":
        return (
            "Use a neutral, professional tone. Default to concise answers. "
            "Avoid template phrases, repeated boilerplate, and meta-instructions in final text. "
            "For vague user questions, ask one clarifying question or restate what you can answer from confirmed information."
        )
    if intent == "unknown_handling":
        return "If information is not confirmed, state it as unconfirmed and do not guess."
    if intent == "meta_text_policy":
        return "No. Final responses should not include template or meta phrases like 'Answer:' or 'Note:'."
    if intent == "style_policy":
        return "Use a neutral, factual, concise style and avoid promotional language or speculation."
    if intent == "avoid_list_policy":
        return "Avoid guessing unconfirmed facts, adding template/meta text, and drifting into unrelated content."
    if intent == "cot_policy":
        return "No. Do not include chain-of-thought or internal reasoning in user-facing answers."
    return "If information is not confirmed, state it as unconfirmed and do not guess."


def ops_intent(question: str) -> str | None:
    low_q = question.strip().lower()
    if contains_any(low_q, ["retail support team"]) and asks_for_metric(question):
        return "retail_deploy_and_kpi"
    if contains_any(low_q, ["retail support team"]):
        return "retail_first_deploy"
    if contains_any(low_q, ["bpo support center", "bpo support centre"]) and asks_for_metric(question):
        return "bpo_deploy_and_kpi"
    if contains_any(low_q, ["banking operations team", "banking operations"]):
        return "banking_first_deploy"
    if contains_any(low_q, ["bpo service desk", "bpo support center", "bpo support centre"]):
        return "bpo_first_deploy"
    if contains_any(low_q, ["which kpi should improve first", "which kpi should improve"]):
        return "first_kpi"
    if contains_any(low_q, ["pilot success in 30 days", "measure pilot success in 30 days"]):
        return "pilot_success_30"
    if contains_any(low_q, ["low-risk pilot", "low risk pilot"]):
        return "low_risk_pilot"
    if contains_any(low_q, ["balance automation with human oversight", "automation with human oversight"]):
        return "automation_human_oversight"
    if contains_any(low_q, ["estimate implementation effort", "implementation effort for a mid-size team", "implementation effort for a mid size team"]):
        return "implementation_effort"
    if contains_any(low_q, ["what data do we need before deployment", "data do we need before deployment"]):
        return "data_before_deployment"
    if contains_any(low_q, ["think about roi for automation", "roi for automation"]):
        return "roi_framework"
    if contains_any(low_q, ["current process is highly manual", "process is highly manual"]):
        return "highly_manual_process"
    if contains_any(low_q, ["choose between products with limited details", "products with limited details"]):
        return "limited_product_choice"
    return None


def ops_response(intent: str) -> str:
    if intent == "retail_first_deploy":
        return "Start with support workflow automation for repetitive customer requests."
    if intent == "retail_deploy_and_kpi":
        return (
            "Start with support workflow automation for repetitive customer requests. "
            "Track first-response time as the first KPI."
        )
    if intent == "banking_first_deploy":
        return "Start with a repetitive approval or document workflow where auditability matters."
    if intent == "bpo_first_deploy":
        return "Start with ticket triage or status-update automation for high-volume repetitive requests."
    if intent == "bpo_deploy_and_kpi":
        return (
            "Start with ticket triage or status-update automation for high-volume repetitive requests. "
            "Track first-response time first."
        )
    if intent == "first_kpi":
        return (
            "First-response time usually improves first after rollout because it reflects faster routing and less manual handling. "
            "Then track resolution time and error rate against the baseline."
        )
    if intent == "pilot_success_30":
        return (
            "Measure pilot success against a pre-pilot baseline using first-response time, resolution time, error rate, and adoption. "
            "Pick one workflow and set a target improvement you can verify within 30 days."
        )
    if intent == "low_risk_pilot":
        return (
            "Start with one narrow high-volume workflow, keep human review for exceptions, and compare results against a baseline. "
            "Expand only after the first KPI trend is stable."
        )
    if intent == "automation_human_oversight":
        return (
            "Automate repetitive decisions and keep humans on exceptions, approvals, and high-impact cases. "
            "That preserves speed while maintaining accountability."
        )
    if intent == "implementation_effort":
        return (
            "Estimate effort as a range based on scope, integrations, data readiness, and review requirements. "
            "For a mid-size team, start with a small pilot rather than a fixed delivery promise."
        )
    if intent == "data_before_deployment":
        return (
            "Gather the current workflow steps, source documents, decision rules, exception paths, and system touchpoints. "
            "Capture baseline metrics too so you can measure impact after deployment."
        )
    if intent == "roi_framework":
        return (
            "Think about ROI in terms of time saved, error reduction, throughput, and service quality against a baseline. "
            "Use a measurement framework rather than assuming a fixed percentage outcome."
        )
    if intent == "highly_manual_process":
        return (
            "If the current process is highly manual, map the highest-volume repetitive steps and automate one workflow first. "
            "Measure cycle time and error rate before expanding."
        )
    if intent == "limited_product_choice":
        return (
            "Choose based on the confirmed use case, required channels, and governance needs. "
            "If important details are missing, treat them as unconfirmed and verify them before deciding."
        )
    return (
        "Start with one narrow repetitive workflow, define a baseline, and measure the first KPI you expect to improve. "
        "Expand only after the result is stable."
    )


def general_integration_response() -> str:
    return (
        "Public pages mention tools and platforms such as Airbyte, n8n, Airtable, Vapi, Google Dialogflow, AWS, Azure, Workday, and SAP in specific service or product contexts. "
        "A single exhaustive public integration list is not clearly published."
    )


def asks_for_metric(question: str) -> bool:
    return contains_any(question.lower(), ["kpi", "metric", "measure", "track"])


def asks_for_recommendation(question: str) -> bool:
    return contains_any(
        question.lower(),
        [
            "deploy first",
            "start with",
            "choose between",
            "run a low-risk pilot",
            "run a low risk pilot",
            "balance automation",
            "what if our current process",
        ],
    )


def validate_ops_response(intent: str, answer: str) -> list[str]:
    low_a = answer.lower()
    reasons: list[str] = []

    if "?" in answer:
        reasons.append("ops_response_contains_question")

    if intent in {"retail_first_deploy", "banking_first_deploy", "bpo_first_deploy", "retail_deploy_and_kpi", "bpo_deploy_and_kpi"}:
        if not contains_any(low_a, ["automate", "automation", "workflow", "ticket", "triage", "approval", "document"]):
            reasons.append("missing_deployment_recommendation")

    if intent in {"retail_deploy_and_kpi", "bpo_deploy_and_kpi", "first_kpi"}:
        if not contains_any(low_a, ["first-response time", "response time", "resolution time", "cycle time", "error rate"]):
            reasons.append("missing_kpi_recommendation")

    if intent == "pilot_success_30":
        if not contains_any(low_a, ["baseline", "30 days", "adoption", "error rate", "resolution time", "first-response time", "metric"]):
            reasons.append("missing_pilot_success_guidance")

    if intent == "low_risk_pilot":
        if not contains_any(low_a, ["baseline", "workflow", "human review", "exceptions", "expand"]):
            reasons.append("missing_low_risk_pilot_guidance")

    if intent == "automation_human_oversight":
        if not contains_any(low_a, ["automate", "automation"]) or not contains_any(low_a, ["human", "review", "exceptions", "approvals"]):
            reasons.append("missing_oversight_guidance")

    if intent == "implementation_effort":
        if not contains_any(low_a, ["range", "depends", "scope", "integrations", "data readiness", "resources", "pilot"]):
            reasons.append("missing_effort_estimation_guidance")

    if intent == "data_before_deployment":
        if not contains_any(low_a, ["workflow", "documents", "decision rules", "exception", "system", "baseline"]):
            reasons.append("missing_deployment_data_guidance")

    if intent == "roi_framework":
        if not contains_any(low_a, ["baseline", "time saved", "error reduction", "throughput", "quality", "measurement"]):
            reasons.append("missing_roi_framework_guidance")

    if intent == "highly_manual_process":
        if not contains_any(low_a, ["manual", "workflow", "automate", "cycle time", "error rate"]):
            reasons.append("missing_manual_process_guidance")
        if contains_any(low_a, ["candidate", "resume", "hiring", "ats", "job board"]):
            reasons.append("ops_response_off_topic_domain")

    if intent == "limited_product_choice":
        if not contains_any(low_a, ["use case", "confirmed", "verify", "unconfirmed", "details", "governance", "channels"]):
            reasons.append("missing_product_choice_guidance")

    return reasons


def run_rule_checks(question: str, answer: str) -> list[str]:
    low_q = question.lower()
    low_a = answer.lower()
    failures: list[str] = []

    if not answer.strip():
        failures.append("empty_answer")
        return failures

    artifact_patterns = [
        r"answer according to:",
        r"follows from:",
        r"answer below",
        r"according to the article",
        r"according to the page",
        r"according to the report",
        r"answer:",
        r"note:",
        r"list the names of",
        r"which statement is correct",
        r"learn more here",
        r"sign up today",
        r"https?://",
        r"www\.",
        r"#\s*\w+",
        *ARTIFACT_LEAK_PATTERNS,
    ]
    if any(re.search(p, low_a) for p in artifact_patterns):
        failures.append("artifact_or_meta_text")

    strict_mode = is_high_risk_question(question) or is_policy_question(question)
    max_sentences = 3 if strict_mode else 6
    if sentence_count(answer) > max_sentences:
        failures.append("too_verbose")

    if ("pricing" in low_q or "cost" in low_q or "price" in low_q) and (
        re.search(r"\byes\b", low_a)
        or "package" in low_a
        or "discount" in low_a
        or "subscription" in low_a
    ):
        failures.append("possible_pricing_hallucination")

    if is_specific_integration_question(question) and (
        "airbyte" in low_a
        or "zapier" in low_a
        or "n8n" in low_a
        or "airtable" in low_a
        or "salesforce" in low_a
    ):
        failures.append("possible_integration_hallucination")

    if is_sla_question(question) and (
        re.search(r"\byes\b", low_a)
        or "non-negotiable" in low_a
        or "guaranteed uptime" in low_a
        or "99.9" in low_a
        or "penalty clause" in low_a
    ):
        failures.append("possible_sla_hallucination")

    if is_roadmap_question(question) and (
        re.search(r"\b\d+\s*(day|days|week|weeks|month|months|quarter|quarters|year|years)\b", low_a)
        or re.search(r"\bq[1-4]\b", low_a)
        or re.search(r"\b20\d{2}\b", low_a)
        or "will be released" in low_a
    ):
        failures.append("possible_roadmap_date_hallucination")

    if is_cert_compliance_question(question) and (
        "iso" in low_a
        or "soc 2" in low_a
        or "hipaa" in low_a
        or "gdpr certified" in low_a
        or "approved by regulator" in low_a
    ):
        failures.append("possible_cert_compliance_hallucination")

    if "roi" in low_q and "guarante" in low_a:
        # Allow clear negative phrasing like "cannot guarantee" or "no guarantee".
        negated_patterns = [
            r"\bno\b[^.]{0,20}\bguarante",
            r"\bcannot\b[^.]{0,20}\bguarante",
            r"\bcan not\b[^.]{0,20}\bguarante",
            r"\bcan't\b[^.]{0,20}\bguarante",
            r"\bnot\b[^.]{0,20}\bguarante",
            r"\bwithout\b[^.]{0,20}\bguarante",
        ]
        if not any(re.search(p, low_a) for p in negated_patterns):
            failures.append("roi_guarantee_claim")

    if is_comparison_question(question) and (
        "healthcare" in low_a
        or "finance" in low_a
        or "banking" in low_a
        or "real-time" in low_a
        or "professional-grade chatbot" in low_a
        or "enterprise-grade chatbot" in low_a
    ):
        failures.append("possible_comparison_hallucination")

    return failures


def has_forbidden_artifact_prefix(text: str) -> bool:
    low = text.lower()
    return any(prefix in low for prefix in FORBIDDEN_ARTIFACT_PREFIXES)


def detect_output_shape_issues(question: str, answer: str) -> list[str]:
    issues: list[str] = []
    normalized = answer.strip()
    low_a = normalized.lower()
    strict_mode = is_high_risk_question(question) or is_policy_question(question)
    max_sentences = 3 if strict_mode else 6
    first_segment = re.split(r"(?<=[.!?])\s+|\n+", normalized, maxsplit=1)[0].strip()

    if not normalized:
        return ["empty_answer"]

    if any(re.search(pattern, low_a) for pattern in ARTIFACT_LEAK_PATTERNS):
        issues.append("artifact_instruction_echo")

    if "```" in normalized or "|" in normalized or "table:" in low_a:
        issues.append("structured_artifact_echo")

    if sentence_count(normalized) > max_sentences:
        issues.append("too_verbose_shape_gate")

    if first_segment.endswith("?"):
        issues.append("answer_starts_with_question")

    # Disallow follow-up questions in answers unless explicitly requested by user.
    if "?" in normalized and "?" not in question:
        issues.append("unexpected_followup_question")

    if any(ord(ch) > 127 for ch in normalized):
        issues.append("non_ascii_artifact")

    if re.search(r"(https?://|www\.|\+\d)", normalized):
        issues.append("contact_or_link_artifact")

    if not re.search(r"[.!?]$", normalized):
        issues.append("unfinished_trailing_fragment")

    return issues


def fallback_for_question(question: str) -> str:
    low_q = question.lower()
    intent = policy_intent(question)
    if intent is not None:
        return policy_response(intent)

    ops_reasoning_intent = ops_intent(question)

    if is_outcome_guarantee_question(question):
        return (
            "No fixed ROI percentage can be guaranteed. "
            "Outcomes depend on implementation scope, execution, and adoption."
        )

    if contains_any(low_q, ["pricing", "price", "cost", "quote"]):
        return (
            "Public pricing details are not confirmed in available information. "
            "Please request an official quote from Synapse Tech."
        )
    if is_general_integration_list_question(question):
        return general_integration_response()
    if is_specific_integration_question(question):
        return (
            "The integration list is not explicitly confirmed in available information. "
            "Please verify supported integrations with Synapse Tech."
        )
    if is_sla_question(question):
        return (
            "Public SLA numbers and exact uptime commitments are not confirmed in available information. "
            "Please verify official SLA terms with Synapse Tech."
        )
    if is_roadmap_question(question):
        return (
            "Roadmap dates and release milestones are not confirmed in available information. "
            "Please verify timelines directly with Synapse Tech."
        )
    if is_cert_compliance_question(question):
        return (
            "Security certifications or compliance approvals are not explicitly confirmed in available information. "
            "Please verify these claims with Synapse Tech."
        )
    if ops_reasoning_intent is not None:
        return ops_response(ops_reasoning_intent)
    if is_legal_question(question):
        return (
            "Legal liability assurances are not confirmed in available information. "
            "Please obtain official contractual terms from Synapse Tech."
        )
    if "roi" in low_q or ("return" in low_q and "investment" in low_q):
        return (
            "Think about ROI in terms of time saved, error reduction, throughput, and service quality against a baseline. "
            "Use a measurement framework rather than assuming a fixed percentage outcome."
        )
    if is_policy_question(question):
        return "If a feature is not confirmed, state it as unconfirmed and do not guess."
    if contains_any(low_q, ["retail", "support team", "kpi", "deploy first"]):
        return (
            "Start with support workflow automation for repetitive requests. "
            "Track first-response time as the first KPI."
        )
    return FALLBACK_RESPONSE


def rule_first_decision(question: str, answer: str, rule_failures: list[str]) -> tuple[bool, list[str]]:
    low_q = question.lower()
    low_a = answer.lower()
    reasons: list[str] = []

    # Hard fail if deterministic failures are present.
    if rule_failures:
        return False, [*rule_failures]

    pricing_question = contains_any(low_q, ["pricing", "price", "cost", "quote"])
    general_integration_question = is_general_integration_list_question(question)
    specific_integration_question = is_specific_integration_question(question)
    outcome_guarantee_question = is_outcome_guarantee_question(question)
    sla_question = is_sla_question(question)
    roadmap_question = is_roadmap_question(question)
    cert_compliance_question = is_cert_compliance_question(question)
    legal_question = is_legal_question(question)
    policy_question = is_policy_question(question)
    comparison_question = is_comparison_question(question)
    ops_reasoning_intent = ops_intent(question)

    unknown_safe_phrases = [
        "not confirmed",
        "not available",
        "not listed",
        "verify with synapse tech",
        "request official quote",
    ]

    if pricing_question:
        if contains_any(low_a, ["yes", "tier", "package", "discount", "subscription"]):
            reasons.append("pricing_claim_without_confirmation")
            return False, reasons
        if not contains_any(low_a, unknown_safe_phrases):
            reasons.append("missing_safe_unknown_pricing_response")
            return False, reasons
        return True, []

    if general_integration_question:
        has_public_examples = contains_any(
            low_a,
            ["airbyte", "n8n", "airtable", "vapi", "dialogflow", "aws", "azure", "workday", "sap"],
        )
        has_list_limitation = contains_any(
            low_a,
            [
                "not clearly published",
                "not clearly listed",
                "not exhaustive",
                "single exhaustive public integration list",
                "specific service or product contexts",
            ],
        )
        if not has_public_examples:
            reasons.append("missing_public_integration_examples")
            return False, reasons
        if not has_list_limitation:
            reasons.append("missing_integration_list_limitation")
            return False, reasons
        return True, []

    if specific_integration_question:
        if not contains_any(low_a, unknown_safe_phrases):
            reasons.append("missing_safe_unknown_integration_response")
            return False, reasons
        return True, []

    if outcome_guarantee_question:
        has_no_guarantee = any(
            re.search(p, low_a)
            for p in [
                r"\bno\b[^.]{0,20}\bguarante",
                r"\bcannot\b[^.]{0,20}\bguarante",
                r"\bcan not\b[^.]{0,20}\bguarante",
                r"\bcan't\b[^.]{0,20}\bguarante",
                r"\bno fixed\b[^.]{0,20}\bguarante",
            ]
        )
        if not has_no_guarantee:
            reasons.append("missing_no_guarantee_statement")
            return False, reasons
        if sentence_count(answer) > 2:
            reasons.append("roi_response_too_long")
            return False, reasons
        return True, []

    if sla_question:
        if contains_any(low_a, ["opira", "aws", "azure", "synapse cloud"]):
            reasons.append("sla_off_domain_reference")
            return False, reasons
        if not contains_any(low_a, unknown_safe_phrases):
            reasons.append("missing_safe_unknown_sla_response")
            return False, reasons
        return True, []

    if roadmap_question:
        if not contains_any(low_a, unknown_safe_phrases):
            reasons.append("missing_safe_unknown_roadmap_response")
            return False, reasons
        return True, []

    if cert_compliance_question:
        if not contains_any(low_a, unknown_safe_phrases):
            reasons.append("missing_safe_unknown_cert_compliance_response")
            return False, reasons
        return True, []

    if legal_question:
        has_legal_guard = contains_any(
            low_a,
            [
                "not confirmed",
                "official contractual terms",
                "verify with synapse tech",
                "cannot guarantee legal liability",
            ],
        )
        if not has_legal_guard:
            reasons.append("missing_safe_unknown_legal_response")
            return False, reasons
        return True, []

    if ops_reasoning_intent is not None:
        if contains_any(low_a, unknown_safe_phrases):
            reasons.append("fallback_on_ops_prompt")
        reasons.extend(validate_ops_response(ops_reasoning_intent, answer))
        if sentence_count(answer) > 3:
            reasons.append("ops_response_too_long")
        if reasons:
            return False, reasons
        return True, []

    if policy_question:
        has_unknown_policy = contains_any(
            low_a,
            [
                "do not guess",
                "don't guess",
                "not confirmed",
                "state it is unknown",
                "avoid speculation",
            ],
        )
        if not has_unknown_policy:
            reasons.append("missing_unknown_handling_policy")
            return False, reasons
        if sentence_count(answer) > 2:
            reasons.append("policy_response_too_long")
            return False, reasons
        return True, []

    if comparison_question:
        has_uncertainty_guard = contains_any(
            low_a,
            [
                "not confirmed",
                "not explicitly confirmed",
                "available information is limited",
                "should be verified",
                "confirm with synapse tech",
            ],
        )
        if not has_uncertainty_guard:
            reasons.append("comparison_missing_uncertainty_guard")
            return False, reasons
        if sentence_count(answer) > 4:
            reasons.append("comparison_response_too_long")
            return False, reasons
        return True, []

    # For non-critical domains, rely on absence of hard rule failures.
    return True, []


def generate_answer(user_input: str) -> str:
    messages = [
        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    strict_mode = is_high_risk_question(user_input) or is_policy_question(user_input)
    return ollama_chat(
        GENERATOR_MODEL,
        messages,
        temperature=0.1,
        num_predict=80 if strict_mode else 180,
        stop=["\n\n", "Answer:", "Note:", "Q:", "You:"],
    )


def judge_answer(user_input: str, answer: str) -> dict[str, Any]:
    judge_user_prompt = (
        f"Question:\n{user_input}\n\n"
        f"Answer:\n{answer}\n\n"
        "Return one single-line JSON object only."
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": judge_user_prompt},
    ]
    raw = ollama_chat(
        JUDGE_MODEL,
        messages,
        temperature=0.0,
        num_predict=140,
        stop=["\n\n"],
    )
    parsed = safe_parse_json(raw)
    contract_checked = validate_judge_contract(parsed)
    rule_failures = run_rule_checks(user_input, answer)
    rule_first_pass, rule_first_reasons = rule_first_decision(user_input, answer, rule_failures)

    judge_reasons = contract_checked.get("reasons", [])
    if not isinstance(judge_reasons, list):
        judge_reasons = ["judge_contract_invalid_reasons"]

    merged_reasons = [*rule_first_reasons, *rule_failures, *judge_reasons]
    merged_pass = rule_first_pass

    return {
        "pass": merged_pass,
        "reasons": merged_reasons,
        "notes": contract_checked.get("notes", ""),
        "rule_failures": rule_failures,
        "judge_pass": bool(contract_checked.get("pass", False)),
        "rule_first_pass": rule_first_pass,
    }


def build_repair_prompt(user_input: str, bad_answer: str, reasons: list[str]) -> str:
    reason_text = ", ".join(reasons) if reasons else "quality_policy_violation"
    strict_mode = is_high_risk_question(user_input) or is_policy_question(user_input)
    max_sentences = 2 if strict_mode else 5
    extra_rules: list[str] = []
    if is_outcome_guarantee_question(user_input):
        extra_rules.append(
            "- For guarantee or ROI questions, use a direct no-guarantee statement and do not offer estimated return ranges."
        )
    ops_reasoning_intent = ops_intent(user_input)
    if ops_reasoning_intent is not None:
        extra_rules.append(
            "- For rollout or advisory questions, give concrete operational guidance rather than saying details are unknown."
        )
    if asks_for_metric(user_input) and asks_for_recommendation(user_input):
        extra_rules.append(
            "- If the question asks for both a recommendation and a KPI or metric, answer both parts explicitly."
        )
    extra_rule_text = "\n".join(extra_rules)
    return f"""You must rewrite the answer to satisfy policy.

Rules:
- Answer ONLY the user's question.
- Use maximum {max_sentences} sentences.
- If detail is unconfirmed, say it is not confirmed in available information.
- Do NOT invent pricing, integrations, SLA, certifications, or roadmap dates.
- No hashtags, no meta text, no "Answer:" or "Note:", no self-questions, no extra topic drift.
- Do not ask additional questions.
- End immediately after the final sentence.
{extra_rule_text}

Question:
{user_input}

Previous bad answer:
{bad_answer}

Failure reasons:
{reason_text}

Return only the corrected final answer.
"""


def log_failure(result: dict[str, Any], user_input: str) -> None:
    FAILURE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "question": user_input,
        "attempts": result["attempts"],
        "used_fallback": result["used_fallback"],
        "first_answer": result["first_answer"],
        "first_verdict": result["first_verdict"],
        "retry_answer": result["retry_answer"],
        "retry_verdict": result["retry_verdict"],
        "final_answer": result["final_answer"],
        "final_verdict": result["final_verdict"],
    }
    with FAILURE_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_with_retry(user_input: str) -> dict[str, Any]:
    if is_small_talk_question(user_input):
        direct = small_talk_response(user_input)
        ok_verdict = {
            "pass": True,
            "reasons": ["small_talk_bypass"],
            "notes": "",
            "rule_failures": [],
            "judge_pass": True,
        }
        return {
            "final_answer": direct,
            "final_verdict": ok_verdict,
            "attempts": 1,
            "first_answer": direct,
            "first_verdict": ok_verdict,
            "retry_answer": None,
            "retry_verdict": None,
            "used_fallback": False,
        }

    intent = policy_intent(user_input)
    if intent is not None:
        direct = policy_response(intent)
        ok_verdict = {
            "pass": True,
            "reasons": [f"policy_bypass:{intent}"],
            "notes": "",
            "rule_failures": [],
            "judge_pass": True,
        }
        return {
            "final_answer": direct,
            "final_verdict": ok_verdict,
            "attempts": 1,
            "first_answer": direct,
            "first_verdict": ok_verdict,
            "retry_answer": None,
            "retry_verdict": None,
            "used_fallback": False,
        }

    first_answer = generate_answer(user_input)
    first_shape_issues = detect_output_shape_issues(user_input, first_answer)
    if has_forbidden_artifact_prefix(first_answer):
        fallback_verdict = {
            "pass": True,
            "reasons": ["fallback_used_forbidden_artifact_prefix_attempt_1"],
            "notes": "",
            "rule_failures": ["forbidden_artifact_prefix"],
            "judge_pass": False,
        }
        return {
            "final_answer": fallback_for_question(user_input),
            "final_verdict": fallback_verdict,
            "attempts": 1,
            "first_answer": first_answer,
            "first_verdict": fallback_verdict,
            "retry_answer": None,
            "retry_verdict": None,
            "used_fallback": True,
        }

    if first_shape_issues:
        first_verdict = {
            "pass": False,
            "reasons": first_shape_issues,
            "notes": "shape_gate_failed_attempt_1",
            "rule_failures": first_shape_issues,
            "judge_pass": False,
            "rule_first_pass": False,
        }
    else:
        first_verdict = judge_answer(user_input, first_answer)
    if first_verdict.get("pass", False):
        return {
            "final_answer": first_answer,
            "final_verdict": first_verdict,
            "attempts": 1,
            "first_answer": first_answer,
            "first_verdict": first_verdict,
            "retry_answer": None,
            "retry_verdict": None,
            "used_fallback": False,
        }

    strict_mode = is_high_risk_question(user_input) or is_policy_question(user_input)
    repair_prompt = build_repair_prompt(
        user_input,
        first_answer,
        first_verdict.get("reasons", []),
    )
    retry_answer = ollama_chat(
        GENERATOR_MODEL,
        [
            {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
            {"role": "user", "content": repair_prompt},
        ],
        temperature=0.0,
        num_predict=60 if strict_mode else 140,
        stop=["\n\n", "Answer:", "Note:", "Q:", "You:"],
    )
    retry_shape_issues = detect_output_shape_issues(user_input, retry_answer)
    if has_forbidden_artifact_prefix(retry_answer):
        fallback_verdict = {
            "pass": True,
            "reasons": ["fallback_used_forbidden_artifact_prefix_attempt_2"],
            "notes": "",
            "rule_failures": ["forbidden_artifact_prefix"],
            "judge_pass": False,
        }
        return {
            "final_answer": fallback_for_question(user_input),
            "final_verdict": fallback_verdict,
            "attempts": 2,
            "first_answer": first_answer,
            "first_verdict": first_verdict,
            "retry_answer": retry_answer,
            "retry_verdict": fallback_verdict,
            "used_fallback": True,
        }

    if retry_shape_issues:
        retry_verdict = {
            "pass": False,
            "reasons": retry_shape_issues,
            "notes": "shape_gate_failed_attempt_2",
            "rule_failures": retry_shape_issues,
            "judge_pass": False,
            "rule_first_pass": False,
        }
    else:
        retry_verdict = judge_answer(user_input, retry_answer)
    if retry_verdict.get("pass", False):
        return {
            "final_answer": retry_answer,
            "final_verdict": retry_verdict,
            "attempts": 2,
            "first_answer": first_answer,
            "first_verdict": first_verdict,
            "retry_answer": retry_answer,
            "retry_verdict": retry_verdict,
            "used_fallback": False,
        }

    fallback_verdict = {
        "pass": True,
        "reasons": ["fallback_used_after_two_failed_attempts"],
        "notes": "",
        "rule_failures": [],
        "judge_pass": True,
    }
    return {
        "final_answer": fallback_for_question(user_input),
        "final_verdict": fallback_verdict,
        "attempts": 2,
        "first_answer": first_answer,
        "first_verdict": first_verdict,
        "retry_answer": retry_answer,
        "retry_verdict": retry_verdict,
        "used_fallback": True,
    }


def main() -> None:
    check_ollama()
    print("=" * 60)
    print("Stage 3: Retry + Fallback Control Flow")
    print("=" * 60)
    print(f"Generator: {GENERATOR_MODEL}")
    print(f"Judge:     {JUDGE_MODEL}")
    print(f"Failure log: {FAILURE_LOG_PATH}")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        try:
            result = run_with_retry(user_input)
        except Exception as exc:
            print(f"\n[Error] {exc}")
            continue

        # Stage 4: persist non-clean cases for future corrective retraining.
        if result["retry_answer"] is not None or result["used_fallback"]:
            log_failure(result, user_input)

        print("\n--- Attempt 1 Answer ---")
        print(result["first_answer"])
        print("\n--- Attempt 1 Verdict ---")
        print(json.dumps(result["first_verdict"], ensure_ascii=False, indent=2))

        if result["retry_answer"] is not None:
            print("\n--- Attempt 2 (Repaired) Answer ---")
            print(result["retry_answer"])
            print("\n--- Attempt 2 Verdict ---")
            print(json.dumps(result["retry_verdict"], ensure_ascii=False, indent=2))

        print("\n--- Final Answer Returned ---")
        print(result["final_answer"])
        print("\n--- Final Metadata ---")
        print(
            json.dumps(
                {
                    "attempts": result["attempts"],
                    "used_fallback": result["used_fallback"],
                    "final_pass": result["final_verdict"]["pass"],
                    "final_reasons": result["final_verdict"]["reasons"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
