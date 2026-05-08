import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

OLLAMA_URL = "http://localhost:11434"

GENERATOR_MODEL = "synapse-qwen1.5b-v5"
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
    return contains_any(low_q, ["legal", "liability", "contractual", "guarantee", "warranty"])


def is_small_talk_question(question: str) -> bool:
    low_q = question.strip().lower()
    small_talk_exact = {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "yo",
        "sup",
        "good morning",
        "good afternoon",
        "good evening",
    }
    if low_q in small_talk_exact:
        return True
    return contains_any(
        low_q,
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
    if contains_any(
        low_q,
        [
            "should you guess",
            "guess or state",
            "if a feature is not confirmed",
            "if information is missing",
            "if info is missing",
            "unconfirmed feature",
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
        r"answer:",
        r"note:",
        r"list the names of",
        r"https?://",
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

    if "integration" in low_q and (
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

    if not normalized:
        return ["empty_answer"]

    if any(re.search(pattern, low_a) for pattern in ARTIFACT_LEAK_PATTERNS):
        issues.append("artifact_instruction_echo")

    if sentence_count(normalized) > max_sentences:
        issues.append("too_verbose_shape_gate")

    # Disallow follow-up questions in answers unless explicitly requested by user.
    if "?" in normalized and "?" not in question:
        issues.append("unexpected_followup_question")

    if any(ord(ch) > 127 for ch in normalized):
        issues.append("non_ascii_artifact")

    if not re.search(r"[.!?]$", normalized):
        issues.append("unfinished_trailing_fragment")

    return issues


def fallback_for_question(question: str) -> str:
    low_q = question.lower()
    intent = policy_intent(question)
    if intent is not None:
        return policy_response(intent)

    if contains_any(low_q, ["pricing", "price", "cost", "quote"]):
        return (
            "Public pricing details are not confirmed in available information. "
            "Please request an official quote from Synapse Tech."
        )
    if "integration" in low_q:
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
    if is_legal_question(question):
        return (
            "Legal liability assurances are not confirmed in available information. "
            "Please obtain official contractual terms from Synapse Tech."
        )
    if "roi" in low_q or ("return" in low_q and "investment" in low_q) or "guarante" in low_q:
        return (
            "No fixed ROI percentage can be guaranteed. "
            "Outcomes depend on implementation scope, execution, and adoption."
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
    integration_question = "integration" in low_q
    roi_question = "roi" in low_q or ("return" in low_q and "investment" in low_q)
    sla_question = is_sla_question(question)
    roadmap_question = is_roadmap_question(question)
    cert_compliance_question = is_cert_compliance_question(question)
    legal_question = is_legal_question(question)
    policy_question = is_policy_question(question)
    comparison_question = is_comparison_question(question)

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

    if integration_question:
        if not contains_any(low_a, unknown_safe_phrases):
            reasons.append("missing_safe_unknown_integration_response")
            return False, reasons
        return True, []

    if roi_question:
        has_no_guarantee = any(
            re.search(p, low_a)
            for p in [
                r"\bno\b[^.]{0,20}\bguarante",
                r"\bcannot\b[^.]{0,20}\bguarante",
                r"\bcan not\b[^.]{0,20}\bguarante",
                r"\bcan't\b[^.]{0,20}\bguarante",
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
    return f"""You must rewrite the answer to satisfy policy.

Rules:
- Answer ONLY the user's question.
- Use maximum {max_sentences} sentences.
- If detail is unconfirmed, say it is not confirmed in available information.
- Do NOT invent pricing, integrations, SLA, certifications, or roadmap dates.
- No hashtags, no meta text, no "Answer:" or "Note:", no self-questions, no extra topic drift.
- Do not ask additional questions.
- End immediately after the final sentence.

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
