"""Local Streamlit UI for testing Synapse model/RAG configurations.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
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
    OLLAMA_URL,
    is_small_talk_question,
    policy_intent,
    policy_response,
    small_talk_response,
)
from kb_answer import kb_grounded_answer_with_meta  # noqa: E402
import answer_with_kb as aw  # noqa: E402


MODEL_OPTIONS: dict[str, dict[str, Any]] = {
    "Synapse 1.5B V1": {
        "fine_tuned_model": "synapse-1.5b-v1",
        "base_model": "qwen2.5:1.5b-instruct",
        "fine_tune_exists": True,
    },
    "Synapse 1.5B V2": {
        "fine_tuned_model": "synapse-1.5b-v2",
        "base_model": "qwen2.5:1.5b-instruct",
        "fine_tune_exists": True,
    },
    "Synapse Llama V1": {
        "fine_tuned_model": "synapse-llama3-v1",
        "base_model": "llama3:latest",
        "fine_tune_exists": True,
    },
    "Synapse Gemma 3 4B V1": {
        "fine_tuned_model": "synapse-gemma3-4b-v1",
        "base_model": "gemma3:4b",
        "fine_tune_exists": True,
    },
    "Qwen2.5 3B": {
        "fine_tuned_model": None,
        "base_model": "qwen2.5:3b",
        "fine_tune_exists": False,
    },
    "Qwen2.5 7B": {
        "fine_tuned_model": None,
        "base_model": "qwen2.5:7b",
        "fine_tune_exists": False,
    },
}

ANSWER_MODES = {
    "Base": "base_plain",
    "Base + RAG": "base_rag",
    "Fine-tuned": "fine_tuned_plain",
    "Fine-tuned + RAG": "fine_tuned_rag",
}
V2_BUCKETS_PATH = PROJECT_ROOT / "eval" / "prompts" / "v2_50_buckets.json"
DEMO_15_GOLD_PATH = PROJECT_ROOT / "eval" / "prompts" / "demo_15_gold.json"
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
    "provided evidence",
    "return json",
    "system prompt",
    "developer message",
    "rewrite query",
    "according to the prompt",
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


def quick_bypass(question: str) -> dict[str, Any] | None:
    if is_small_talk_question(question):
        return {
            "answer": small_talk_response(question),
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


def answer_question(
    *,
    question: str,
    model_config: dict[str, Any],
    answer_mode: str,
    rag_generated: bool,
    combined_planning: bool,
    fast_rag: bool,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    aw.RAG_GENERATE_ORDINARY_ANSWERS = rag_generated
    aw.RAG_COMBINED_PLANNING = combined_planning
    aw.RAG_FAST_MODE = fast_rag

    if answer_mode.endswith("_rag"):
        bypass = quick_bypass(question)
        if bypass is not None:
            return bypass

    base_model = model_config["base_model"]
    fine_tuned_model = model_config.get("fine_tuned_model")

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

    if answer_mode == "base_rag":
        return kb_grounded_answer_with_meta(
            question,
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


def render_message(item: dict[str, Any], *, debug_enabled: bool, show_sources: bool) -> None:
    with st.chat_message("user"):
        st.write(item["question"])

    with st.chat_message("assistant"):
        st.write(item["answer"])
        st.caption(
            f"Mode: {item['mode_label']} | Model: {item['model_label']} | "
            f"Generator: {item.get('generation_model') or 'quick bypass'} | "
            f"Time: {item.get('elapsed_s', 0.0):.2f}s"
        )

        sources = source_rows(item.get("raw", {}))
        if show_sources and sources:
            st.markdown("**Sources**")
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
            model_config=job["model_config"],
            answer_mode=job["answer_mode"],
            rag_generated=job["rag_generated"],
            combined_planning=job["combined_planning"],
            fast_rag=job["fast_rag"],
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
    model_config: dict[str, Any],
    mode_label: str,
    answer_mode: str,
    model_label: str,
    rag_generated: bool,
    combined_planning: bool,
    fast_rag: bool,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    job = {
        "id": str(time.time_ns()),
        "question": question,
        "model_config": dict(model_config),
        "mode_label": mode_label,
        "answer_mode": answer_mode,
        "model_label": model_label,
        "rag_generated": rag_generated,
        "combined_planning": combined_planning,
        "fast_rag": fast_rag,
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
def load_demo_gold_manifest() -> dict[str, dict[str, Any]]:
    if not DEMO_15_GOLD_PATH.exists():
        return {}
    payload = json.loads(DEMO_15_GOLD_PATH.read_text(encoding="utf-8"))
    rows = payload.get("cases") or []
    return {normalize_text(row.get("question", "")): row for row in rows if row.get("question")}


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
    return (
        "@" in answer
        or "add to cart" in low
        or "checkout" in low
        or bool(re.search(r"\b(?:\+?\d[\d(). -]{7,}\d)\b", answer))
    )


def gold_case_issues(case: dict[str, Any], answer: str) -> list[str]:
    issues: list[str] = []
    low_a = answer.lower()
    if not case.get("allowed_refusal", False) and is_refusal_answer(answer):
        issues.append("over_refusal")
    if case.get("allowed_refusal", False) and not is_refusal_answer(answer):
        issues.append("missing_required_refusal")

    for needle in case.get("must_include") or []:
        if needle.lower() not in low_a:
            issues.append(f"missing_required:{needle}")

    for key in ("must_include_any", "must_include_any_group_2"):
        needles = case.get(key) or []
        if needles and not contains_any(answer, needles):
            issues.append(f"missing_any:{'|'.join(needles)}")

    for needle in case.get("must_not_include") or []:
        if needle.lower() in low_a:
            issues.append(f"forbidden_claim:{needle}")

    return issues


def evaluate_answer(question: str, answer: str, raw: dict[str, Any]) -> dict[str, Any]:
    demo_manifest = load_demo_gold_manifest()
    demo_case = demo_manifest.get(normalize_text(question), {})
    manifest = load_v2_eval_manifest()
    meta = manifest.get(normalize_text(question), {})
    expected_behavior = demo_case.get("expected_behavior") or meta.get("expected_behavior", "")
    bucket = meta.get("bucket", "CUSTOM")
    low_q = question.lower()
    low_a = answer.lower()
    issues: list[str] = []
    verdict = "✅ Pass"

    if demo_case:
        bucket = "DEMO_15_GOLD"
        issues.extend(gold_case_issues(demo_case, answer))

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
    temperature: float,
    max_tokens: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    progress = st.progress(0)
    status = st.empty()
    total = len(questions)
    for index, question in enumerate(questions, start=1):
        status.write(f"Running {index}/{total}: {question}")
        started = time.perf_counter()
        try:
            raw = answer_question(
                question=question,
                model_config=model_config,
                answer_mode=ANSWER_MODES[mode_label],
                rag_generated=rag_generated,
                combined_planning=combined_planning,
                fast_rag=fast_rag,
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
        evaluation = evaluate_answer(question, answer, raw)
        rows.append(
            {
                "question": question,
                "answer": answer,
                "verdict": evaluation["verdict"],
                "issue": evaluation["issue"],
                "bucket": evaluation["bucket"],
                "expected_behavior": evaluation["expected_behavior"],
                "latency_s": round(elapsed_s, 2),
                "generation_model": raw.get("generation_model"),
                "sources": ", ".join(source_rows(raw)),
                "is_refusal": evaluation["is_refusal"],
                "debug": raw,
            }
        )
        progress.progress(index / total)
    status.empty()
    return rows


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
    score = ((pass_count + (partial_count * 0.5)) / total * 100) if total else 0.0
    return {
        "total": total,
        "pass": pass_count,
        "fail": fail_count,
        "partial": partial_count,
        "avg_latency_s": round(avg_latency, 2),
        "refusal_count": refusal_count,
        "score": round(score, 1),
        "issue_counts": issue_counts,
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
        "latency_s",
        "generation_model",
        "sources",
        "is_refusal",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return output.getvalue()


def render_history_card(entry: dict[str, Any]) -> None:
    summary = entry.get("summary") or {}
    score = float(summary.get("score") or 0.0)
    color, label = score_style(score)
    entry_id = str(entry.get("id") or "")
    rag_label = str(entry.get("rag_style") or "Unknown RAG")
    title = (
        f"{entry.get('model_label', 'Unknown model')} | "
        f"{entry.get('mode_label', 'Unknown mode')} | "
        f"{rag_label}"
    )
    config_line = (
        f"Combined planning: {'on' if entry.get('combined_planning') else 'off'} | "
        f"Fast RAG: {'on' if entry.get('fast_rag') else 'off'} | "
        f"Temp: {entry.get('temperature')} | Max tokens: {entry.get('max_tokens')}"
    )
    with st.container(border=True):
        left, right = st.columns([5, 1.5])
        with left:
            st.markdown(f"**{title}**")
            st.caption(entry.get("created_at", ""))
            st.caption(config_line)
            st.markdown(
                f"Total `{summary.get('total', 0)}` · Pass `{summary.get('pass', 0)}` · "
                f"Partial `{summary.get('partial', 0)}` · Fail `{summary.get('fail', 0)}` · "
                f"Avg `{summary.get('avg_latency_s', 0)}s` · Refusals `{summary.get('refusal_count', 0)}`"
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
        rows.append(
            {
                "created_at": entry.get("created_at", ""),
                "model": entry.get("model_label", ""),
                "mode": entry.get("mode_label", ""),
                "rag_behavior": entry.get("rag_style", ""),
                "combined_planning": "on" if entry.get("combined_planning") else "off",
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
    st.caption(f"{row.get('model')} | {row.get('mode')} | {row.get('rag_behavior')}")


def rows_to_csv_report(rows: list[dict[str, Any]]) -> str:
    import csv
    import io

    output = io.StringIO()
    fields = [
        "created_at",
        "model",
        "mode",
        "rag_behavior",
        "combined_planning",
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


def render_batch_history() -> None:
    history = load_batch_history()
    if not history:
        st.info("No saved batch eval runs yet.")
        return

    show_report = st.toggle("Show report", value=False)
    if show_report:
        render_history_report(history)
        st.divider()

    model_options = sorted({str(entry.get("model_label") or "Unknown model") for entry in history})
    mode_options = sorted({str(entry.get("mode_label") or "Unknown mode") for entry in history})
    rag_options = sorted({str(entry.get("rag_style") or "Unknown RAG") for entry in history})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sort_order = st.selectbox(
            "Sort by answer quality",
            ["Original order", "Highest first", "Lowest first"],
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
    for entry in history:
        if model_filter != "All" and entry.get("model_label") != model_filter:
            continue
        if mode_filter != "All" and entry.get("mode_label") != mode_filter:
            continue
        if rag_filter != "All" and entry.get("rag_style") != rag_filter:
            continue
        filtered_history.append(entry)

    if sort_order != "Original order":
        reverse = sort_order == "Highest first"
        filtered_history.sort(
            key=lambda entry: float((entry.get("summary") or {}).get("score") or 0.0),
            reverse=reverse,
        )

    st.caption(f"Showing {len(filtered_history)} of {len(history)} saved runs.")
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
    temperature: float,
    max_tokens: int,
) -> None:
    st.subheader("Batch Eval")
    st.caption("Paste one question per line. The batch uses the current sidebar configuration.")
    show_history = st.toggle("Show history", value=True)
    if show_history:
        render_batch_history()
        st.divider()

    default_questions = (
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
    )
    batch_text = st.text_area("Questions", value=default_questions, height=180)
    questions = [line.strip() for line in batch_text.splitlines() if line.strip()]

    if st.button("Run Batch Eval", type="primary", disabled=not questions):
        with st.spinner(f"Running {len(questions)} questions..."):
            st.session_state.batch_eval_rows = run_batch_eval(
                questions=questions,
                model_config=selected_config,
                mode_label=mode_label,
                model_label=model_label,
                rag_generated=rag_style == "Model-generated",
                combined_planning=combined_planning,
                fast_rag=fast_rag,
                temperature=temperature,
                max_tokens=max_tokens,
            )
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
            )
            add_batch_history_entry(history_entry)
            st.success("Batch eval saved to history.")

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
        model_label = st.selectbox(
            "Model family / selected model",
            list(MODEL_OPTIONS),
            help=(
                "Selects the model pair to test. For fine-tuned modes, this uses the Synapse-tuned "
                "model if it exists. For base modes, it uses the original base model for that family."
            ),
        )
        mode_label = st.selectbox(
            "Answer mode",
            list(ANSWER_MODES),
            index=1,
            help=(
                "Base = original model only. Base + RAG = original model with KB retrieval. "
                "Fine-tuned = Synapse model only. Fine-tuned + RAG = Synapse model with KB retrieval."
            ),
        )
        rag_style = st.radio(
            "RAG behavior",
            ["Deterministic/template", "Model-generated"],
            index=0,
            help=(
                "Deterministic/template uses safer fixed policy answers for known cases. "
                "Model-generated lets the selected LLM write more natural answers from KB evidence, "
                "but it can be slower and may need stronger validation."
            ),
        )
        combined_planning = st.toggle(
            "Combined planning",
            value=True,
            help=(
                "When on, one model call handles both query rewriting and intent classification "
                "before retrieval. This usually reduces latency because the RAG path makes fewer "
                "LLM calls. Turn it off only when comparing against the older separate rewrite + "
                "classifier flow."
            ),
        )
        fast_rag = st.toggle(
            "Fast RAG",
            value=False,
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
                "found the right Synapse page or product before blaming the final model."
            ),
        )

        st.divider()
        st.markdown("**Resolved models**")
        selected_config = MODEL_OPTIONS[model_label]
        st.code(
            json.dumps(
                {
                    "base_model": selected_config["base_model"],
                    "fine_tuned_model": selected_config.get("fine_tuned_model"),
                    "fine_tune_exists": selected_config.get("fine_tune_exists"),
                },
                indent=2,
            ),
            language="json",
        )

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    chat_tab, batch_tab = st.tabs(["Chat", "Batch Eval"])

    with chat_tab:
        st.session_state.setdefault("messages", [])
        st.session_state.setdefault("active_job", None)

        for item in st.session_state.messages:
            render_message(item, debug_enabled=debug_enabled, show_sources=show_sources)

        if st.session_state.active_job:
            render_active_job(debug_enabled=debug_enabled, show_sources=show_sources)
            return

        question = st.chat_input("Ask a Synapse Tech question")
        if question:
            st.session_state.active_job = start_generation_job(
                question=question,
                model_config=selected_config,
                mode_label=mode_label,
                answer_mode=ANSWER_MODES[mode_label],
                model_label=model_label,
                rag_generated=rag_style == "Model-generated",
                combined_planning=combined_planning,
                fast_rag=fast_rag,
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
            temperature=temperature,
            max_tokens=max_tokens,
        )


if __name__ == "__main__":
    main()
