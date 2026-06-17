"""Local Streamlit UI for testing Synapse model/RAG configurations.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
import time
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
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    aw.RAG_GENERATE_ORDINARY_ANSWERS = rag_generated
    aw.RAG_COMBINED_PLANNING = combined_planning

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
            rewrite_model=base_model,
            classifier_model=base_model,
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


def main() -> None:
    st.set_page_config(page_title="Synapse Local Chat Lab", layout="wide")
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

    st.session_state.setdefault("messages", [])

    for item in st.session_state.messages:
        render_message(item, debug_enabled=debug_enabled, show_sources=show_sources)

    question = st.chat_input("Ask a Synapse Tech question")
    if not question:
        return

    with st.spinner("Generating answer..."):
        started = time.perf_counter()
        try:
            result = answer_question(
                question=question,
                model_config=selected_config,
                answer_mode=ANSWER_MODES[mode_label],
                rag_generated=rag_style == "Model-generated",
                combined_planning=combined_planning,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            result = {
                "answer": f"Error: {exc}",
                "generation_model": None,
                "usage": {},
            }
        elapsed_s = time.perf_counter() - started
        result["elapsed_s"] = round(elapsed_s, 3)

    item = {
        "question": question,
        "answer": result.get("answer", ""),
        "mode_label": mode_label,
        "model_label": model_label,
        "generation_model": result.get("generation_model"),
        "elapsed_s": elapsed_s,
        "raw": result,
    }
    st.session_state.messages.append(item)
    st.rerun()


if __name__ == "__main__":
    main()
