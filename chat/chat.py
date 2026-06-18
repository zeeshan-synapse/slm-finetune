"""
Interactive comparison chat:
  1. Synapse V1 fine-tune + local KB/RAG
  2. Synapse V2 fine-tune + local KB/RAG
  3. untrained base model + the same local KB/RAG
  4. untrained base model without retrieval

See the repo README for model and KB configuration.
"""
import json
import requests
import sys
import threading
import time
from pathlib import Path

_CHAT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CHAT_DIR.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
for path in (_CHAT_DIR, _SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from guardrail_stage1 import (
    GENERATOR_SYSTEM_PROMPT,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_URL,
    check_ollama,
    is_high_risk_question,
    is_policy_question,
    is_small_talk_question,
    policy_intent,
    small_talk_response,
    run_with_retry,
)

from kb_answer import kb_grounded_answer_with_meta
import answer_with_kb as aw

BASE_MODEL = "qwen2.5:1.5b-instruct"
BASE_MODEL_FALLBACK = "qwen2.5:1.5b-instruct"
FINE_TUNED_V1_MODEL = "synapse-1.5b-v1"
FINE_TUNED_V2_MODEL = "synapse-1.5b-v2"
FINE_TUNED_LLAMA_MODEL = "synapse-llama3-v1"
FINE_TUNED_GEMMA_MODEL = "synapse-gemma3-4b-v1"
FINE_TUNED_MODEL = FINE_TUNED_V2_MODEL
SELECTED_FINE_TUNE_EXISTS = True
STOP_SEQUENCES = ["\n\n", "Answer:", "Note:", "Q:", "You:"]
FINE_TUNED_MODEL_CHOICES = {
    "1": {
        "label": "Synapse 1.5B V1",
        "fine_tuned_model": FINE_TUNED_V1_MODEL,
        "base_model": "qwen2.5:1.5b-instruct",
    },
    "2": {
        "label": "Synapse 1.5B V2",
        "fine_tuned_model": FINE_TUNED_V2_MODEL,
        "base_model": "qwen2.5:1.5b-instruct",
    },
    "3": {
        "label": "Synapse Llama 3B V1",
        "fine_tuned_model": FINE_TUNED_LLAMA_MODEL,
        "base_model": "llama3:latest",
    },
    "4": {
        "label": "Synapse Gemma 3 4B V1",
        "fine_tuned_model": FINE_TUNED_GEMMA_MODEL,
        "base_model": "gemma3:4b",
    },
    "5": {
        "label": "Qwen2.5 3B",
        "fine_tuned_model": None,
        "base_model": "qwen2.5:3b",
        "fine_tune_exists": False,
    },
    "6": {
        "label": "Qwen2.5 7B",
        "fine_tuned_model": None,
        "base_model": "qwen2.5:7b",
        "fine_tune_exists": False,
    },
}

# Printed on startup / batch so the three columns are never ambiguous.
COLUMN_LEGEND = (
    "Base + Fine-Tuned = selected Synapse model; "
    "Base = original model the Synapse model was trained from; "
    "RAG = local FAISS knowledge-base retrieval."
)
DISPLAY_MODES = {
    "1": "fine_tuned_rag",
    "2": "fine_tuned_plain",
    "3": "base_rag",
    "4": "base_plain",
    "5": "all",
}
FOLLOW_UP_CONFIRMATIONS = {
    "are you sure",
    "are u sure",
    "u sure",
    "you sure",
    "sure",
    "really",
}


def quick_guardrail_observability(question: str) -> dict:
    intent = "small_talk" if is_small_talk_question(question) else "policy"
    event = {
        "question": question,
        "intent": intent,
        "confidence": 1.0,
        "route": "quick_guardrail",
        "rewrite_query": "",
        "sources": [],
        "used_template": True,
        "used_refusal": False,
    }
    aw.log_kb_debug({"stage": "answer_observability", **event})
    return event


def quick_bypass_result(question: str) -> dict | None:
    if is_small_talk_question(question):
        return {
            "answer": small_talk_response(question),
            "usage": {"quick_bypass": None},
            "observability": quick_guardrail_observability(question),
        }
    if policy_intent(question) is not None:
        result = run_with_retry(question)
        return {
            "answer": result["final_answer"],
            "usage": {"quick_bypass": None},
            "observability": quick_guardrail_observability(question),
        }
    return None


def is_confirmation_follow_up(question: str) -> bool:
    normalized = " ".join(question.strip().lower().replace("?", "").split())
    return normalized in FOLLOW_UP_CONFIRMATIONS


def _infer_ft_inference_label(result: dict) -> str:
    """How the fine-tuned side was produced (for titles + debug)."""
    reasons = (result.get("first_verdict") or {}).get("reasons") or []
    if any(r == "kb_grounded_no_guardrail_retry" for r in reasons):
        return "kb_rag"
    if any(r == "small_talk_bypass" for r in reasons):
        return "small_talk_no_kb"
    if any(str(r).startswith("policy_bypass:") for r in reasons):
        return "policy_no_kb"
    return "ft_model_guardrail"


_FT_PRIMARY_TITLES = {
    "kb_rag": "FINE-TUNED + local KB (FAISS RAG)",
    "small_talk_no_kb": "FINE-TUNED — small talk (no KB retrieval)",
    "policy_no_kb": "FINE-TUNED — policy reply (no KB retrieval)",
    "ft_model_guardrail": "FINE-TUNED — model + judge/retry (no KB RAG)",
}


def is_batch_instruction_line(text: str) -> bool:
    low = text.strip().lower()
    return (
        low.startswith("paste batch questions")
        or low.startswith("running batch of")
        or low.startswith("[")
    )


def _usage_from_ollama_response(data: dict, model_name: str) -> dict:
    input_tokens = data.get("prompt_eval_count")
    output_tokens = data.get("eval_count")
    return {
        "model": model_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": (input_tokens or 0) + (output_tokens or 0),
    }


def generate_model_result(model_name: str, question: str) -> dict:
    strict_mode = is_high_risk_question(question) or is_policy_question(question)
    model_profile = aw.get_model_profile(model_name)
    plain_instructions = str(model_profile.get("plain_answer_instructions") or "").strip()
    system_prompt = GENERATOR_SYSTEM_PROMPT
    if plain_instructions:
        system_prompt = f"{system_prompt}\n\nModel-specific instructions:\n{plain_instructions}"
    temperature = float(model_profile["plain_temperature"])
    num_predict = 80 if strict_mode else int(model_profile["plain_num_predict"])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model_name,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "stop": STOP_SEQUENCES,
            },
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return {
        "answer": data.get("message", {}).get("content", "").strip(),
        "usage": _usage_from_ollama_response(data, model_name),
    }


def generate_model_answer(model_name: str, question: str) -> str:
    return generate_model_result(model_name, question)["answer"]


def generate_base_result(question: str) -> dict:
    try:
        return generate_model_result(BASE_MODEL, question)
    except Exception as exc:
        if BASE_MODEL_FALLBACK == BASE_MODEL:
            return {"answer": f"[Base model error: {exc}]", "usage": None}
        try:
            return generate_model_result(BASE_MODEL_FALLBACK, question)
        except Exception as fallback_exc:
            return {
                "answer": (
                    f"[Base model error: {exc}; "
                    f"fallback {BASE_MODEL_FALLBACK} failed: {fallback_exc}]"
                ),
                "usage": None,
            }


def generate_base_answer(question: str) -> str:
    return generate_base_result(question)["answer"]


def fine_tuned_result(question: str, model_name: str) -> dict:
    """
    Fine-tuned column: KB-grounded generation for product questions; guardrail
    fast paths (small talk / policy) unchanged. No judge/retry on KB answers.
    """
    if is_small_talk_question(question):
        generated = generate_model_result(model_name, question)
        answer = generated["answer"]
        verdict = {
            "pass": True,
            "reasons": ["small_talk_no_kb"],
            "notes": "",
            "rule_failures": [],
            "judge_pass": True,
        }
        return {
            "final_answer": answer,
            "final_verdict": verdict,
            "attempts": 1,
            "first_answer": answer,
            "first_verdict": verdict,
            "retry_answer": None,
            "retry_verdict": None,
            "used_fallback": False,
            "kb_usage": {"small_talk_generation": generated.get("usage")},
            "observability": quick_guardrail_observability(question),
        }
    if policy_intent(question) is not None:
        result = run_with_retry(question)
        result["observability"] = quick_guardrail_observability(question)
        return result
    meta = kb_grounded_answer_with_meta(
        question,
        generation_model=model_name,
        rewrite_model=model_name,
        classifier_model=model_name,
    )
    answer = meta["answer"]
    ok_verdict = {
        "pass": True,
        "reasons": ["kb_grounded_no_guardrail_retry"],
        "notes": "",
        "rule_failures": [],
        "judge_pass": True,
    }
    return {
        "final_answer": answer,
        "final_verdict": ok_verdict,
        "attempts": 1,
        "first_answer": answer,
        "first_verdict": ok_verdict,
        "retry_answer": None,
        "retry_verdict": None,
        "used_fallback": False,
        "kb_usage": meta.get("usage", {}),
        "rewrite": meta.get("rewrite"),
        "classification": meta.get("classification"),
        "answer_policy": meta.get("answer_policy"),
        "observability": meta.get("observability"),
    }


def base_rag_result(question: str) -> dict:
    """Run the same KB/RAG pipeline with only the generation model changed."""
    bypass = quick_bypass_result(question)
    if bypass is not None:
        return bypass
    return kb_grounded_answer_with_meta(
        question,
        generation_model=BASE_MODEL,
        rewrite_model=BASE_MODEL,
        classifier_model=BASE_MODEL,
    )


def run_with_loader(question: str, prefix: str = "", mode: str = "all") -> dict:
    stop_event = threading.Event()

    def spinner() -> None:
        frames = ["|", "/", "-", "\\"]
        idx = 0
        label = f"{prefix} Processing"
        while not stop_event.is_set():
            print(f"\r{label} {frames[idx % len(frames)]}", end="", flush=True)
            idx += 1
            time.sleep(0.1)
        print("\r" + " " * (len(label) + 6) + "\r", end="", flush=True)

    worker = threading.Thread(target=spinner, daemon=True)
    worker.start()
    started = time.perf_counter()
    try:
        result = {"attempts": 1, "used_fallback": False}
        if mode in {"fine_tuned_rag", "all"}:
            if SELECTED_FINE_TUNE_EXISTS:
                ft_rag = fine_tuned_result(question, FINE_TUNED_MODEL)
                result["fine_tuned_rag"] = ft_rag
            else:
                result["fine_tuned_rag_unavailable"] = True
        if mode in {"fine_tuned_plain", "all"}:
            if SELECTED_FINE_TUNE_EXISTS:
                ft_plain = generate_model_result(FINE_TUNED_MODEL, question)
                result["fine_tuned_plain_answer"] = ft_plain["answer"]
                result["fine_tuned_plain_usage"] = ft_plain.get("usage")
            else:
                result["fine_tuned_plain_unavailable"] = True
        if mode in {"base_rag", "all"}:
            base_rag = base_rag_result(question)
            result["base_rag_answer"] = base_rag["answer"]
            result["base_rag_usage"] = base_rag.get("usage", {})
            result["base_rag_observability"] = base_rag.get("observability")
        if mode in {"base_plain", "all"}:
            base_result = generate_base_result(question)
            result["base_answer"] = base_result["answer"]
            result["base_usage"] = base_result.get("usage")
    finally:
        stop_event.set()
        worker.join(timeout=1.0)
    elapsed = time.perf_counter() - started
    result["_elapsed_s"] = elapsed
    return result


def print_answer_block(title: str, answer: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("-" * 60)
    print(answer)


def print_comparison(result: dict, *, prefix: str = "", mode: str = "all") -> None:
    label_prefix = f"{prefix} " if prefix else ""
    if mode in {"fine_tuned_rag", "all"}:
        if SELECTED_FINE_TUNE_EXISTS:
            print_answer_block(
                f"{label_prefix}BASE + FINE-TUNED {FINE_TUNED_MODEL} + local KB (FAISS RAG)",
                result["fine_tuned_rag"]["final_answer"],
            )
        else:
            print_answer_block(
                f"{label_prefix}BASE + FINE-TUNED + local KB (FAISS RAG)",
                f"Fine-tuned version for {BASE_MODEL} does not exist in this project.",
            )
    if mode in {"fine_tuned_plain", "all"}:
        if SELECTED_FINE_TUNE_EXISTS:
            print_answer_block(
                f"{label_prefix}BASE + FINE-TUNED {FINE_TUNED_MODEL} — plain model only (no KB)",
                result["fine_tuned_plain_answer"],
            )
        else:
            print_answer_block(
                f"{label_prefix}BASE + FINE-TUNED — plain model only (no KB)",
                f"Fine-tuned version for {BASE_MODEL} does not exist in this project.",
            )
    if mode in {"base_rag", "all"}:
        print_answer_block(
            f"{label_prefix}BASE {BASE_MODEL} + local KB (same FAISS RAG)",
            result["base_rag_answer"],
        )
    if mode in {"base_plain", "all"}:
        print_answer_block(
            f"{label_prefix}BASE {BASE_MODEL} — plain LLM only (no KB)",
            result["base_answer"],
        )
    print("\n" + "=" * 60)


def collect_batch_questions() -> list[str]:
    print("\nPaste batch questions (one per line). Type 'END' on a new line to run.")
    questions: list[str] = []
    while True:
        try:
            line = input().strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line == "END":
            break
        if line and not is_batch_instruction_line(line):
            questions.append(line)
    return questions


def print_debug(result: dict, mode: str) -> None:
    ft_rag = result.get("fine_tuned_rag") or {}
    print("\n--- Debug ---")
    print(
        json.dumps(
            {
                "mode": mode,
                "fine_tuned_rag": {
                    "model": FINE_TUNED_MODEL,
                    "observability": ft_rag.get("observability"),
                    "usage": ft_rag.get("kb_usage"),
                },
                "token_usage": {
                    "fine_tuned_plain": result.get("fine_tuned_plain_usage"),
                    "base_rag": result.get("base_rag_usage"),
                    "base_plain": result.get("base_usage"),
                },
                "selected_fine_tuned_model": FINE_TUNED_MODEL,
                "selected_fine_tune_exists": SELECTED_FINE_TUNE_EXISTS,
                "selected_pipeline_model": BASE_MODEL,
                "selected_model_profile": aw.get_model_profile(BASE_MODEL),
                "base_rag_observability": result.get("base_rag_observability"),
                "elapsed_s": round(result["_elapsed_s"], 2),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_batch(debug: bool, mode: str) -> None:
    questions = collect_batch_questions()
    if not questions:
        print("No questions provided.")
        return

    total = len(questions)
    fallback_count = 0
    total_time = 0.0
    print(f"\nRunning batch of {total} questions...\n")
    print(COLUMN_LEGEND)
    print()

    for i, question in enumerate(questions, start=1):
        prefix = f"[{i}/{total}]"
        try:
            result = run_with_loader(question, prefix=prefix, mode=mode)
        except Exception as exc:
            print(f"[{i}/{total}] Error: {exc}")
            continue

        total_time += result["_elapsed_s"]
        if result["used_fallback"]:
            fallback_count += 1

        print(f"[{i}/{total}] Q: {question}")
        print_comparison(result, prefix=prefix, mode=mode)
        print(
            f"[{i}/{total}] status: attempts={result['attempts']}, "
            f"fallback={result['used_fallback']}, elapsed={result['_elapsed_s']:.2f}s\n"
        )
        if debug:
            print_debug(result, mode)
            print()

    avg = total_time / total if total else 0.0
    print("=" * 60)
    print("Batch Summary")
    print("=" * 60)
    print(f"Questions: {total}")
    print(f"Fallbacks: {fallback_count}")
    print(f"Average time/question: {avg:.2f}s")


def choose_display_mode() -> str:
    print("Choose answer mode:")
    print("1. Base + Fine-Tuned + RAG")
    print("2. Base + Fine-Tuned")
    print("3. Base + RAG")
    print("4. Base")
    print("5. All Answers / Ground Truth")
    while True:
        choice = input("Mode [1-5]: ").strip()
        if choice in DISPLAY_MODES:
            return DISPLAY_MODES[choice]
        print("Enter 1, 2, 3, 4, or 5.")


def choose_fine_tuned_model() -> dict:
    print("Choose model:")
    print("1. Synapse 1.5B V1")
    print("2. Synapse 1.5B V2")
    print("3. Synapse Llama 3B V1")
    print("4. Synapse Gemma 3 4B V1")
    print("5. Qwen2.5 3B")
    print("6. Qwen2.5 7B")
    while True:
        choice = input("Model [1-6]: ").strip()
        if choice in FINE_TUNED_MODEL_CHOICES:
            return FINE_TUNED_MODEL_CHOICES[choice]
        print("Enter 1, 2, 3, 4, 5, or 6.")


def choose_rag_behavior() -> bool:
    print("Choose RAG behavior:")
    print("1. Deterministic/template RAG")
    print("2. Model-generated RAG")
    while True:
        choice = input("RAG mode [1-2]: ").strip()
        if choice == "1":
            return False
        if choice == "2":
            return True
        print("Enter 1 or 2.")


def choose_fast_rag() -> bool:
    print("Fast RAG:")
    print("1. Off — allow retry/correction calls")
    print("2. On — skip retry/correction calls")
    while True:
        choice = input("Fast RAG [1-2]: ").strip()
        if choice == "1":
            return False
        if choice == "2":
            return True
        print("Enter 1 or 2.")


def chat() -> None:
    global BASE_MODEL, BASE_MODEL_FALLBACK, FINE_TUNED_MODEL, SELECTED_FINE_TUNE_EXISTS

    print("=" * 60)
    print("Synapse SLM — fine-tuned vs base comparison")
    print("=" * 60)
    print(COLUMN_LEGEND)
    model_choice = choose_fine_tuned_model()
    FINE_TUNED_MODEL = model_choice["fine_tuned_model"]
    BASE_MODEL = model_choice["base_model"]
    SELECTED_FINE_TUNE_EXISTS = bool(model_choice.get("fine_tune_exists", True))
    BASE_MODEL_FALLBACK = BASE_MODEL
    aw.DEFAULT_REWRITE_MODEL = BASE_MODEL
    aw.DEFAULT_CLASSIFIER_MODEL = BASE_MODEL
    if SELECTED_FINE_TUNE_EXISTS:
        print(f"Selected fine-tuned model: {FINE_TUNED_MODEL}")
        print(f"Matched base model: {BASE_MODEL}")
    else:
        print(f"Selected base model: {BASE_MODEL}")
        print(f"Fine-tuned version: not available in this project")
    aw.RAG_GENERATE_ORDINARY_ANSWERS = choose_rag_behavior()
    aw.RAG_FAST_MODE = choose_fast_rag()
    mode = choose_display_mode()
    print("Type 'exit' to quit.")
    print("Type 'debug' to toggle attempt-level output.")
    print("Type 'batch' to paste multiple questions and run together.")

    debug = False
    last_context_question = ""

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            break
        if user_input.lower() == "debug":
            debug = not debug
            print(f"Debug mode: {'ON' if debug else 'OFF'}")
            continue
        if user_input.lower() == "batch":
            run_batch(debug, mode)
            continue

        effective_input = user_input
        if is_confirmation_follow_up(user_input) and last_context_question:
            effective_input = (
                f"The user is asking whether your previous answer to this question was correct: "
                f"{last_context_question}. Verify it from the KB and correct it if needed."
            )

        try:
            result = run_with_loader(effective_input, mode=mode)
        except Exception as exc:
            print(f"[Error] {exc}")
            continue

        print_comparison(result, mode=mode)
        if not is_small_talk_question(user_input) and not is_confirmation_follow_up(user_input):
            last_context_question = user_input

        if debug:
            print_debug(result, mode)


if __name__ == "__main__":
    check_ollama()
    chat()
