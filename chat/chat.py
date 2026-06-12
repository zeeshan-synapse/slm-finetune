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
    OLLAMA_URL,
    check_ollama,
    is_high_risk_question,
    is_policy_question,
    is_small_talk_question,
    policy_intent,
    run_with_retry,
)

from kb_answer import kb_grounded_answer_with_meta
import answer_with_kb as aw

BASE_MODEL = "qwen2.5:1.5b-instruct"
BASE_MODEL_FALLBACK = "qwen2.5:1.5b-instruct"
FINE_TUNED_V1_MODEL = "synapse-1.5b-v1"
FINE_TUNED_V2_MODEL = "synapse-1.5b-v2"
STOP_SEQUENCES = ["\n\n", "Answer:", "Note:", "Q:", "You:"]
BASE_MODEL_CHOICES = {
    "1": "qwen2.5:1.5b-instruct",
    "2": "qwen2.5:3b",
    "3": "qwen2.5:7b",
    "4": "llama3:latest",
}

# Printed on startup / batch so the three columns are never ambiguous.
COLUMN_LEGEND = (
    "V1/V2 fine-tuned = selected Synapse model + KB/RAG; "
    "base + RAG = untrained base model + the same KB/RAG; "
    "base plain = untrained base model without KB."
)
DISPLAY_MODES = {
    "1": "fine_tuned_v1_rag",
    "2": "fine_tuned_v2_rag",
    "3": "base_rag",
    "4": "base_plain",
    "5": "all",
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
    messages = [
        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 80 if strict_mode else 180,
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
        rewrite_model=BASE_MODEL,
        classifier_model=BASE_MODEL,
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
        if mode in {"fine_tuned_v1_rag", "all"}:
            v1_result = fine_tuned_result(question, FINE_TUNED_V1_MODEL)
            result["fine_tuned_v1"] = v1_result
        if mode in {"fine_tuned_v2_rag", "all"}:
            v2_result = fine_tuned_result(question, FINE_TUNED_V2_MODEL)
            result["fine_tuned_v2"] = v2_result
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
    if mode in {"fine_tuned_v1_rag", "all"}:
        print_answer_block(
            f"{label_prefix}SYNAPSE 1.5B V1 + local KB (FAISS RAG)",
            result["fine_tuned_v1"]["final_answer"],
        )
    if mode in {"fine_tuned_v2_rag", "all"}:
        print_answer_block(
            f"{label_prefix}SYNAPSE 1.5B V2 + local KB (FAISS RAG)",
            result["fine_tuned_v2"]["final_answer"],
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
    v1 = result.get("fine_tuned_v1") or {}
    v2 = result.get("fine_tuned_v2") or {}
    print("\n--- Debug ---")
    print(
        json.dumps(
            {
                "mode": mode,
                "fine_tuned_v1": {
                    "model": FINE_TUNED_V1_MODEL,
                    "observability": v1.get("observability"),
                    "usage": v1.get("kb_usage"),
                },
                "fine_tuned_v2": {
                    "model": FINE_TUNED_V2_MODEL,
                    "observability": v2.get("observability"),
                    "usage": v2.get("kb_usage"),
                },
                "token_usage": {
                    "base_rag": result.get("base_rag_usage"),
                    "base_plain": result.get("base_usage"),
                },
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
    print("1. Synapse 1.5B V1 + RAG")
    print("2. Synapse 1.5B V2 + RAG")
    print("3. Base + RAG")
    print("4. Plain base")
    print("5. All answers")
    while True:
        choice = input("Mode [1-5]: ").strip()
        if choice in DISPLAY_MODES:
            return DISPLAY_MODES[choice]
        print("Enter 1, 2, 3, 4, or 5.")


def choose_base_model() -> str:
    print("Choose pipeline model:")
    print("1. Qwen2.5 1.5B Instruct")
    print("2. Qwen2.5 3B")
    print("3. Qwen2.5 7B")
    print("4. Llama 3 latest")
    while True:
        choice = input("Model [1-4]: ").strip()
        if choice in BASE_MODEL_CHOICES:
            return BASE_MODEL_CHOICES[choice]
        print("Enter 1, 2, 3, or 4.")


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


def chat() -> None:
    global BASE_MODEL, BASE_MODEL_FALLBACK

    print("=" * 60)
    print("Synapse SLM — V1 RAG vs V2 RAG vs base RAG vs plain base")
    print("=" * 60)
    print(COLUMN_LEGEND)
    BASE_MODEL = choose_base_model()
    BASE_MODEL_FALLBACK = BASE_MODEL
    aw.DEFAULT_REWRITE_MODEL = BASE_MODEL
    aw.DEFAULT_CLASSIFIER_MODEL = BASE_MODEL
    print(f"Selected base model: {BASE_MODEL}")
    aw.RAG_GENERATE_ORDINARY_ANSWERS = choose_rag_behavior()
    mode = choose_display_mode()
    print("Type 'exit' to quit.")
    print("Type 'debug' to toggle attempt-level output.")
    print("Type 'batch' to paste multiple questions and run together.")

    debug = False

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

        try:
            result = run_with_loader(user_input, mode=mode)
        except Exception as exc:
            print(f"[Error] {exc}")
            continue

        print_comparison(result, mode=mode)

        if debug:
            print_debug(result, mode)


if __name__ == "__main__":
    check_ollama()
    chat()
