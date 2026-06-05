"""
Interactive comparison chat: fine-tuned column uses local KB + RAG for most
questions; base column is a plain Ollama completion (no retrieval). See repo
README for env vars (KB_ANSWER_MODEL), Ollama tags, and KB index paths.
"""
import json
import requests
import threading
import time

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

BASE_MODEL = "qwen-base"
BASE_MODEL_FALLBACK = "qwen2.5:1.5b-instruct"
STOP_SEQUENCES = ["\n\n", "Answer:", "Note:", "Q:", "You:"]

# Printed on startup / batch so the two columns are never ambiguous.
COLUMN_LEGEND = (
    "Fine-tuned column = KB retrieval + grounded answer when applicable; "
    "base column = single LLM call, no KB."
)


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


def fine_tuned_result(question: str) -> dict:
    """
    Fine-tuned column: KB-grounded generation for product questions; guardrail
    fast paths (small talk / policy) unchanged. No judge/retry on KB answers.
    """
    if is_small_talk_question(question) or policy_intent(question) is not None:
        return run_with_retry(question)
    meta = kb_grounded_answer_with_meta(question)
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
    }


def run_with_loader(question: str, prefix: str = "") -> dict:
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
        result = fine_tuned_result(question)
        result["_ft_inference_label"] = _infer_ft_inference_label(result)
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


def print_comparison(result: dict, *, prefix: str = "") -> None:
    label_prefix = f"{prefix} " if prefix else ""
    ft_label = result.get("_ft_inference_label") or _infer_ft_inference_label(result)
    ft_title = _FT_PRIMARY_TITLES.get(ft_label, _FT_PRIMARY_TITLES["ft_model_guardrail"])
    print_answer_block(f"{label_prefix}{ft_title}", result["first_answer"])
    if result["final_answer"] != result["first_answer"]:
        final_title = (
            f"{label_prefix}FINE-TUNED — guardrailed final (judge/retry; no extra KB pass)"
            if ft_label == "kb_rag"
            else f"{label_prefix}FINE-TUNED — guardrailed final (judge/retry)"
        )
        print_answer_block(final_title, result["final_answer"])
    print_answer_block(
        f"{label_prefix}BASE — plain LLM only (no KB, no RAG)",
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


def print_debug(result: dict) -> None:
    ft_path = result.get("_ft_inference_label") or _infer_ft_inference_label(result)
    rewrite = result.get("rewrite") or {}
    kb_usage = dict(result.get("kb_usage") or {})
    if rewrite.get("usage"):
        kb_usage = {"query_rewrite": rewrite.get("usage"), **kb_usage}
    kb_total = sum(
        item.get("total_tokens", 0)
        for item in kb_usage.values()
        if isinstance(item, dict)
    )
    if kb_usage:
        kb_usage["total_tokens"] = kb_total
    print("\n--- Debug (fine-tuned path + attempts) ---")
    print(
        json.dumps(
            {
                "fine_tuned_path": ft_path,
                "attempts": result["attempts"],
                "used_fallback": result["used_fallback"],
                "attempt_1_reasons": result["first_verdict"]["reasons"],
                "attempt_2_reasons": (
                    result["retry_verdict"]["reasons"] if result["retry_verdict"] else []
                ),
                "rewrite": rewrite,
                "token_usage": {
                    "fine_tuned_kb": kb_usage,
                    "base": result.get("base_usage"),
                },
                "elapsed_s": round(result["_elapsed_s"], 2),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_batch(debug: bool) -> None:
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
            result = run_with_loader(question, prefix=prefix)
        except Exception as exc:
            print(f"[{i}/{total}] Error: {exc}")
            continue

        total_time += result["_elapsed_s"]
        if result["used_fallback"]:
            fallback_count += 1

        print(f"[{i}/{total}] Q: {question}")
        print_comparison(result, prefix=prefix)
        print(
            f"[{i}/{total}] status: attempts={result['attempts']}, "
            f"fallback={result['used_fallback']}, elapsed={result['_elapsed_s']:.2f}s\n"
        )
        if debug:
            print_debug(result)
            print()

    avg = total_time / total if total else 0.0
    print("=" * 60)
    print("Batch Summary")
    print("=" * 60)
    print(f"Questions: {total}")
    print(f"Fallbacks: {fallback_count}")
    print(f"Average time/question: {avg:.2f}s")


def chat() -> None:
    print("=" * 60)
    print("Synapse SLM — fine-tuned vs base (KB on fine-tuned only)")
    print("=" * 60)
    print(COLUMN_LEGEND)
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
            run_batch(debug)
            continue

        try:
            result = run_with_loader(user_input)
        except Exception as exc:
            print(f"[Error] {exc}")
            continue

        print_comparison(result)

        if debug:
            print_debug(result)


if __name__ == "__main__":
    check_ollama()
    chat()
