import json
import threading
import time

from guardrail_stage1 import check_ollama, run_with_retry


def is_batch_instruction_line(text: str) -> bool:
    low = text.strip().lower()
    return (
        low.startswith("paste batch questions")
        or low.startswith("running batch of")
        or low.startswith("[")
    )


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
        result = run_with_retry(question)
    finally:
        stop_event.set()
        worker.join(timeout=1.0)
    elapsed = time.perf_counter() - started
    result["_elapsed_s"] = elapsed
    return result


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
    print("\n--- Guardrail Metadata ---")
    print(
        json.dumps(
            {
                "attempts": result["attempts"],
                "used_fallback": result["used_fallback"],
                "attempt_1_reasons": result["first_verdict"]["reasons"],
                "attempt_2_reasons": (
                    result["retry_verdict"]["reasons"] if result["retry_verdict"] else []
                ),
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
        print(f"[{i}/{total}] A: {result['final_answer']}")
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
    print("Synapse SLM Guardrailed Chat")
    print("=" * 60)
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

        print(f"\nAssistant: {result['final_answer']}")

        if debug:
            print_debug(result)


if __name__ == "__main__":
    check_ollama()
    chat()