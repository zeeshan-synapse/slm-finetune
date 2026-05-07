import json

from guardrail_stage1 import check_ollama, run_with_retry


def chat() -> None:
    print("=" * 60)
    print("Synapse SLM Guardrailed Chat")
    print("=" * 60)
    print("Type 'exit' to quit.")
    print("Type 'debug' to toggle attempt-level output.")

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

        try:
            result = run_with_retry(user_input)
        except Exception as exc:
            print(f"[Error] {exc}")
            continue

        print(f"\nAssistant: {result['final_answer']}")

        if debug:
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
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )


if __name__ == "__main__":
    check_ollama()
    chat()