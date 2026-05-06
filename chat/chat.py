import requests
import sys
import json

OLLAMA_URL = "http://localhost:11434"

FINETUNED_MODEL = "synapse-3b"
BASE_MODEL = "base-llama"   # or "qwen-base" if you created it

SYSTEM_PROMPT = """You are a helpful assistant for Synapse Tech Inc.,
a leading AI technology solutions company. Answer questions accurately
based on your knowledge of Synapse Tech's products, services, and industries."""

MAX_TOKENS = 512
TEMPERATURE = 0.7


def check_ollama():
    try:
        requests.get(OLLAMA_URL, timeout=5)
    except:
        print("❌ Ollama is not running.")
        print("Run: ollama serve")
        sys.exit(1)


def generate(model_name, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model_name,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": TEMPERATURE,
                    "num_predict": MAX_TOKENS,
                    "stop": ["<|eot_id|>", "<|start_header_id|>", "user:", "User:"]
                }
            },
            stream=True,
            timeout=120
        )

        response.raise_for_status()

        full = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if "message" in chunk and "content" in chunk["message"]:
                    token = chunk["message"]["content"]
                    print(token, end="", flush=True)
                    full += token
                if chunk.get("done"):
                    break

        return full

    except Exception as e:
        return f"[Error: {e}]"


def compare(user_input, hist_ft, hist_base):
    print("\n" + "="*60)
    print("🟢 FINE-TUNED MODEL (Synapse Tech)")
    print("-"*60)

    hist_ft.append({"role": "user", "content": user_input})
    ft_resp = generate(FINETUNED_MODEL, hist_ft)
    hist_ft.append({"role": "assistant", "content": ft_resp})

    print("\n\n" + "="*60)
    print("🔵 BASE MODEL (not fine-tuned)")
    print("-"*60)

    hist_base.append({"role": "user", "content": user_input})
    base_resp = generate(BASE_MODEL, hist_base)
    hist_base.append({"role": "assistant", "content": base_resp})

    print("\n" + "="*60)


def chat():
    hist_ft = []
    hist_base = []

    print("="*60)
    print("Synapse SLM Comparison")
    print("="*60)

    while True:
        try:
            user_input = input("You: ").strip()
        except:
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit"]:
            break

        if user_input.lower() == "clear":
            hist_ft = []
            hist_base = []
            print("Cleared\n")
            continue

        compare(user_input, hist_ft, hist_base)


if __name__ == "__main__":
    check_ollama()
    chat()