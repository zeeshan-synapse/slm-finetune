import os
import json
import requests
import time
import re

# ── Config ─────────────────────────────────────────────────────
RAW_DIR = "data/raw/cleaned-data"
OUTPUT_PATH = "data/dataset.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

MAX_RETRIES = 3
RETRY_DELAY = 5
MIN_CONTENT_LENGTH = 200
MAX_CONTENT_CHARS = 3500
MIN_ANSWER_WORDS = 4
MAX_ANSWER_WORDS = 45
MIN_OVERLAP_WORDS = 2
GENERATION_TEMPERATURE = 0.2

BAD_PATTERNS = [
    "according to the content",
    "according to the provided content",
    "based on the content",
    "from the content",
    "the text says",
    "the article says",
    "i don't know",
    "not sure",
]

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "their", "this", "to", "was", "with", "your", "you", "they", "we", "our",
}


# ── Prompt ─────────────────────────────────────────────────────

def build_prompt(page_name: str, content: str) -> str:
    return f"""
You are generating GROUNDED training data for a website assistant.

Website page:
{page_name}

Your job:
- Create training examples that help an assistant answer questions about THIS page only.
- Keep every answer strictly grounded in the page text.
- If the page does not support a claim, do not invent it.
- Prefer factual support-style answers over slogans or ad copy.
- Use natural English so the assistant can still hold a conversation.

Generate exactly 8 items:
1. 4 factual single-turn Q&A pairs
2. 2 conversational single-turn Q&A pairs
3. 2 multi-turn conversations with 4 messages each

Strict rules:
- Answers must be 1-3 sentences.
- No bullet points, numbered lists, markdown, or tables.
- No made-up facts, numbers, integrations, features, or guarantees.
- No phrases like "according to the content", "the page says", or "from the text above".
- No exaggerated marketing language unless the page explicitly uses it.
- For multi-turn examples, the assistant must stay consistent with previous turns.
- Include a healthy mix of: what it is, who it is for, benefits, constraints, comparisons, and follow-up clarification.
- When a user asks something broader than the page supports, answer narrowly with only what can be grounded in the page.

Return ONLY valid JSON object in this format:

{{
  "items": [
    {{
      "messages": [
        {{"role": "user", "content": "..."}},
        {{"role": "assistant", "content": "..."}}
      ]
    }},
    {{
      "messages": [
        {{"role": "user", "content": "..."}},
        {{"role": "assistant", "content": "..."}},
        {{"role": "user", "content": "..."}},
        {{"role": "assistant", "content": "..."}}
      ]
    }}
  ]
}}

Page content:
{content}
"""


# ── Ollama call ───────────────────────────────────────────────

def simplify_text(content: str) -> str:
    content = re.sub(r"\s+", " ", content)
    return content[:MAX_CONTENT_CHARS].strip()


def tokenize(text: str):
    return [
        t for t in re.findall(r"[a-z0-9]+", text.lower())
        if t not in STOPWORDS and len(t) > 2
    ]


def overlap_score(text: str, source_tokens: set[str]) -> int:
    return len(set(tokenize(text)) & source_tokens)


def is_valid_message(msg: dict) -> bool:
    return (
        isinstance(msg, dict)
        and msg.get("role") in {"user", "assistant"}
        and isinstance(msg.get("content"), str)
        and msg["content"].strip()
    )


def is_valid_sample(item: dict, source_tokens: set[str]) -> bool:
    if "messages" not in item or not isinstance(item["messages"], list):
        return False

    msgs = item["messages"]
    if len(msgs) < 2 or len(msgs) % 2 != 0:
        return False

    roles = [m.get("role") for m in msgs]
    if roles[0] != "user":
        return False

    for idx, msg in enumerate(msgs):
        if not is_valid_message(msg):
            return False
        expected_role = "user" if idx % 2 == 0 else "assistant"
        if msg["role"] != expected_role:
            return False

    assistant_turns = [m["content"].strip() for m in msgs if m["role"] == "assistant"]
    for answer in assistant_turns:
        answer_lower = answer.lower()
        if any(pattern in answer_lower for pattern in BAD_PATTERNS):
            return False

        word_count = len(answer.split())
        if word_count < MIN_ANSWER_WORDS or word_count > MAX_ANSWER_WORDS:
            return False

        if overlap_score(answer, source_tokens) < MIN_OVERLAP_WORDS:
            return False

    return True

def extract_json(text):
    text = text.strip()

    fenced = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)

    object_match = re.search(r"\{.*\}", text, re.DOTALL)
    if object_match:
        return object_match.group(0)

    array_match = re.search(r"\[.*\]", text, re.DOTALL)
    if array_match:
        return array_match.group(0)

    return None


def unwrap_items(data):
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        items = data.get("items")
        if isinstance(items, list):
            return items

    return []


def generate_pairs(page_name: str, content: str):
    content = simplify_text(content)
    prompt = build_prompt(page_name, content)
    source_tokens = set(tokenize(content))

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  🔄 Attempt {attempt}/{MAX_RETRIES}...")

            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {
                        "temperature": GENERATION_TEMPERATURE,
                    }
                },
                timeout=120
            )

            raw = response.json().get("response", "").strip()

            # Extract JSON safely
            json_text = extract_json(raw)

            if not json_text:
                print("  ❌ No JSON found, retrying...")
                time.sleep(RETRY_DELAY)
                continue

            data = json.loads(json_text)
            items = unwrap_items(data)

            # Validate
            clean = []
            for item in items:
                if is_valid_sample(item, source_tokens):
                    clean.append(item)

            if clean:
                return clean

            print("  ❌ JSON parsed but no grounded samples passed validation, retrying...")
            time.sleep(RETRY_DELAY)

        except Exception as e:
            print(f"  ❌ Error: {e}, retrying...")
            time.sleep(RETRY_DELAY)

    print("  ❌ Failed after retries")
    return []


# ── Main ─────────────────────────────────────────────────────

def prepare():
    if not os.path.exists(RAW_DIR):
        print("❌ data/raw directory not found")
        return

    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".txt")]

    if not files:
        print("❌ No .txt files found in data/raw")
        return

    dataset = []
    seen = set()

    print(f"Found {len(files)} files\n")

    for i, file in enumerate(files):
        print(f"[{i+1}/{len(files)}] Processing: {file}")

        with open(os.path.join(RAW_DIR, file), "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Skip small content
        if len(content) < MIN_CONTENT_LENGTH:
            print("  ⚠️ Skipping (too little content)")
            continue

        print(f"  📄 Content length: {len(content)}")

        page_name = os.path.splitext(file)[0].replace("-", " ")
        pairs = generate_pairs(page_name, content)

        if not pairs:
            print("  ❌ No valid pairs generated\n")
            continue

        added = 0

        for p in pairs:
            if "messages" not in p:
                continue

            msgs = p["messages"]

            if len(msgs) < 2:
                continue

            first_user = re.sub(r"\s+", " ", msgs[0]["content"].strip().lower())

            if first_user in seen:
                continue

            seen.add(first_user)

            dataset.append({
                "messages": msgs
            })

            added += 1

        print(f"  ✅ Added {added} samples (total: {len(dataset)})\n")


    # Save dataset
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    print("─" * 50)
    print(f"✅ Dataset ready: {len(dataset)} samples")
    print(f"📁 Saved to: {OUTPUT_PATH}")

    # Preview sample
    print("\nSample:")
    print(json.dumps(dataset[:3], indent=2))


# ── Entry ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Checking Ollama...")

    try:
        requests.get("http://localhost:11434", timeout=5)
        print("✅ Ollama is running\n")
    except:
        print("❌ Run: ollama serve")
        exit()

    prepare()
