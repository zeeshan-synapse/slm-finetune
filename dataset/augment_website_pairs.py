import json
import os
import re
import time

import requests

RAW_DIR = "data/raw/cleaned-data"
INPUT_PATH = "data/dataset.json"
OUTPUT_PATH = "data/dataset_augmented.jsonl"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

MAX_RETRIES = 3
RETRY_DELAY = 5
NEW_ITEMS_PER_PAGE = 8
MAX_CONTENT_CHARS = 3500


def simplify_text(content: str) -> str:
    return re.sub(r"\s+", " ", content)[:MAX_CONTENT_CHARS].strip()


def extract_json(text: str):
    text = text.strip()

    fenced = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)

    object_match = re.search(r"\{.*\}", text, re.DOTALL)
    if object_match:
        return object_match.group(0)

    array_match = re.search(r"\[.*\]", text, re.DOTALL)
    return array_match.group(0) if array_match else None


def unwrap_items(data):
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        items = data.get("items")
        if isinstance(items, list):
            return items

    return []


def normalize_messages(msgs):
    if not isinstance(msgs, list) or len(msgs) < 2 or len(msgs) % 2 != 0:
        return None

    clean = []
    for idx, msg in enumerate(msgs):
        if not isinstance(msg, dict):
            return None

        role = msg.get("role")
        content = msg.get("content")
        expected_role = "user" if idx % 2 == 0 else "assistant"

        if role != expected_role or not isinstance(content, str) or not content.strip():
            return None

        clean.append({"role": role, "content": content.strip()})

    return clean


def load_existing_questions(path: str):
    seen = set()
    if not os.path.exists(path):
        return seen

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                msgs = item.get("messages", [])
                if msgs and msgs[0].get("role") == "user":
                    seen.add(msgs[0]["content"].strip().lower())
            except Exception:
                continue
    return seen


def build_prompt(page_name: str, content: str, existing_questions: list[str]) -> str:
    banned = "\n".join(f"- {q}" for q in existing_questions[:30])
    return f"""
You are creating additional grounded training data for a website assistant.

Page name:
{page_name}

Generate exactly {NEW_ITEMS_PER_PAGE} NEW items that are different from the existing questions below.

Rules:
- Every answer must be grounded in the page content only.
- Use natural, conversational English.
- Prefer support-style questions users would actually ask.
- Mix direct questions, clarifying follow-ups, comparisons, and practical use-case questions.
- Answers must be 1-3 sentences.
- No bullet points or numbered lists.
- No invented claims.
- Return valid JSON only.
- Do not repeat or closely paraphrase these existing questions:
{banned}

Return only a JSON object like this:
{{
  "items": [
    {{
      "messages": [
        {{"role": "user", "content": "..."}},
        {{"role": "assistant", "content": "..."}}
      ]
    }}
  ]
}}

Page content:
{content}
"""


def generate_pairs(page_name: str, content: str, existing_questions: list[str]):
    prompt = build_prompt(page_name, simplify_text(content), existing_questions)

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
                    "options": {"temperature": 0.3},
                },
                timeout=120,
            )

            raw = response.json().get("response", "").strip()
            json_text = extract_json(raw)
            if not json_text:
                time.sleep(RETRY_DELAY)
                continue

            data = json.loads(json_text)
            items = unwrap_items(data)
            clean = []
            for item in items:
                if not isinstance(item, dict):
                    continue

                msgs = normalize_messages(item.get("messages"))
                if msgs:
                    clean.append({"messages": msgs})
            if clean:
                return clean

            print("  ❌ No valid structured pairs returned")
        except Exception:
            time.sleep(RETRY_DELAY)

    return []


def main():
    existing_questions = load_existing_questions(INPUT_PATH)
    dataset = []
    seen = set(existing_questions)

    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".txt")]
    for i, file in enumerate(files):
        print(f"[{i+1}/{len(files)}] Processing {file}")
        with open(os.path.join(RAW_DIR, file), "r", encoding="utf-8") as f:
            content = f.read().strip()

        page_name = os.path.splitext(file)[0].replace("-", " ")
        pairs = generate_pairs(page_name, content, sorted(existing_questions))
        if not pairs:
            print("  ❌ No new pairs generated")
            continue

        added = 0

        for item in pairs:
            msgs = item.get("messages")
            if not isinstance(msgs, list) or not msgs:
                continue

            first = msgs[0]
            if not isinstance(first, dict):
                continue

            first_user = first.get("content", "").strip().lower()
            if not first_user:
                continue

            if first_user in seen:
                continue
            seen.add(first_user)
            dataset.append(item)
            added += 1

        print(f"  ✅ Added {added} samples")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Added samples: {len(dataset)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
