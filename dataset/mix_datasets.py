import json
import os
import random

WEBSITE_DATASET = "data/dataset.json"
WEBSITE_AUGMENTED_DATASET = "data/dataset_augmented.jsonl"
CHAT_DATASET = "data/ultrachat_sample.clean.jsonl"
OUTPUT_PATH = "data/mixed_dataset.jsonl"

# Keep website knowledge dominant while still teaching normal chat flow.
WEBSITE_WEIGHT = 0.85
CHAT_WEIGHT = 0.15
SEED = 42


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_sample(item: dict):
    msgs = item.get("messages", [])
    if not isinstance(msgs, list) or len(msgs) < 2:
        return None

    clean = []
    for msg in msgs:
        role = msg.get("role")
        content = msg.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
            return None
        clean.append({"role": role, "content": content.strip()})

    if clean[0]["role"] != "user":
        return None

    return {"messages": clean}


def dedupe(samples: list[dict]):
    seen = set()
    unique = []

    for item in samples:
        norm = normalize_sample(item)
        if not norm:
            continue

        key = tuple((m["role"], m["content"].lower()) for m in norm["messages"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(norm)

    return unique


def main():
    if not os.path.exists(WEBSITE_DATASET):
        raise FileNotFoundError(f"Missing website dataset: {WEBSITE_DATASET}")

    if not os.path.exists(CHAT_DATASET):
        raise FileNotFoundError(f"Missing chat dataset: {CHAT_DATASET}")

    website = load_jsonl(WEBSITE_DATASET)
    if os.path.exists(WEBSITE_AUGMENTED_DATASET):
        website.extend(load_jsonl(WEBSITE_AUGMENTED_DATASET))
    website = dedupe(website)
    chat = dedupe(load_jsonl(CHAT_DATASET))

    if not website:
        raise ValueError("No valid website samples found")

    if not chat:
        raise ValueError("No valid chat samples found")

    random.seed(SEED)

    website_target = len(website)
    chat_target = max(1, round((website_target * CHAT_WEIGHT) / WEBSITE_WEIGHT))
    chat_sample = random.sample(chat, min(chat_target, len(chat)))

    mixed = website + chat_sample
    random.shuffle(mixed)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in mixed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Website samples: {len(website)}")
    print(f"Chat samples: {len(chat_sample)}")
    print(f"Mixed total: {len(mixed)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
