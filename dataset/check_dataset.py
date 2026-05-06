import json

FILE = "data/dataset.json"

bad_phrases = [
    "according to",
    "the provided content",
    "in conclusion",
    "this article",
    "synapse tech is a leading",
]

MAX_ANSWER_LEN = 180
MIN_ANSWER_LEN = 2


def analyze(q, a):
    issues = []

    # Length checks
    if len(a) > MAX_ANSWER_LEN:
        issues.append(("too long", "Shorten answer to 1-2 sentences"))

    if len(a) < MIN_ANSWER_LEN:
        # allow short conversational replies
        if a.lower() not in ["hi!", "hello!", "ok", "thanks!", "you're welcome!"]:
            issues.append(("too short", "Answer is likely useless"))

    # Phrase checks
    for phrase in bad_phrases:
        if phrase in a.lower():
            issues.append(("bad phrase", f"Remove '{phrase}'"))

    # Style checks
    if a.endswith(":"):
        issues.append(("bad ending", "Answer should be complete sentence"))

    if "?" in a:
        issues.append(("question in answer", "Answer should not ask questions"))

    # Repetition pattern
    if len(a.split()) > 0:
        words = a.lower().split()
        if len(set(words)) < len(words) * 0.6:
            issues.append(("repetition", "Answer may be repetitive"))

    return issues


def main():
    seen = set()
    total = 0
    issues_count = 0

    with open(FILE, "r") as f:
        for i, line in enumerate(f):
            total += 1
            data = json.loads(line)

            try:
                q = data["messages"][0]["content"]
                a = data["messages"][1]["content"]
            except:
                print(f"[{i}] ❌ Invalid format → Fix structure")
                issues_count += 1
                continue

            # Duplicate
            if q in seen:
                print(f"[{i}] ⚠️ Duplicate question")
                print(f"     Fix: Remove duplicate\n")
                issues_count += 1
            seen.add(q)

            # Analyze answer
            problems = analyze(q, a)

            if problems:
                print(f"[{i}] ⚠️ Question: {q}")
                print(f"     Answer: {a}")
                for issue, fix in problems:
                    print(f"     → Issue: {issue}")
                    print(f"       Fix: {fix}")
                print()
                issues_count += 1

    print("──────────────")
    print(f"Total: {total}")
    print(f"Issues: {issues_count}")
    print(f"Quality Score: {(1 - issues_count/total)*100:.2f}%")


if __name__ == "__main__":
    main()