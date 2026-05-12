#!/usr/bin/env python3
import argparse
import json
import pathlib
import re
from collections import Counter
from typing import Any


DEFAULT_BUCKETS_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "eval/prompts/v2_50_buckets.json"
)

KNOWN_PUBLIC_ABSENCE_PROMPT_IDS = {9, 10, 14, 16}
NO_GUARANTEE_PROMPT_IDS = {11}

ARTIFACT_PATTERNS = [
    r"answer according to",
    r"sentence:",
    r"according to the article",
    r"table below",
    r"relevant material found",
    r"^keyword:",
    r"^direct answer:",
    r"^answer under",
    r"```",
    r"\b[a-d]\)\s",
]

UNKNOWN_PATTERNS = [
    r"not explicitly confirmed",
    r"not confirmed",
    r"not publicly confirmed",
    r"not publicly listed",
    r"not listed",
    r"not available",
    r"not specified",
    r"not stated",
    r"not mentioned",
    r"available information",
    r"available sources",
    r"please verify",
    r"request an official quote",
    r"verify (?:the )?(?:current )?(?:details|list|pricing|integrations?)",
]

PUBLIC_ABSENCE_PATTERNS = [
    r"public pricing",
    r"not publicly listed",
    r"not listed publicly",
    r"not publicly available",
    r"not clearly listed",
    r"request an official quote",
    r"verify .* synapse tech",
    r"not publicly confirmed",
]

NO_GUARANTEE_PATTERNS = [
    r"\bno\b.*\bguarantee",
    r"\bcannot\b.*\bguarantee",
    r"\bcan'?t\b.*\bguarantee",
    r"\bnot guaranteed\b",
    r"\boutcomes depend\b",
    r"\bdepends on\b",
    r"\bvaries by\b",
]

POLICY_KEYWORDS = [
    "guess",
    "unknown",
    "unconfirmed",
    "concise",
    "brief",
    "factual",
    "direct",
    "tone",
    "clarifying",
    "clarify",
    "vague",
    "unsupported",
    "security claims",
    "legal requests",
    "benchmark",
    "final answers",
    "meta",
    "repetitive",
    "response",
]


def has_artifact(text: str) -> bool:
    low = text.lower()
    return any(re.search(pattern, low) for pattern in ARTIFACT_PATTERNS)


def has_loop(text: str) -> bool:
    norm = " ".join(text.lower().split())
    if not norm:
        return False
    chunks = re.split(r"[?.!]\s+", norm)
    chunks = [c.strip() for c in chunks if c.strip()]
    counts = Counter(chunks)
    return any(v >= 2 for v in counts.values())


def likely_hallucination(text: str) -> bool:
    low = text.lower()
    triggers = [
        "yes!",
        "subscription basis",
        "flexible payment",
        "certified",
        "free trial",
        "sales@synapsetech.com",
    ]
    return any(t in low for t in triggers)


def is_empty_response(text: str) -> bool:
    return not text.strip()


def has_non_answer_shape(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if "```" in stripped:
        return True
    first_segment = re.split(r"(?<=[?.!])\s+|\n+", stripped, maxsplit=1)[0].strip()
    if first_segment.endswith("?"):
        return True
    if stripped.count("?") >= 2:
        return True
    if re.search(r"\b[a-d]\)\s", stripped.lower()):
        return True
    return False


def is_unknown_style_response(text: str) -> bool:
    low = text.lower()
    return any(re.search(pattern, low) for pattern in UNKNOWN_PATTERNS)


def is_public_absence_response(text: str) -> bool:
    low = text.lower()
    return any(re.search(pattern, low) for pattern in PUBLIC_ABSENCE_PATTERNS)


def is_no_guarantee_response(text: str) -> bool:
    low = text.lower()
    return any(re.search(pattern, low) for pattern in NO_GUARANTEE_PATTERNS)


def looks_like_policy_answer(text: str) -> bool:
    low = text.lower()
    return any(keyword in low for keyword in POLICY_KEYWORDS)


def short_preview(text: str, limit: int = 220) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bucket-aware heuristic scorer for eval JSONL."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to eval results JSONL file.",
    )
    parser.add_argument(
        "--buckets",
        default=str(DEFAULT_BUCKETS_PATH),
        help="Path to prompt bucket manifest JSON.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write annotated scored JSONL.",
    )
    parser.add_argument(
        "--show-failures",
        type=int,
        default=10,
        help="How many failed rows to print in the console summary.",
    )
    return parser.parse_args()


def load_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_prompt_manifest(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = data.get("prompt_buckets", [])
    return {item["prompt"]: item for item in prompts}


def evaluate_row(row: dict[str, Any], prompt_meta: dict[str, Any] | None) -> dict[str, Any]:
    response = row.get("response", "") or ""
    checks = {
        "artifact": has_artifact(response),
        "loop": has_loop(response),
        "hallucination_flag": likely_hallucination(response),
        "empty": is_empty_response(response),
        "non_answer_shape": has_non_answer_shape(response),
        "unknown_style_response": is_unknown_style_response(response),
        "public_absence_response": is_public_absence_response(response),
        "no_guarantee_response": is_no_guarantee_response(response),
        "policy_like_response": looks_like_policy_answer(response),
    }

    reasons: list[str] = []
    for name in ("artifact", "loop", "hallucination_flag", "empty", "non_answer_shape"):
        if checks[name]:
            reasons.append(name)

    bucket = "UNCLASSIFIED"
    prompt_id = row.get("id")
    expected_behavior = ""
    verdict = "fail"

    if prompt_meta is None:
        reasons.append("prompt_not_in_manifest")
    else:
        bucket = prompt_meta["bucket"]
        prompt_id = prompt_meta["id"]
        expected_behavior = prompt_meta.get("expected_behavior", "")

        if reasons:
            verdict = "fail"
        elif bucket == "KNOWN_FROM_SITE":
            if prompt_id in KNOWN_PUBLIC_ABSENCE_PROMPT_IDS:
                if checks["public_absence_response"] or checks["unknown_style_response"]:
                    verdict = "pass"
                else:
                    verdict = "fail"
                    reasons.append("missed_known_public_absence_answer")
            elif checks["unknown_style_response"]:
                verdict = "fail"
                reasons.append("fallback_on_known_prompt")
            else:
                verdict = "pass"
        elif bucket == "TRULY_UNKNOWN":
            if prompt_id in NO_GUARANTEE_PROMPT_IDS:
                if checks["no_guarantee_response"] or checks["unknown_style_response"]:
                    verdict = "pass"
                else:
                    verdict = "fail"
                    reasons.append("missing_safe_no_guarantee_answer")
            elif checks["unknown_style_response"]:
                verdict = "pass"
            else:
                verdict = "fail"
                reasons.append("missing_safe_unknown_answer")
        elif bucket == "POLICY_BEHAVIOR":
            if checks["unknown_style_response"]:
                verdict = "fail"
                reasons.append("generic_unknown_on_policy_prompt")
            elif checks["policy_like_response"]:
                verdict = "pass"
            else:
                verdict = "fail"
                reasons.append("missing_policy_behavior_answer")
        elif bucket == "GENERAL_OPS_REASONING":
            if checks["unknown_style_response"]:
                verdict = "fail"
                reasons.append("fallback_on_ops_prompt")
            else:
                verdict = "pass"
        else:
            reasons.append("unsupported_bucket")

    return {
        **row,
        "prompt_id": prompt_id,
        "bucket": bucket,
        "expected_behavior": expected_behavior,
        "checks": checks,
        "verdict": verdict,
        "reasons": reasons,
    }


def write_rows(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    input_path = pathlib.Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing eval file: {input_path}")

    bucket_path = pathlib.Path(args.buckets)
    if not bucket_path.exists():
        raise FileNotFoundError(f"Missing bucket manifest: {bucket_path}")

    rows = load_rows(input_path)
    if not rows:
        raise ValueError("No rows found in eval file.")

    prompt_manifest = load_prompt_manifest(bucket_path)
    scored_rows = [evaluate_row(row, prompt_manifest.get(row.get("prompt", ""))) for row in rows]

    if args.output:
        write_rows(pathlib.Path(args.output), scored_rows)

    total = len(scored_rows)
    bucket_totals = Counter(row["bucket"] for row in scored_rows)
    bucket_passes = Counter(row["bucket"] for row in scored_rows if row["verdict"] == "pass")
    failures = [row for row in scored_rows if row["verdict"] != "pass"]

    def count_check(name: str) -> int:
        return sum(1 for row in scored_rows if row["checks"][name])

    def pct(n: int, denom: int = total) -> str:
        if denom == 0:
            return "0.0%"
        return f"{(100.0 * n / denom):.1f}%"

    artifact = count_check("artifact")
    loop = count_check("loop")
    halluc = count_check("hallucination_flag")
    empty = count_check("empty")
    non_answer = count_check("non_answer_shape")
    unknown_style = count_check("unknown_style_response")

    abstention_on_known = sum(
        1
        for row in scored_rows
        if row["bucket"] == "KNOWN_FROM_SITE"
        and row["prompt_id"] not in KNOWN_PUBLIC_ABSENCE_PROMPT_IDS
        and row["checks"]["unknown_style_response"]
    )
    safe_unknown_pass = sum(
        1
        for row in scored_rows
        if row["bucket"] == "TRULY_UNKNOWN" and row["verdict"] == "pass"
    )
    policy_unknown_fail = sum(
        1
        for row in scored_rows
        if row["bucket"] == "POLICY_BEHAVIOR"
        and "generic_unknown_on_policy_prompt" in row["reasons"]
    )
    ops_fallback_fail = sum(
        1
        for row in scored_rows
        if row["bucket"] == "GENERAL_OPS_REASONING"
        and "fallback_on_ops_prompt" in row["reasons"]
    )

    print(f"file: {input_path}")
    print(f"buckets: {bucket_path}")
    print(f"total: {total}")
    print(f"pass_count: {total - len(failures)} ({pct(total - len(failures))})")
    print(f"fail_count: {len(failures)} ({pct(len(failures))})")
    print()
    print("=== Core Failure Signals ===")
    print(f"artifact_count: {artifact} ({pct(artifact)})")
    print(f"loop_count: {loop} ({pct(loop)})")
    print(f"hallucination_flag_count: {halluc} ({pct(halluc)})")
    print(f"empty_count: {empty} ({pct(empty)})")
    print(f"non_answer_shape_count: {non_answer} ({pct(non_answer)})")
    print(f"unknown_style_response_count: {unknown_style} ({pct(unknown_style)})")
    print()
    print("=== Bucket-Aware Signals ===")
    print(
        f"known_prompt_fallback_count: {abstention_on_known} ({pct(abstention_on_known)})"
    )
    print(
        f"truly_unknown_safe_pass_count: {safe_unknown_pass} ({pct(safe_unknown_pass)})"
    )
    print(
        f"policy_prompt_generic_unknown_fail_count: {policy_unknown_fail} ({pct(policy_unknown_fail)})"
    )
    print(
        f"ops_prompt_fallback_fail_count: {ops_fallback_fail} ({pct(ops_fallback_fail)})"
    )
    print()
    print("=== By Bucket ===")
    for bucket in (
        "KNOWN_FROM_SITE",
        "TRULY_UNKNOWN",
        "POLICY_BEHAVIOR",
        "GENERAL_OPS_REASONING",
        "UNCLASSIFIED",
    ):
        bucket_total = bucket_totals.get(bucket, 0)
        if bucket_total == 0:
            continue
        bucket_pass = bucket_passes.get(bucket, 0)
        print(
            f"{bucket}: {bucket_pass}/{bucket_total} pass ({pct(bucket_pass, bucket_total)})"
        )

    reason_counts = Counter(reason for row in failures for reason in row["reasons"])
    if reason_counts:
        print()
        print("=== Top Failure Reasons ===")
        for reason, count in reason_counts.most_common(10):
            print(f"{reason}: {count} ({pct(count)})")

    if args.show_failures > 0 and failures:
        print()
        print("=== Sample Failed Rows ===")
        for row in failures[: args.show_failures]:
            print(
                f"[{row['bucket']}] Q{row['prompt_id']}: {row['prompt']}"
            )
            print(f"  verdict: {row['verdict']}")
            print(f"  reasons: {', '.join(row['reasons'])}")
            print(f"  response: {short_preview(row.get('response', ''))}")

    if args.output:
        print()
        print(f"annotated_output: {args.output}")


if __name__ == "__main__":
    main()
