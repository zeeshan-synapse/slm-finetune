#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import requests
from kb_paths import knowledge_base_dir


DEFAULT_KB_DIR = knowledge_base_dir()
DEFAULT_INPUT_PATH = DEFAULT_KB_DIR / "chunks.jsonl"
DEFAULT_INDEX_PATH = DEFAULT_KB_DIR / "faiss.index"
DEFAULT_META_PATH = DEFAULT_KB_DIR / "index_meta.jsonl"
DEFAULT_MANIFEST_PATH = DEFAULT_KB_DIR / "index_manifest.json"
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_EMBED_MODEL = os.environ.get("KB_EMBED_MODEL", "nomic-embed-text")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a local FAISS index from KB chunks using Ollama embeddings."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Path to chunked KB JSONL.",
    )
    parser.add_argument(
        "--index-output",
        default=str(DEFAULT_INDEX_PATH),
        help="Path to write the FAISS index.",
    )
    parser.add_argument(
        "--meta-output",
        default=str(DEFAULT_META_PATH),
        help="Path to write index metadata JSONL in vector order.",
    )
    parser.add_argument(
        "--manifest-output",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to write index manifest JSON.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBED_MODEL,
        help="Ollama embedding model to use.",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help="Base URL for the local Ollama server.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="How many chunk texts to embed per Ollama request when /api/embed is available.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional max number of chunks to index for smoke testing.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def check_ollama(ollama_url: str) -> None:
    response = requests.get(ollama_url, timeout=5)
    response.raise_for_status()


def embed_batch_via_api_embed(
    ollama_url: str,
    model: str,
    texts: list[str],
) -> list[list[float]]:
    response = requests.post(
        f"{ollama_url}/api/embed",
        json={
            "model": model,
            "input": texts,
        },
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    embeddings = data.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        raise ValueError("Unexpected /api/embed response shape from Ollama.")
    return embeddings


def embed_batch_via_legacy_api(
    ollama_url: str,
    model: str,
    texts: list[str],
) -> list[list[float]]:
    embeddings = []
    for text in texts:
        response = requests.post(
            f"{ollama_url}/api/embeddings",
            json={
                "model": model,
                "prompt": text,
            },
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            raise ValueError("Unexpected /api/embeddings response shape from Ollama.")
        embeddings.append(embedding)
    return embeddings


def describe_row(row: dict[str, Any]) -> str:
    return (
        f"chunk_id={row.get('chunk_id')} "
        f"document_type={row.get('document_type')} "
        f"title={row.get('title')} "
        f"source_url={row.get('source_url')}"
    )


def embed_texts(
    rows: list[dict[str, Any]],
    ollama_url: str,
    model: str,
    batch_size: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    embeddings: list[list[float]] = []
    successful_rows: list[dict[str, Any]] = []
    use_legacy_api = False

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        texts = [row["text"] for row in batch]
        batch_number = start // batch_size + 1
        total_batches = (len(rows) + batch_size - 1) // batch_size
        print(f"Embedding batch {batch_number}/{total_batches} ({len(batch)} chunks)...")

        if use_legacy_api:
            try:
                batch_embeddings = embed_batch_via_legacy_api(ollama_url, model, texts)
            except Exception as exc:
                if len(batch) == 1:
                    print(f"Skipping failed chunk after legacy API error: {describe_row(batch[0])}")
                    print(f"Legacy API error: {exc}")
                    continue
                raise
            embeddings.extend(batch_embeddings)
            successful_rows.extend(batch)
            continue

        try:
            batch_embeddings = embed_batch_via_api_embed(ollama_url, model, texts)
        except Exception as exc:
            print(f"/api/embed failed ({exc}); falling back to legacy /api/embeddings.")
            use_legacy_api = True
            try:
                batch_embeddings = embed_batch_via_legacy_api(ollama_url, model, texts)
            except Exception as legacy_exc:
                if len(batch) == 1:
                    print(f"Skipping failed chunk after both embedding APIs failed: {describe_row(batch[0])}")
                    print(f"Legacy API error: {legacy_exc}")
                    continue
                raise

        embeddings.extend(batch_embeddings)
        successful_rows.extend(batch)

    array = np.asarray(embeddings, dtype="float32")
    if array.ndim != 2 or array.shape[0] != len(successful_rows):
        raise ValueError("Embedding array shape does not match input rows.")
    return array, successful_rows


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def write_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            metadata = dict(row)
            metadata["index_id"] = idx
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")


def write_manifest(
    path: Path,
    *,
    input_path: Path,
    index_path: Path,
    meta_path: Path,
    model: str,
    ollama_url: str,
    total_vectors: int,
    dimension: int,
) -> None:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": model,
        "ollama_url": ollama_url,
        "source_chunks_path": str(input_path),
        "index_path": str(index_path),
        "metadata_path": str(meta_path),
        "total_vectors": total_vectors,
        "dimension": dimension,
        "metric": "cosine_via_normalized_inner_product",
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer when provided.")

    input_path = Path(args.input)
    index_output = Path(args.index_output)
    meta_output = Path(args.meta_output)
    manifest_output = Path(args.manifest_output)

    rows = load_rows(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No chunk rows found to index.")

    index_output.parent.mkdir(parents=True, exist_ok=True)
    meta_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Checking Ollama at: {args.ollama_url}")
    check_ollama(args.ollama_url)
    print(f"Embedding model: {args.model}")
    print(f"Input chunks: {len(rows)}")

    embeddings, embedded_rows = embed_texts(
        rows,
        ollama_url=args.ollama_url,
        model=args.model,
        batch_size=args.batch_size,
    )
    if not embedded_rows:
        raise ValueError("No chunks were successfully embedded.")
    index = build_index(embeddings)

    faiss.write_index(index, str(index_output))
    write_metadata(meta_output, embedded_rows)
    write_manifest(
        manifest_output,
        input_path=input_path,
        index_path=index_output,
        meta_path=meta_output,
        model=args.model,
        ollama_url=args.ollama_url,
        total_vectors=index.ntotal,
        dimension=embeddings.shape[1],
    )

    skipped = len(rows) - len(embedded_rows)
    print(f"FAISS vectors: {index.ntotal}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    if skipped:
        print(f"Skipped chunks: {skipped}")
    print(f"Saved index: {index_output}")
    print(f"Saved metadata: {meta_output}")
    print(f"Saved manifest: {manifest_output}")


if __name__ == "__main__":
    main()
