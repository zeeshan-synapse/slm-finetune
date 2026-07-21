# BIEK Assistant - Dataset, Model, and Method Summary

## Overview

We built a BIEK-focused local assistant for student-facing questions about the Board of Intermediate Education Karachi website. The system answers using a BIEK knowledge base built from public BIEK website pages, PDFs, forms, date sheets, model papers, notifications, and result documents.

Important clarification: we did not train a new BIEK-only model. BIEK knowledge is supplied through retrieval-augmented generation (RAG), where the model answers from indexed BIEK evidence.

## Dataset

The BIEK dataset was created from public BIEK website content.

Main sources:

- BIEK HTML pages
- BIEK downloadable PDFs
- online forms
- date sheets
- model papers
- notifications and circulars
- result gazettes
- affiliated college and recognition pages
- public navigation links from the BIEK website

Extraction methods:

- HTML text extraction for website pages
- PDF text extraction with `pymupdf`
- OCR with `tesseract_ocr` where PDFs required it
- nav-tree extraction from the visible BIEK website menu

Final BIEK KB corpus:

- 967 total chunks before indexing
- 965 vectors indexed in FAISS

Chunk breakdown:

- 660 student-facing chunks
- 307 nav-backed website coverage chunks

Source material:

- 372 enriched BIEK source documents

The final indexed source file is:

`data/runtime/biek_student_plus_nav_chunks.jsonl`

## Knowledge Base

The BIEK knowledge base is stored locally using FAISS.

Index details:

- FAISS index: `data/knowledge-base/biek/faiss.index`
- metadata: `data/knowledge-base/biek/index_meta.jsonl`
- manifest: `data/knowledge-base/biek/index_manifest.json`
- embedding model: `nomic-embed-text`
- embedding dimension: 768
- similarity metric: normalized cosine similarity

The KB includes both document chunks and nav-backed chunks. The nav-backed layer was added because important pages like Contact Us, Authorized Banks, Board Members, forms, and model papers must be reliably reachable from the public site structure.

## Model

The app uses local Ollama models.

Primary runtime model:

- UI label: `Synapse Qwen 2.5 1.5B V2`
- Ollama tag: `synapse-1.5b-v2`
- base family: `Qwen2.5-1.5B-Instruct`
- embedding model: `nomic-embed-text`

The model generates natural answers after the BIEK KB retrieves the relevant evidence. The BIEK facts are not memorized in the model weights.

## Training Method

The project contains a LoRA fine-tuning pipeline for compact local models.

For the current Synapse Qwen 1.5B V2 model:

- base model: Qwen2.5 1.5B Instruct
- method: LoRA fine-tuning
- framework: MLX-LM
- optimizer: Adam
- iterations: 800
- batch size: 1
- learning rate: `1e-4`
- train/validation split: 90/10
- dataset: `data/mixed_dataset_v2.jsonl`
- total samples: 558
- generated train rows: 502
- adapter output: `models/qwen1.5b-finetuned-v2`
- exported Ollama model: `synapse-1.5b-v2`

For BIEK specifically, the work was not additional fine-tuning. The BIEK implementation was:

1. scrape and extract public BIEK website data
2. enrich and classify documents
3. chunk student-relevant content
4. extract and use the public BIEK nav tree
5. create nav-backed chunks
6. merge student chunks with nav chunks
7. embed chunks with `nomic-embed-text`
8. build a FAISS index
9. answer with the local SLM using retrieved BIEK evidence

## Runtime Flow

When a user asks a BIEK question:

1. The Streamlit app receives the question.
2. BIEK-specific routes check for direct cases such as contact, authorized banks, result lookup, forms, and nav sections.
3. If needed, the question is embedded with `nomic-embed-text`.
4. FAISS retrieves relevant BIEK chunks.
5. BIEK-specific reranking and guards reduce stale or unrelated evidence.
6. The local SLM receives the question plus retrieved evidence.
7. The model generates a grounded answer with links when useful.

## BIEK-Specific Improvements

Key improvements added for BIEK:

- BIEK-only knowledge domain
- nav-tree coverage file
- nav-to-content mapping
- nav-backed KB chunks
- merged student + nav FAISS index
- year-aware reranking for 2026 and 2025 content
- direct handling for Contact Us and Authorized Banks
- result lookup by roll number
- shared-link handling for verification and duplicate admit card forms
- safer handling for chairman, board members, and committees
- cleanup to prevent Synapse-style wording in BIEK mode
- batch evaluation set for BIEK questions

## Evaluation

The BIEK evaluation set tests realistic questions students are likely to ask:

- contact information
- authorized bank
- date sheet
- forms
- verification forms
- duplicate admit card
- model papers
- result lookup
- nav section summaries
- scheme of studies

The best BIEK-oriented run discussed during development reached about 89.8 percent accuracy on 59 questions, with an average response time around 2.82 seconds.

## Final Summary

This is a local BIEK RAG assistant powered by a compact Qwen/Synapse SLM. The model provides the language generation, while the BIEK knowledge comes from a local FAISS index built from scraped and extracted BIEK website content.

This approach is better than BIEK-only fine-tuning for now because BIEK data changes frequently. Updating the KB is faster, safer, and easier to audit than retraining the model every time the website changes.
