# Augmented Dataset Audit v1

## Scope
- source file: `data/dataset_augmented.jsonl`
- source rows: `229`
- cleaned output: `data/dataset_augmented_clean_v1.jsonl`
- kept rows: `41`

## Main Issue

The primary problem in `data/dataset_augmented.jsonl` is not classic artifact junk like code blocks or `Answer:` prefixes.

The larger issue is that the file teaches the model to answer with:
- confident product claims,
- sales-style wording,
- speculative capability statements,
- blog-derived generic advice,
- and productized names that are broader than the underlying source pages clearly confirm.

This creates exactly the kind of model behavior seen in recent raw eval failures:
- overconfident `Yes` answers,
- inflated integration/capability claims,
- weak boundary control on what is actually confirmed,
- and broader marketing summaries instead of narrow factual responses.

## High-Risk Patterns Removed

### 1. Speculative capability confirmations
Examples removed:
- direct `Yes` answers for CRM/calendar-style integrations
- support/contact availability claims
- claims that imply a fully confirmed product capability when the source page is more suggestive than explicit

### 2. Blog-derived generic advice
Removed rows derived from blog/article-style content such as:
- digital marketing advice
- hiring video tips
- candidate sourcing software recommendations
- AI writing assistant guidance
- generic consulting best practices

These rows are not well aligned with the target assistant behavior for company-grounded Q&A.

### 3. Productized service naming
Removed rows that turned service-page content into product-like entities, especially:
- `Synapse Cloud AI`

The cloud services material supports AWS/Azure service offerings, but not a clearly defined product entity with that name.

### 4. Marketing-heavy or CTA-heavy answers
Removed rows with patterns such as:
- `competitive advantage`
- broad transformation claims
- conversational sales CTAs
- polished benefit summaries that exceed what should be treated as tightly confirmed

## Keep Criteria Used

Rows were kept only when they were both:

1. directly useful for company/product/service grounding, and
2. narrow enough to avoid teaching speculative confidence.

The kept subset focuses mainly on:
- direct product/service descriptions
- grounded service capability summaries
- some industry-specific value statements that map closely to source pages
- product rows for `Opira AI`, `iRecruit One`, voice agents, automation, and related services

## Phase Impact

This cleanup is intended to reduce:
- raw-model overconfidence,
- invented integrations/contact details,
- generic blog answer patterns,
- and broad sales-style completions.

It is not meant to solve every failure mode by itself.

The main purpose is to stop `data/dataset_augmented.jsonl` from diluting the cleaner corrective work already added in Phase C.
