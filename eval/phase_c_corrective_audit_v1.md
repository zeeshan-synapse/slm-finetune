# Phase C Corrective Audit

## Scope
- audited `data/corrective_pairs_from_logs.jsonl`
- audited `data/corrective_pairs_from_logs_clean_v1.jsonl`
- audited `data/corrective_pairs_manual_high_impact_v1.jsonl`
- audited `data/corrective_pairs.jsonl`
- cross-checked product and service facts against the public source files under `data/raw/manual-data/`

## Key Findings

### 1. `data/corrective_pairs_from_logs.jsonl` is not safe to train on directly
This file still contains actively harmful rows:
- raw dialogue artifacts
- instruction-following noise
- fabricated or garbled SLA language
- unfinished continuation text
- duplicate prompt variants that overweight one fallback wording

Examples:
- customer logos/reference rows with dialogue or template artifacts
- SLA row with invented terminology
- certification row with off-policy purchasing advice

Verdict:
- do not feed this file directly into the next retrain

### 2. `data/corrective_pairs_from_logs_clean_v1.jsonl` is better, but still too abstention-heavy in places
This file is much safer than the raw log export, but some rows still teach the model to under-answer:
- product comparison rows answer with generic abstention even when the site has confirmed high-level differences
- some prompts that should produce a direct policy answer still use a safe fallback pattern

Verdict:
- good source material, but not strong enough to use as-is for the next repair run

### 3. `data/corrective_pairs_manual_high_impact_v1.jsonl` contains useful structure but a few wrong teaching signals
Main issues:
- one product list row names a product variant not clearly confirmed in the source pages
- `iRecruit` interview-related row teaches uncertainty where the site explicitly confirms voice/video AI interviews
- some product rows are too vague and teach weak recall on known-answer prompts

Verdict:
- useful intent coverage, but should not be used unchanged as the authoritative corrective layer

### 4. `data/corrective_pairs.jsonl` is broadly useful for style and safety, but not ideal as the only corrective source
Strengths:
- concise style guidance
- decent unknown-handling patterns
- rollout and KPI prompts

Weaknesses:
- some product-definition rows are too generic
- a few known-fact rows hedge too much instead of answering directly from site data

Verdict:
- good reference pool, but not the clean replacement dataset by itself

## Phase C Output

Created:
- `data/corrective_pairs_phase_c_curated_v1.jsonl`

This file is the Phase C replacement corrective set. It is designed to be:
- short
- user-facing
- fact-grounded
- free of meta artifacts
- stronger on known product/service answers
- stricter on unknown/high-risk abstention

## Design Rules Used For The Curated Set
- known product/service questions answer directly from site-confirmed information
- truly unknown specifics use short non-speculative abstention
- policy questions answer the policy, not a generic fallback
- ops questions give practical rollout/KPI guidance
- no placeholders
- no quoted fake dialogue
- no instruction scaffolding
- no invented numbers, discounts, certifications, roadmap dates, or integrations

## Phase D Wiring Recommendation

Use the Phase C curated file as the corrective layer for the next retrain.

Recommended Phase D behavior:
1. load `data/corrective_pairs_phase_c_curated_v1.jsonl`
2. stop loading `data/corrective_pairs_from_logs.jsonl`
3. stop loading `data/corrective_pairs_from_logs_clean_v1.jsonl`
4. stop loading `data/corrective_pairs_manual_high_impact_v1.jsonl`

Important:
- do not simply append the curated file after the legacy corrective files
- that would keep conflicting targets in the mix and dilute the repair

## Why This Matters

The current failure pattern is not just hallucination. It is also:
- fallback on known-answer prompts
- weak product recall
- artifact contamination
- generic policy answers where direct policy should be learned

The curated Phase C set is meant to teach the model:
- answer known facts when the site supports them
- abstain only when details are truly unconfirmed
- stop cleanly
- stay natural and brief
