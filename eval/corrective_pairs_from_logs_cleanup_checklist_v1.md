# `data/corrective_pairs_from_logs.jsonl` Cleanup Checklist

Goal: remove low-quality training targets before the next fine-tune so the model learns:
- short safe abstentions,
- natural wording,
- correct category fallback behavior,
- no artifact leakage.

## Priority 0: Delete contaminated rows first

Delete these rows completely because they are actively harmful:

- `L16`
  - bad target: tells the model to start with a quoted phrase and "add one visual"
  - teaches instruction-following noise instead of a safe answer
- `L23`
  - bad target: fabricated/garbled SLA explanation
  - teaches hallucinated contract language
- `L24`
  - bad target: "it should be avoided when purchasing its products"
  - off-policy, strange, and not a safe abstention
- `L26`
  - bad target: raw dialogue artifact (`Customer: ... User: Sure!`)
  - contaminates assistant style
- `L29`
  - bad target: unfinished meta answer (`I will provide them after verifying...`)
  - teaches policy leakage and continuation drift

## Priority 1: Merge/remove duplicates and near-duplicates

Keep one clean version only.

### Pricing
- keep `L1`
- delete `L4`
- keep one of `L9` or `L22` after rewrite
- keep `L14`
- keep `L20`

Reason:
- too many near-duplicate pricing rows can overweight one wording pattern
- some variants are verbose or awkward

### Integrations
- keep `L2`
- delete `L10`
- keep one of `L19` or `L30` after rewrite

### ROI
- keep one of `L3` or `L11`
- delete the duplicate

### Security certifications
- keep one clean row from `L12`, `L24`, `L27`
- delete the others

Recommended keep:
- rewrite `L27`

### Roadmap dates
- keep one clean row from `L13`, `L25`, `L28`
- delete the others

Recommended keep:
- rewrite `L25`

### Product comparisons
- keep at most one general comparison abstention from `L15`, `L17`, `L18`
- only keep more than one if the user prompts are meaningfully different and the targets are not identical

## Priority 2: Rewrite policy-question rows

These are not contaminated, but they teach the wrong style because they answer policy questions with generic fallback text.

Rewrite:

- `L5`
  - current target is too generic
  - should become a one-sentence policy answer
- `L6`
  - should explicitly say `No` and forbid template/meta phrases
- `L7`
  - should answer legal/compliance policy directly, not generic unknown fallback
- `L8`
  - should list exactly three things to avoid
- `L21`
  - should answer the "one concise sentence" ask directly instead of repeating the integration fallback

## Priority 3: Normalize remaining wording

For the rows you keep, normalize to one consistent style:

- prefer:
  - `not confirmed in available information`
  - `please verify with Synapse Tech`
  - `please request an official quote from Synapse Tech`
- avoid mixing:
  - `available sources`
  - `current sources`
  - `official channels`
  - `released in available information`

Why:
- inconsistent wording can teach the model to ramble or vary unnaturally on safety responses

## Priority 4: Enforce response-shape rules

For every remaining assistant target in this file:

- max 2 sentences for high-risk abstention responses
- no colons unless user explicitly requested a list
- no quoted fake dialogue
- no placeholders
- no instructions
- no unfinished continuation
- no "According to available information..." unless truly necessary

## Recommended Keep/Rewrite Set

### Safe to keep with minimal or no edits
- `L1`
- `L2`
- `L3` or `L11`
- `L14`
- `L20`
- `L30`

### Keep but rewrite
- `L5`
- `L6`
- `L7`
- `L8`
- `L9` or `L22`
- `L12` or `L27`
- `L13` or `L25`
- `L15` / `L17` / `L18` (reduce to one or rewrite as differentiated prompts)
- `L19`
- `L21`
- `L28` only if rewritten concisely

### Delete
- `L4`
- `L10`
- one of `L3`/`L11`
- extra duplicates among pricing/integration/certification/roadmap rows
- `L16`
- `L23`
- `L24`
- `L26`
- `L29`

## Final acceptance rule for this file

Before using `data/corrective_pairs_from_logs.jsonl` in the next run:

- no raw dialogue artifacts
- no fabricated examples
- no robotic meta-policy tails for simple policy prompts
- no duplicate user prompts with only cosmetic wording changes
- every assistant target must sound like a final user-facing answer

