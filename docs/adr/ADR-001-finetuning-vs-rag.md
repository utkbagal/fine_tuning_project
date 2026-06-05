# ADR-001: Fine-Tuning vs RAG for Code Review Comments

## Status
Accepted

## Date
2026-06-04

## Context
The capstone goal is to produce code review comments in a strict structure and style. Prompt-only outputs were inconsistent for structured formatting at scale. RAG improves grounding but does not consistently enforce response schema compliance.

## Decision
Use LoRA fine-tuning on top of `microsoft/Phi-3.5-mini-instruct` as the primary method for structure/style specialization, while keeping RAG as an optional augmentation path.

## Options Considered
1. Prompt engineering only
2. Base model + RAG only
3. LoRA fine-tuning only
4. LoRA fine-tuning + RAG hybrid

## Trade-offs
- Prompt-only:
  - Pros: zero training cost, immediate
  - Cons: weak format consistency
- RAG-only:
  - Pros: fresh context injection
  - Cons: format adherence still variable
- LoRA-only:
  - Pros: strongest structure/style control
  - Cons: retraining required for behavior updates
- LoRA+RAG:
  - Pros: structure from tuning + contextual grounding from retrieval
  - Cons: higher system complexity and latency

## Consequences
- Maintain a training pipeline and adapter versioning strategy
- Keep four-variant evaluation for evidence-based comparison
- Use adapter promotion (`artifacts/models/latest`) to simplify inference routing

## Decision Outcomes to Validate
- Fine-tuned variants should improve format adherence significantly
- Hybrid variant may improve context-sensitive outputs on complex prompts
- Evaluation scorecards will determine production recommendation boundaries
