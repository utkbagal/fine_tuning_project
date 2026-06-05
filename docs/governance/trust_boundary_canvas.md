# Trust Boundary Canvas

## System Boundaries

- Goal: classify each capability into exactly one trust zone for Capstone 4 deployment planning.
- Workload: code review comment generation using base, RAG, and fine-tuned adapter variants.

## Zones

- Zone A: Public/Untrusted Input Boundary
- Zone B: Application Control Plane
- Zone C: Model Runtime Boundary
- Zone D: Data and Artifact Store Boundary
- Zone E: Human Oversight Boundary

## Capability-to-Zone Mapping (exactly one zone each)

| Capability | Zone | Rationale |
|---|---|---|
| Prompt ingress from UI/notebook/CLI | Zone A | User-provided text is untrusted until validated and normalized. |
| Input validation and request shaping | Zone B | Application layer enforces schema/size rules before model invocation. |
| Variant routing logic (base/base+rag/fine-tuned/fine-tuned+rag) | Zone B | Orchestration is a control-plane concern and must remain deterministic. |
| RAG retrieval execution | Zone B | Retrieval policy and source-selection logic are application-governed. |
| Base model inference | Zone C | Model execution happens in isolated runtime with constrained resources. |
| PEFT adapter loading for fine-tuned inference | Zone C | Adapter application alters runtime behavior and is part of model boundary. |
| Output post-processing and format scoring | Zone B | Programmatic quality checks run outside model runtime. |
| Training run execution (QLoRA) | Zone C | GPU training environment is a model-compute trust boundary. |
| Dataset storage (`data/raw`, `data/processed`, `data/golden`) | Zone D | Persistent data assets require lineage and controlled write access. |
| RAG corpus storage (`data/rag_corpus`) | Zone D | Retrieval knowledge base is mutable and must be curated/versioned. |
| Model/adapters/checkpoints (`artifacts/models`, `artifacts/checkpoints`) | Zone D | Serialized model artifacts are critical assets requiring integrity controls. |
| Evaluation reports and verification reports (`artifacts/reports`) | Zone D | Evidence artifacts support governance/audit and must be immutable per run. |
| Human review, sign-off, and release decision | Zone E | Final production decision is a human-governed accountability boundary. |

## Boundary Controls

- Zone A -> Zone B:
  - Prompt length limits and schema shaping.
  - Escaping/normalization for UI rendering.
- Zone B -> Zone C:
  - Explicit model source and adapter path resolution.
  - Controlled generation params (`max_new_tokens`, no arbitrary code execution).
- Zone C -> Zone D:
  - Versioned artifact writes by run id.
  - Latest-adapter promotion only after successful run completion.
- Zone D -> Zone E:
  - Evaluation scorecard and failure-mode evidence required before promotion.

## Release Gate

Production promotion requires all of the following:

1. Backend verification passes for required variants.
2. Evaluation scorecard meets thresholds.
3. Failure-mode checks reviewed for current run.
4. Human sign-off recorded for adapter version.
