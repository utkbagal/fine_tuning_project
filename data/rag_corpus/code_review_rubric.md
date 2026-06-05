# Code Review Rubric (RAG Seed)

Use this rubric to produce one structured review comment.

## Required Fields

- Severity: Critical | High | Medium | Low
- FileLine: path:line
- Category: Security | Correctness | Performance | Reliability | Style | Maintainability
- Issue: one concise sentence (max 25 words)
- WhyItMatters: impact and consequence
- SuggestedFix: concrete and actionable
- PatchSnippet: optional corrected snippet
- Confidence: float in [0.0, 1.0]

## Severity Guidance

- Critical: exploitable security issue or high blast-radius data integrity failure.
- High: likely production incident, major reliability/correctness risk.
- Medium: clear defect with moderate impact or recoverable failure mode.
- Low: minor maintainability/style risk with low operational impact.

## Category Guidance

- Security: auth/authz, injection, secret handling, unsafe deserialization.
- Correctness: wrong logic, wrong condition, null/None handling errors.
- Performance: unnecessary expensive operations in hot paths.
- Reliability: error handling, retries, idempotency, resource cleanup.
- Maintainability: unclear structure, poor naming, duplicated logic.
- Style: formatting/consistency issues with low functional risk.
