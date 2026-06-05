# Code Review Pattern Examples (RAG Seed)

## Example 1: SQL Injection Risk

- Severity: High
- Category: Security
- Issue: Query string interpolation allows SQL injection via unescaped user input.
- WhyItMatters: Attackers can modify or exfiltrate data by injecting SQL clauses.
- SuggestedFix: Use parameterized queries with bound variables.

## Example 2: Missing Null Guard

- Severity: Medium
- Category: Correctness
- Issue: Accessing nested object fields without null checks can raise runtime exceptions.
- WhyItMatters: Production requests with partial payloads can fail unpredictably.
- SuggestedFix: Add null-safe access and explicit validation fallback.

## Example 3: Resource Leak

- Severity: High
- Category: Reliability
- Issue: File handle is opened but not closed on exception path.
- WhyItMatters: Repeated leaks can exhaust descriptors and degrade service stability.
- SuggestedFix: Use context manager or finally block to guarantee close.

## Example 4: Hot-Path Inefficiency

- Severity: Medium
- Category: Performance
- Issue: Sorting full collection for top-k retrieval causes avoidable O(n log n) work.
- WhyItMatters: Latency increases significantly under larger dataset sizes.
- SuggestedFix: Use heap-based top-k selection or pre-indexed retrieval strategy.
