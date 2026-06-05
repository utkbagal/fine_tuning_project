from __future__ import annotations

import re
from typing import Any


REQUIRED_SECTIONS = [
    "Severity",
    "FileLine",
    "Category",
    "Issue",
    "WhyItMatters",
    "SuggestedFix",
    "Confidence",
]


def approx_token_count(text: str) -> int:
    # Lightweight approximation to avoid tokenizer dependency in evaluation.
    return max(1, int(len(text) / 4))


def _normalize(text: str) -> set[str]:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return {w for w in cleaned.split() if len(w) > 2}


def lexical_overlap(a: str, b: str) -> float:
    a_terms = _normalize(a)
    b_terms = _normalize(b)
    if not a_terms or not b_terms:
        return 0.0
    inter = a_terms.intersection(b_terms)
    union = a_terms.union(b_terms)
    return round(len(inter) / max(1, len(union)), 4)


def format_adherence_score(text: str) -> float:
    lowered = text.lower()
    hits = 0
    for section in REQUIRED_SECTIONS:
        if f"{section.lower()}:" in lowered:
            hits += 1
    return round(hits / len(REQUIRED_SECTIONS), 4)


def maybe_extract_field(text: str, field_name: str) -> str:
    pattern = rf"{re.escape(field_name)}\s*:\s*(.+)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return ""
    value = match.group(1).strip()
    # Stop at next section marker if present.
    next_section = re.search(r"\n[A-Za-z][A-Za-z_]+\s*:\s*", value)
    if next_section:
        value = value[: next_section.start()].strip()
    return value


def correctness_proxy(prediction: str, expected: dict[str, Any]) -> dict[str, float]:
    expected_out = expected.get("output", {})
    expected_category = str(expected_out.get("Category", "")).strip().lower()
    expected_severity = str(expected_out.get("Severity", "")).strip().lower()

    pred_category = maybe_extract_field(prediction, "Category").lower()
    pred_severity = maybe_extract_field(prediction, "Severity").lower()

    category_match = 1.0 if pred_category == expected_category and pred_category else 0.0
    severity_match = 1.0 if pred_severity == expected_severity and pred_severity else 0.0

    expected_text = " ".join(
        [
            str(expected_out.get("Issue", "")),
            str(expected_out.get("WhyItMatters", "")),
            str(expected_out.get("SuggestedFix", "")),
        ]
    )
    overlap = lexical_overlap(prediction, expected_text)

    return {
        "category_match": category_match,
        "severity_match": severity_match,
        "lexical_overlap": overlap,
    }
