from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InferenceInput:
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceOutput:
    variant: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
