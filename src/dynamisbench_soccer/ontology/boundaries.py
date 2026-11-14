"""Reference heuristic for decision-boundary rules.

This is a *baseline* screen over free text: it lets M4 graders and adversarial
controls check that boundary fixtures are classified as intended. It is never the
sole grader for a boundary violation (RES-677 adds rubric and deterministic checks).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from dynamisbench_soccer.ontology.models import Boundaries, BoundaryRule


@dataclass(frozen=True)
class Hit:
    rule_id: str
    severity: str
    pattern: str
    excerpt: str


def _compiled(rule: BoundaryRule) -> list[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in rule.reference_patterns]


def screen(text: str, boundaries: Boundaries) -> list[Hit]:
    """Return every pattern hit in ``text``, sentence by sentence."""
    hits = []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for rule in boundaries.rule:
        for pattern in _compiled(rule):
            for sentence in sentences:
                if pattern.search(sentence):
                    hits.append(Hit(rule.id, rule.severity, pattern.pattern, sentence[:160]))
    return hits


def screenable(boundaries: Boundaries) -> list[BoundaryRule]:
    """Rules that have reference patterns (others are graded deterministically in M4)."""
    return [r for r in boundaries.rule if r.reference_patterns]
