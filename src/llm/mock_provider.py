"""
Offline mock LLM provider for the UMiKT-GAT pipeline.

This is the DEFAULT and PRIMARY evaluation path. It uses keyword/rule matching
against the misconception taxonomy and does NOT make any API calls.

The mock provider ensures:
  1. The full ML pipeline can be trained and evaluated with zero API cost.
  2. Results are deterministic and reproducible.
  3. The LLM extraction component is testable in isolation.

Accuracy note: The mock provider is deliberately simple — it matches keywords
and patterns. Its extraction quality is lower than a real LLM. This is:
  (a) Expected and documented.
  (b) Intentional for the offline fallback path.
  (c) The reason why we separately evaluate extraction quality on the
      handwritten Q&A set (see tests/test_schema_validation.py).

NEVER use the mock provider's extraction metrics as evidence for the quality
of real LLM-based extraction.
"""

import re
from .base_provider import LLMProvider
from .schema import ExtractionRequest, ExtractionResult


# ── Keyword patterns for each misconception class ────────────────────────────
# Patterns are applied to the student_answer text (lowercased).
# These are heuristics, not exhaustive — false positives and negatives expected.
MISCONCEPTION_PATTERNS = {
    "syntax_misunderstanding": [
        r"\bsame\b.*\bequals?\b",
        r"\bequals?\b.*\bsame\b",
        r"\bassignment\b.*\bcomparison\b",
        r"\bif\s+\w+\s*=\s*\w+\b",  # 'if x = y' style
        r"\bsyntax\s+error\b.*\bno\b",
        r"\bboth.*print\b",  # "prints both branches"
        r"\bboth.*execute\b",
        r"\bcorrect\b.*\bnothing\s+wrong\b",
    ],
    "variable_scope_misunderstanding": [
        r"\bglobal\b.*\bno\b",
        r"\bcan.*access.*anywhere\b",
        r"\boutside.*function\b.*\baccessible\b",
        r"\bpersist\b",
        r"\bremain\b.*\bafter\b.*\bcall\b",
        r"\bavailable.*program\b",
    ],
    "loop_termination_misunderstanding": [
        r"\b1\s+to\s+5\b",
        r"\bstarts?\s+(at\s+)?1\b",
        r"\brange.*1.*5\b",
        r"\bnoth\w+\s+wrong\b.*\bwhile\b",
        r"\bwill\s+print\b.*\b[12345]\b.*\bstop\b",
        r"\bfront\b",  # "removes from front" for stacks
        r"\bfaster\b.*\bgreedy\b",  # misunderstanding termination
    ],
    "function_parameter_misunderstanding": [
        r"\bmodif\w+.*\bcaller\b",
        r"\bx\s*=\s*6\b",           # classic increment(x) answer
        r"\bchange\w*.*\bvariable\b.*\boutside\b",
        r"\bsame.*argument\b.*\bpositional\b",
    ],
    "recursion_base_case_misunderstanding": [
        r"\bnoth\w+\s+wrong\b.*\bcount\w*\s+down\b",
        r"\bstop\w*.*\b0\b",        # "stops at 0"
        r"\bglobal\s+variable\b.*\brecurs\w+\b",
        r"\bback\s+to\s+the\s+start\b",
        r"\bstart\w*\s+point\b",
        r"\bunwind\b.*\brun\s+again\b",
    ],
    "reference_vs_value_misunderstanding": [
        r"\bcopy\b.*\b=\b",         # "b = a makes a copy"
        r"\bb\s*=\s*a\b.*\bcopy\b",
        r"\bindependent\b.*\b=\b",
        r"\by.*change.*x\s+change\b",
        r"\bdynamic\w*\s+link\w*\b",
        r"\bin.place\b.*\bmerge\b",
        r"\bmerge\s+sort\b.*\bin.place\b",
    ],
}

# Severity heuristics based on the confidence of pattern match
# and some answer characteristics
SEVERITY_KEYWORDS_HIGH = [
    r"\bnothing\s+wrong\b",
    r"\bcorrect\b.*\bwill\s+work\b",
    r"\bworks?\s+fine\b",
    r"\balways\b.*\bsame\b",
]

SEVERITY_KEYWORDS_LOW = [
    r"\bnot\s+sure\b",
    r"\bmaybe\b",
    r"\bthink\b",
    r"\bprobably\b",
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)


def _count_matches(text: str, patterns: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for p in patterns if re.search(p, text_lower))


class MockLLMProvider(LLMProvider):
    """
    Offline rule-based mock LLM provider.

    Uses keyword/regex matching to estimate correctness and detect misconceptions.
    Does NOT make any API calls. Fully deterministic and reproducible.

    Used as the default provider in all training and evaluation runs.
    Gated by: always available (no env vars required).
    """

    def __init__(self, max_retries: int = 0):
        super().__init__(max_retries=max_retries)

    def _extract_raw(self, request: ExtractionRequest) -> dict:
        """
        Apply rule-based extraction.

        Correctness estimation:
          - If expected_answer is provided, use a fuzzy match heuristic.
          - Otherwise, use a conservative 0.5 (neutral).

        Misconception detection:
          - Apply all pattern dictionaries to the student answer.
          - Select the best-matching class.
          - Estimate severity from answer tone.
        """
        answer = request.student_answer.lower().strip()
        expected = (request.expected_answer or "").lower().strip()

        # ── Correctness estimation ────────────────────────────────────────────
        correctness = self._estimate_correctness(answer, expected, request)

        # ── Misconception detection ───────────────────────────────────────────
        mc_label, mc_severity, mc_confidence = self._detect_misconception(
            answer, request.concept_id
        )

        # ── Reasoning quality heuristic ───────────────────────────────────────
        # Longer answers with domain vocabulary tend to show better reasoning
        reasoning_quality = self._estimate_reasoning_quality(answer)

        # ── Overall provider confidence ───────────────────────────────────────
        # Mock provider has lower confidence than a real LLM
        # Expected answer available → higher confidence
        confidence = 0.65 if expected else 0.50

        return {
            "correctness": round(correctness, 3),
            "concepts_detected": [request.concept_name] if request.concept_name else [],
            "misconception_label": mc_label,
            "misconception_severity": round(mc_severity, 3),
            "reasoning_quality": round(reasoning_quality, 3),
            "confidence": round(confidence, 3),
            "reasoning_trace": f"[MockProvider] concept={request.concept_name}, "
                               f"mc={mc_label}, sev={mc_severity:.2f}",
        }

    def _estimate_correctness(
        self, answer: str, expected: str, request: ExtractionRequest
    ) -> float:
        """Heuristic correctness estimation."""
        if not expected:
            # No rubric: check if answer is non-trivially long
            if len(answer) < 3:
                return 0.1
            return 0.5

        # Token overlap heuristic
        answer_tokens = set(answer.split())
        expected_tokens = set(expected.split())
        stop_words = {"the", "a", "an", "is", "are", "in", "of", "to", "and",
                      "or", "it", "this", "that", "be", "will", "not", "can"}
        answer_tokens -= stop_words
        expected_tokens -= stop_words

        if not expected_tokens:
            return 0.5

        overlap = len(answer_tokens & expected_tokens)
        precision = overlap / max(len(answer_tokens), 1)
        recall = overlap / len(expected_tokens)

        if precision + recall == 0:
            return 0.1
        f1 = 2 * precision * recall / (precision + recall)
        # Scale: F1 is a rough proxy — cap at 0.9 to reflect uncertainty
        return min(0.9, f1 * 0.85 + 0.1)

    def _detect_misconception(
        self, answer: str, concept_id: int
    ) -> tuple[str, float, float]:
        """
        Detect misconception class, severity, and confidence.

        Returns (label, severity, confidence).
        """
        best_label = "none"
        best_count = 0
        best_confidence = 0.5

        for mc_label, patterns in MISCONCEPTION_PATTERNS.items():
            count = _count_matches(answer, patterns)
            if count > best_count:
                best_count = count
                best_label = mc_label
                best_confidence = min(0.85, 0.50 + count * 0.12)

        # Severity estimation
        if best_label == "none":
            severity = 0.0
            best_confidence = 0.7
        else:
            base_severity = 0.40

            if _matches_any(answer, SEVERITY_KEYWORDS_HIGH):
                base_severity += 0.25
            if _matches_any(answer, SEVERITY_KEYWORDS_LOW):
                base_severity -= 0.15

            severity = max(0.10, min(0.95, base_severity))

        return best_label, severity, best_confidence

    def _estimate_reasoning_quality(self, answer: str) -> float:
        """
        Heuristic estimate of reasoning quality from answer text.

        Longer, vocabulary-rich answers tend to reflect better reasoning.
        This is a crude proxy — not a validated measurement.
        """
        words = answer.split()
        if len(words) < 3:
            return 0.15
        if len(words) < 8:
            return 0.35
        # Check for domain vocabulary
        domain_words = {
            "variable", "function", "loop", "recursion", "scope", "return",
            "parameter", "reference", "value", "copy", "stack", "list",
            "algorithm", "complexity", "base", "case", "condition", "iteration",
            "assignment", "comparison", "operator", "syntax", "error"
        }
        domain_count = sum(1 for w in words if w in domain_words)
        base = min(0.75, 0.3 + len(words) * 0.01)
        domain_boost = min(0.20, domain_count * 0.05)
        return round(base + domain_boost, 3)
