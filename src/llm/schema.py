"""
LLM extraction output schema with Pydantic v2 validation.

The LLM extraction component produces structured JSON conforming to this schema.
Schema validation is enforced at the provider level — any provider (real or mock)
must return an ExtractionResult that passes Pydantic validation.

Design principle: The LLM produces evidence (ExtractionResult). The ML model
integrates this evidence over time. The LLM does NOT directly update learner state.

All fields use strict type annotations to catch provider output errors early.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


# Valid misconception labels (must match misconception_taxonomy.md)
VALID_MISCONCEPTION_LABELS = [
    "none",
    "syntax_misunderstanding",
    "variable_scope_misunderstanding",
    "loop_termination_misunderstanding",
    "function_parameter_misunderstanding",
    "recursion_base_case_misunderstanding",
    "reference_vs_value_misunderstanding",
]

# Valid concept names (must match concept_graph.json)
VALID_CONCEPT_NAMES = [
    "Variables", "Conditions", "Loops", "Functions",
    "Recursion", "Data Structures", "Algorithms"
]


class ExtractionResult(BaseModel):
    """
    Structured output from the LLM semantic extraction component.

    All float fields are in [0.0, 1.0]. Validation is enforced by Pydantic.
    If any field fails validation, the provider must retry or return a fallback.

    Fields:
        correctness: Estimated correctness of the student answer (0=wrong, 1=correct).
        concepts_detected: List of concept names detected in the answer.
        misconception_label: Detected misconception from the taxonomy, or 'none'.
        misconception_severity: Severity of detected misconception (0=none, 1=severe).
        reasoning_quality: Quality of the student's reasoning process (0=poor, 1=excellent).
        confidence: Provider's confidence in this extraction (0=low, 1=high).
        reasoning_trace: Optional free-text explanation from the provider (not used in ML).
    """

    correctness: float = Field(ge=0.0, le=1.0,
                               description="Estimated answer correctness in [0,1]")
    concepts_detected: list[str] = Field(default_factory=list,
                                         description="Concept names detected in the answer")
    misconception_label: str = Field(default="none",
                                     description="Misconception class from taxonomy")
    misconception_severity: float = Field(ge=0.0, le=1.0, default=0.0,
                                          description="Severity of detected misconception")
    reasoning_quality: float = Field(ge=0.0, le=1.0, default=0.5,
                                     description="Quality of student reasoning")
    confidence: float = Field(ge=0.0, le=1.0, default=0.75,
                              description="Provider confidence in this extraction")
    reasoning_trace: Optional[str] = Field(default=None,
                                           description="Free-text explanation (not used in ML)")

    @field_validator("misconception_label")
    @classmethod
    def validate_misconception_label(cls, v: str) -> str:
        if v not in VALID_MISCONCEPTION_LABELS:
            raise ValueError(
                f"misconception_label '{v}' not in taxonomy. "
                f"Valid: {VALID_MISCONCEPTION_LABELS}"
            )
        return v

    @field_validator("concepts_detected")
    @classmethod
    def validate_concepts(cls, v: list[str]) -> list[str]:
        # Filter to only valid concept names; don't raise (LLM may hallucinate)
        return [c for c in v if c in VALID_CONCEPT_NAMES]

    @field_validator("misconception_severity")
    @classmethod
    def zero_severity_for_none(cls, v: float, info) -> float:
        # If misconception_label is 'none', severity should be 0
        # We check during model validation, not here (field order matters)
        return v

    def to_feature_dict(self) -> dict:
        """
        Convert to feature dict for injection into the feature builder.
        Only the numerical/categorical fields that the ML model consumes.
        """
        return {
            "misconception_severity": self.misconception_severity,
            "misconception_confidence": self.confidence,
            "reasoning_quality": self.reasoning_quality,
            "extracted_correctness": self.correctness,
            "misconception_label": self.misconception_label,
        }

    def to_misconception_class_id(self) -> int:
        """Convert misconception_label to integer class ID."""
        try:
            return VALID_MISCONCEPTION_LABELS.index(self.misconception_label)
        except ValueError:
            return 0  # default: 'none'


class ExtractionRequest(BaseModel):
    """
    Input to the LLM extraction component.

    All fields should be provided for best extraction quality.
    The model uses question context and rubric to ground the misconception detection.
    """
    question: str = Field(description="The assessment question text")
    student_answer: str = Field(description="The student's answer text")
    concept_name: str = Field(description="Primary concept being assessed")
    expected_answer: Optional[str] = Field(default=None, description="Rubric/expected answer")
    concept_id: int = Field(ge=0, le=6, default=0, description="Concept ID (0-6)")
