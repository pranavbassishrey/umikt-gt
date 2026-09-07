"""
Abstract base class for LLM extraction providers.

All providers (mock, OpenAI, etc.) must implement this interface.
This ensures the ML pipeline is decoupled from any specific LLM provider.
"""

from abc import ABC, abstractmethod
from .schema import ExtractionRequest, ExtractionResult


class LLMProvider(ABC):
    """
    Abstract LLM provider for semantic extraction.

    The provider takes a student answer + context and returns a validated
    ExtractionResult. The ML pipeline consumes ExtractionResult objects
    without knowing which provider produced them.

    Design note: Providers may retry on schema validation errors.
    The max_retries parameter controls this behavior.
    """

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries
        self._call_count = 0
        self._token_count = 0

    @abstractmethod
    def _extract_raw(self, request: ExtractionRequest) -> dict:
        """
        Perform the actual extraction and return a raw dict.
        Subclasses implement this method.

        Must return a dict with at minimum the required ExtractionResult fields.
        Pydantic validation is applied by the base class after this call.
        """
        ...

    def extract(self, request: ExtractionRequest) -> ExtractionResult:
        """
        Extract structured evidence from a student answer.

        Applies Pydantic schema validation. On validation failure, retries
        up to max_retries times. If all retries fail, returns a safe default.

        Args:
            request: ExtractionRequest with question, answer, and context.

        Returns:
            Validated ExtractionResult.
        """
        self._call_count += 1

        for attempt in range(self.max_retries + 1):
            try:
                raw = self._extract_raw(request)
                result = ExtractionResult(**raw)
                return result
            except Exception as e:
                if attempt == self.max_retries:
                    # Return a safe default on final failure
                    return self._safe_default(request, error=str(e))

        return self._safe_default(request)

    def _safe_default(
        self, request: ExtractionRequest, error: str = ""
    ) -> ExtractionResult:
        """
        Return a conservative default ExtractionResult when extraction fails.

        This ensures the pipeline can continue even if the provider fails.
        The confidence field is set to 0 to signal low reliability to the ML model.
        """
        return ExtractionResult(
            correctness=0.5,    # neutral
            concepts_detected=[request.concept_name] if request.concept_name else [],
            misconception_label="none",
            misconception_severity=0.0,
            reasoning_quality=0.5,
            confidence=0.0,     # signal: unreliable extraction
            reasoning_trace=f"EXTRACTION_FAILED: {error}" if error else "EXTRACTION_FAILED",
        )

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def token_count(self) -> int:
        return self._token_count

    def get_usage_stats(self) -> dict:
        return {
            "provider": self.__class__.__name__,
            "call_count": self._call_count,
            "token_count": self._token_count,
        }
