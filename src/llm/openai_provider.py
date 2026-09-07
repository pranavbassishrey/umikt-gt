"""
OpenAI LLM provider for the UMiKT-GAT pipeline.

This provider makes real API calls to the OpenAI API to perform semantic
extraction. It is gated behind the OPENAI_API_KEY environment variable.

Usage:
  export OPENAI_API_KEY="your-key-here"
  Then pass provider="openai" to the pipeline configuration.

If OPENAI_API_KEY is not set, this provider will raise at instantiation time.
The ML pipeline's correctness is not dependent on this provider — use the
MockLLMProvider (default) for all training and evaluation runs.

Cost note: Each extraction call uses approximately 200-500 input tokens and
50-150 output tokens (GPT-4o-mini pricing ~$0.00015/1K input tokens as of 2024).
A full demo session of 5 interactions costs approximately $0.0005. A full run
of the 50 handwritten Q&A pairs costs approximately $0.005.
"""

import json
import os
import time
from .base_provider import LLMProvider
from .schema import ExtractionRequest, ExtractionResult, VALID_MISCONCEPTION_LABELS


EXTRACTION_PROMPT_TEMPLATE = """You are a precise educational assessment analyzer.

Analyze the student's answer and return ONLY valid JSON (no markdown, no explanation outside JSON).

Question: {question}
Expected answer/rubric: {expected_answer}
Concept being assessed: {concept_name}
Student answer: {student_answer}

Misconception taxonomy (use EXACTLY one of these labels):
{misconception_labels}

Return a JSON object with exactly these fields:
{{
  "correctness": <float 0.0-1.0, how correct is the student answer>,
  "concepts_detected": <list of concept names mentioned in the answer>,
  "misconception_label": <one of the taxonomy labels above>,
  "misconception_severity": <float 0.0-1.0, 0 if label is 'none'>,
  "reasoning_quality": <float 0.0-1.0, quality of reasoning shown>,
  "confidence": <float 0.0-1.0, your confidence in this extraction>
}}"""


class OpenAILLMProvider(LLMProvider):
    """
    Real OpenAI API provider for semantic extraction.

    Requires OPENAI_API_KEY environment variable.
    Gated: not used in any training or evaluation run by default.
    Set the 'llm_provider' key in the config to 'openai' to enable.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_retries: int = 2,
    ):
        super().__init__(max_retries=max_retries)

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set. "
                "Use MockLLMProvider for offline evaluation, or "
                "set the API key to use real LLM extraction."
            )

        try:
            import requests
            self._requests = requests
        except ImportError:
            raise ImportError("requests library required for OpenAILLMProvider")

        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._base_url = "https://api.openai.com/v1/chat/completions"

    def _extract_raw(self, request: ExtractionRequest) -> dict:
        """Call OpenAI API and parse structured response."""
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            question=request.question,
            expected_answer=request.expected_answer or "Not provided",
            concept_name=request.concept_name,
            student_answer=request.student_answer,
            misconception_labels="\n".join(f"  - {m}" for m in VALID_MISCONCEPTION_LABELS),
        )

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system",
                 "content": "You are a precise JSON-only educational assessment analyzer. "
                            "Return ONLY valid JSON, no other text."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self._temperature,
            "max_tokens": 300,
        }

        t0 = time.perf_counter()
        resp = self._requests.post(
            self._base_url,
            headers=headers,
            json=payload,
            timeout=30,
        )
        latency = time.perf_counter() - t0

        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI API error {resp.status_code}: {resp.text[:200]}")

        resp_json = resp.json()
        content = resp_json["choices"][0]["message"]["content"].strip()

        # Track token usage
        usage = resp_json.get("usage", {})
        self._token_count += usage.get("total_tokens", 0)

        # Parse JSON — strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        parsed = json.loads(content)
        parsed["reasoning_trace"] = (
            f"[OpenAI:{self._model}] latency={latency:.2f}s "
            f"tokens={usage.get('total_tokens', 0)}"
        )
        return parsed
