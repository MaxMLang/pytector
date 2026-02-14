from __future__ import annotations

from typing import Any, Optional

try:
    from langchain_core.runnables import RunnableConfig, RunnableSerializable
except ImportError as exc:
    raise ImportError(
        "LangChain support is optional. Install it with: pip install pytector[langchain]"
    ) from exc

from pydantic import Field, PrivateAttr

from .detector import PromptInjectionDetector


class PromptInjectionBlockedError(ValueError):
    """Raised when a prompt is blocked by the guard."""


class PytectorGuard(RunnableSerializable[str, str]):
    """
    LangChain Runnable that blocks unsafe prompts before downstream steps run.

    For safe inputs the original string is passed through unchanged.
    """

    model_name_or_url: str = "deberta"
    threshold: float = 0.5
    use_groq: bool = False
    api_key: Optional[str] = None
    groq_model: str = "openai/gpt-oss-safeguard-20b"
    fallback_message: Optional[str] = None
    block_on_api_error: bool = True
    detector_kwargs: dict[str, Any] = Field(default_factory=dict)

    _detector: Optional[PromptInjectionDetector] = PrivateAttr(default=None)

    def _get_detector(self) -> PromptInjectionDetector:
        if self._detector is None:
            init_kwargs: dict[str, Any] = {
                "model_name_or_url": self.model_name_or_url,
                "use_groq": self.use_groq,
                "api_key": self.api_key,
                "groq_model": self.groq_model,
            }
            init_kwargs.update(self.detector_kwargs)
            self._detector = PromptInjectionDetector(**init_kwargs)
        return self._detector

    def _block(self, message: str) -> str:
        if self.fallback_message is not None:
            return self.fallback_message
        raise PromptInjectionBlockedError(message)

    def invoke(
        self,
        input: str,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> str:
        del config, kwargs

        if not isinstance(input, str):
            raise TypeError(f"PytectorGuard expects string input, got {type(input)!r}.")

        detector = self._get_detector()

        if detector.use_groq:
            is_safe = detector.detect_injection_api(input)
            if is_safe is None:
                if self.block_on_api_error:
                    return self._block(
                        "Security alert: prompt safety could not be determined due to an API error."
                    )
                return input
            if is_safe is False:
                return self._block("Security alert: unsafe prompt blocked by safeguard model.")
            return input

        is_injection, score = detector.detect_injection(input, threshold=self.threshold)
        if is_injection:
            score_text = "unknown" if score is None else f"{score:.4f}"
            return self._block(
                "Security alert: prompt injection detected "
                f"(score={score_text}, threshold={self.threshold:.4f})."
            )

        return input
