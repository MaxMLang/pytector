"""Screen tool and web content for indirect prompt injection.

:class:`ToolOutputGuard` checks untrusted text (tool results, browsed pages,
files) before it is passed back to an LLM and returns a :class:`GuardDecision`.
Framework-agnostic — no LangChain dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .detector import PromptInjectionDetector
from .sanitizer import PromptSanitizer

ACTION_ALLOW = "allow"
ACTION_REDACT = "redact"
ACTION_BLOCK = "block"


class PromptInjectionBlockedError(ValueError):
    """Raised when content is blocked due to detected prompt injection."""


@dataclass
class GuardDecision:
    """Outcome of screening tool/web content.

    Attributes
    ----------
    action :
        ``"allow"``, ``"redact"``, or ``"block"``.
    is_injection :
        Whether the content was flagged as prompt injection.
    score :
        Detector confidence, or ``None`` when the backend does not expose one.
    tool_name :
        Source tool name, if provided.
    original_content :
        Normalized text that was screened.
    content :
        Text safe to pass downstream (``None`` when blocked).
    reasons :
        Human-readable explanations.
    metadata :
        Extra context (threshold, backend, etc.).
    """

    action: str
    is_injection: bool
    original_content: str
    content: Optional[str] = None
    score: Optional[float] = None
    tool_name: Optional[str] = None
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return self.action == ACTION_BLOCK

    @property
    def allowed(self) -> bool:
        return self.action in (ACTION_ALLOW, ACTION_REDACT)


class ToolOutputGuard:
    """Screen tool/web content for prompt injection before it reaches the LLM.

    Example::

        >>> guard = ToolOutputGuard(threshold=0.7)
        >>> decision = guard.scan_tool_result("browse", page_text)
        >>> if decision.action == "block":
        ...     raise SystemExit(decision.reasons)
        >>> context = decision.content
    """

    def __init__(
        self,
        detector: Optional[PromptInjectionDetector] = None,
        *,
        model_name_or_url: str = "deberta",
        threshold: float = 0.5,
        use_groq: bool = False,
        api_key: Optional[str] = None,
        groq_model: str = "openai/gpt-oss-safeguard-20b",
        detector_kwargs: Optional[Dict[str, Any]] = None,
        sanitizer: Optional[PromptSanitizer] = None,
        on_detection: str = ACTION_BLOCK,
        raise_on_block: bool = False,
        block_on_api_error: bool = True,
    ) -> None:
        if on_detection not in (ACTION_BLOCK, ACTION_REDACT):
            raise ValueError(
                "on_detection must be 'block' or 'redact', "
                f"got {on_detection!r}."
            )

        self.threshold = threshold
        self.on_detection = on_detection
        self.raise_on_block = raise_on_block
        self.block_on_api_error = block_on_api_error

        self._detector = detector
        self._detector_config = {
            "model_name_or_url": model_name_or_url,
            "use_groq": use_groq,
            "api_key": api_key,
            "groq_model": groq_model,
        }
        self._detector_kwargs = dict(detector_kwargs or {})

        self._sanitizer = sanitizer
        if self.on_detection == ACTION_REDACT and self._sanitizer is None:
            self._sanitizer = PromptSanitizer()

    def _get_detector(self) -> PromptInjectionDetector:
        if self._detector is None:
            init_kwargs = dict(self._detector_config)
            init_kwargs.update(self._detector_kwargs)
            self._detector = PromptInjectionDetector(**init_kwargs)
        return self._detector

    def _get_sanitizer(self) -> PromptSanitizer:
        if self._sanitizer is None:
            self._sanitizer = PromptSanitizer()
        return self._sanitizer

    @staticmethod
    def _normalize_payload(payload: Any) -> str:
        """Convert a tool payload into a screenable string."""
        if isinstance(payload, str):
            return payload
        if isinstance(payload, (bytes, bytearray)):
            return bytes(payload).decode("utf-8", errors="replace")
        if isinstance(payload, (dict, list)):
            return json.dumps(payload, ensure_ascii=False, default=str)
        return str(payload)

    def _run_detection(
        self, text: str
    ) -> tuple[bool, Optional[float], List[str], Dict[str, Any]]:
        """Return ``(is_injection, score, reasons, metadata)``.

        Normalizes local/GGUF vs Groq polarity into a single boolean.
        """
        detector = self._get_detector()
        reasons: List[str] = []
        metadata: Dict[str, Any] = {}

        if detector.use_groq:
            metadata["backend"] = "groq"
            is_safe = detector.detect_injection_api(text)
            if is_safe is None:
                is_injection = self.block_on_api_error
                reasons.append(
                    "Prompt safety could not be determined due to a Groq API "
                    "error; treating as "
                    + ("unsafe." if is_injection else "safe.")
                )
                metadata["api_error"] = True
                return is_injection, None, reasons, metadata

            is_injection = is_safe is False
            if is_injection:
                reasons.append("Groq safeguard model flagged content as unsafe.")
            return is_injection, None, reasons, metadata

        metadata["backend"] = "gguf" if detector.is_gguf else "local"
        is_injection, score = detector.detect_injection(
            text, threshold=self.threshold
        )
        metadata["threshold"] = self.threshold
        if is_injection:
            score_text = "unknown" if score is None else f"{score:.4f}"
            reasons.append(
                f"Prompt injection detected (score={score_text}, "
                f"threshold={self.threshold:.4f})."
            )
        return is_injection, score, reasons, metadata

    def scan_text(
        self,
        text: str,
        *,
        tool_name: Optional[str] = None,
    ) -> GuardDecision:
        """Screen a string and return a :class:`GuardDecision`."""
        if not isinstance(text, str):
            raise TypeError(
                f"ToolOutputGuard.scan_text expects a string, got {type(text)!r}. "
                "Use scan_tool_result for non-string payloads."
            )

        is_injection, score, reasons, metadata = self._run_detection(text)

        if not is_injection:
            return GuardDecision(
                action=ACTION_ALLOW,
                is_injection=False,
                original_content=text,
                content=text,
                score=score,
                tool_name=tool_name,
                reasons=reasons,
                metadata=metadata,
            )

        if self.on_detection == ACTION_REDACT:
            cleaned, was_modified, changes = self._get_sanitizer().sanitize(
                text, return_details=True
            )
            metadata["sanitizer_modified"] = was_modified
            metadata["sanitizer_changes"] = changes
            reasons.append(
                "Content redacted before use."
                if was_modified
                else "Sanitizer made no changes; content forwarded as-is."
            )
            return GuardDecision(
                action=ACTION_REDACT,
                is_injection=True,
                original_content=text,
                content=cleaned,
                score=score,
                tool_name=tool_name,
                reasons=reasons,
                metadata=metadata,
            )

        decision = GuardDecision(
            action=ACTION_BLOCK,
            is_injection=True,
            original_content=text,
            content=None,
            score=score,
            tool_name=tool_name,
            reasons=reasons,
            metadata=metadata,
        )
        if self.raise_on_block:
            self._raise_blocked(decision)
        return decision

    def scan_tool_result(self, tool_name: str, payload: Any) -> GuardDecision:
        """Normalize *payload* to text and screen it."""
        return self.scan_text(
            self._normalize_payload(payload),
            tool_name=tool_name,
        )

    def interrupt_if_blocked(self, decision: GuardDecision) -> GuardDecision:
        """Raise if *decision* is a block; otherwise return it unchanged."""
        if decision.blocked:
            self._raise_blocked(decision)
        return decision

    @staticmethod
    def _raise_blocked(decision: GuardDecision) -> None:
        detail = "; ".join(decision.reasons) or "prompt injection detected"
        source = f" from tool {decision.tool_name!r}" if decision.tool_name else ""
        raise PromptInjectionBlockedError(
            f"Security alert: blocked untrusted content{source}. {detail}"
        )
