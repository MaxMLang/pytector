from .canary import CanaryToken
from .detector import PromptInjectionDetector
from .guard import GuardDecision, PromptInjectionBlockedError, ToolOutputGuard
from .pii import PIIScanner
from .pipeline import SecurityPipeline
from .regex_scanner import RegexScanner
from .sanitizer import PromptSanitizer
from .toxicity import ToxicityDetector
from .web import WebPageScanner, chunk_text, extract_text_from_html

__version__ = "0.4.0"

__all__ = [
    "PromptInjectionDetector",
    "PromptSanitizer",
    "PIIScanner",
    "ToxicityDetector",
    "RegexScanner",
    "CanaryToken",
    "ToolOutputGuard",
    "GuardDecision",
    "PromptInjectionBlockedError",
    "WebPageScanner",
    "extract_text_from_html",
    "chunk_text",
    "SecurityPipeline",
]


def __getattr__(name):
    if name in {"PytectorGuard", "PytectorToolGuard"}:
        try:
            from .langchain import PytectorGuard, PytectorToolGuard
        except ImportError as exc:
            raise ImportError(
                "LangChain support is optional. Install it with: pip install pytector[langchain]"
            ) from exc

        globals()["PytectorGuard"] = PytectorGuard
        globals()["PytectorToolGuard"] = PytectorToolGuard
        return globals()[name]
    raise AttributeError(f"module 'pytector' has no attribute {name!r}")
