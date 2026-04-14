from .canary import CanaryToken
from .detector import PromptInjectionDetector
from .pii import PIIScanner
from .regex_scanner import RegexScanner
from .sanitizer import PromptSanitizer
from .toxicity import ToxicityDetector

__version__ = "0.3.2"

__all__ = [
    "PromptInjectionDetector",
    "PromptSanitizer",
    "PIIScanner",
    "ToxicityDetector",
    "RegexScanner",
    "CanaryToken",
]


def __getattr__(name):
    if name in {"PytectorGuard", "PromptInjectionBlockedError"}:
        try:
            from .langchain import PromptInjectionBlockedError, PytectorGuard
        except ImportError as exc:
            raise ImportError(
                "LangChain support is optional. Install it with: pip install pytector[langchain]"
            ) from exc

        globals()["PytectorGuard"] = PytectorGuard
        globals()["PromptInjectionBlockedError"] = PromptInjectionBlockedError
        return globals()[name]
    raise AttributeError(f"module 'pytector' has no attribute {name!r}")
