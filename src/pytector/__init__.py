from .detector import PromptInjectionDetector

__version__ = "0.2.2"

__all__ = ["PromptInjectionDetector"]


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
