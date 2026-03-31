"""Rule-based PII and credential detection using customisable regex patterns.

This module is **pure Python stdlib** — no model downloads, no heavy
dependencies.  It ships with sensible defaults for common PII types and
lets users add, remove, or completely replace patterns at construction
time or at runtime.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_PATTERNS: Dict[str, str] = {
    "EMAIL": r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    "PHONE": (
        r"(?:\+?1[\s\-.]?)?"
        r"(?:\(?\d{3}\)?[\s\-.]?)"
        r"\d{3}[\s\-.]?\d{4}"
    ),
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "CREDIT_CARD": (
        r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))"
        r"[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{1,4}\b"
    ),
    "IP_ADDRESS": (
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ),
    "API_KEY": (
        r"(?:sk|pk)[-_](?:live|test|prod|dev)[-_][A-Za-z0-9]{16,}"
    ),
    "JWT_TOKEN": (
        r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"
    ),
}


class RegexScanner:
    """Scan text for sensitive data using compiled regular expressions.

    Parameters
    ----------
    patterns : dict[str, str] | None
        Mapping of ``{PATTERN_NAME: regex_string}``.  Merged with the
        built-in defaults when *use_defaults* is ``True``, or used alone
        when ``False``.
    use_defaults : bool
        Whether to include the built-in patterns (EMAIL, PHONE, SSN,
        CREDIT_CARD, IP_ADDRESS, API_KEY, JWT_TOKEN).
    """

    def __init__(
        self,
        patterns: Optional[Dict[str, str]] = None,
        use_defaults: bool = True,
    ) -> None:
        raw: Dict[str, str] = {}
        if use_defaults:
            raw.update(DEFAULT_PATTERNS)
        if patterns:
            raw.update(patterns)

        self._patterns: Dict[str, str] = raw
        self._compiled: Dict[str, re.Pattern[str]] = {
            name: re.compile(pat) for name, pat in raw.items()
        }

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def scan(self, text: str) -> Tuple[bool, List[Dict[str, Any]]]:
        """Scan *text* against all active patterns.

        Returns ``(has_matches, matches)`` where each match dict contains
        ``pattern_name``, ``match``, ``start``, and ``end``.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        matches: List[Dict[str, Any]] = []
        for name, compiled in self._compiled.items():
            for m in compiled.finditer(text):
                matches.append({
                    "pattern_name": name,
                    "match": m.group(),
                    "start": m.start(),
                    "end": m.end(),
                })

        matches.sort(key=lambda m: m["start"])
        return len(matches) > 0, matches

    def redact(self, text: str, replacement: str = "[REDACTED]") -> str:
        """Return a copy of *text* with all matches replaced by *replacement*.

        Non-overlapping matches are replaced right-to-left so offsets stay valid.
        """
        _, matches = self.scan(text)
        merged = self._merge_overlapping(matches)
        result = text
        for m in reversed(merged):
            result = result[: m["start"]] + replacement + result[m["end"] :]
        return result

    def report(self, text: str) -> None:
        """Print a human-readable scan summary."""
        has_matches, matches = self.scan(text)
        if not has_matches:
            print("No matches found.")
            return
        print(f"Regex scan — {len(matches)} match{'es' if len(matches) != 1 else ''} found:")
        for m in matches:
            print(
                f"  [{m['pattern_name']}] \"{m['match']}\" "
                f"(pos={m['start']}:{m['end']})"
            )

    # ------------------------------------------------------------------
    # Pattern management
    # ------------------------------------------------------------------

    def add_pattern(self, name: str, pattern: str) -> None:
        """Add or overwrite a pattern at runtime."""
        self._patterns[name] = pattern
        self._compiled[name] = re.compile(pattern)

    def remove_pattern(self, name: str) -> None:
        """Remove a pattern by name.  No-op if not present."""
        self._patterns.pop(name, None)
        self._compiled.pop(name, None)

    def get_patterns(self) -> Dict[str, str]:
        """Return a copy of the active pattern dictionary."""
        return self._patterns.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_overlapping(
        matches: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge overlapping spans so redaction doesn't double-replace."""
        if not matches:
            return []
        sorted_matches = sorted(matches, key=lambda m: m["start"])
        merged: List[Dict[str, Any]] = [sorted_matches[0].copy()]
        for m in sorted_matches[1:]:
            prev = merged[-1]
            if m["start"] <= prev["end"]:
                if m["end"] > prev["end"]:
                    prev["end"] = m["end"]
                    prev["match"] = prev["match"] + m["match"][prev["end"] - m["start"]:]
            else:
                merged.append(m.copy())
        return merged
