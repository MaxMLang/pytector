"""Input sanitization for prompt injection defense."""

from __future__ import annotations

import base64
import codecs
import re
import unicodedata
from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# Default keyword list (mirrors PromptInjectionDetector.default_input_keywords)
# ---------------------------------------------------------------------------

DEFAULT_KEYWORDS = [
    "ignore previous",
    "ignore above",
    "ignore all",
    "ignore everything",
    "forget everything",
    "forget all",
    "forget previous",
    "forget above",
    "disregard everything",
    "disregard all",
    "disregard previous",
    "disregard above",
    "bypass all",
    "bypass everything",
    "bypass previous",
    "bypass above",
    "system prompt",
    "system message",
    "system instruction",
    "system directive",
    "override system",
    "override prompt",
    "override instruction",
    "override directive",
    "ignore system",
    "ignore prompt",
    "ignore instruction",
    "ignore directive",
    "forget system",
    "forget prompt",
    "forget instruction",
    "forget directive",
    "disregard system",
    "disregard prompt",
    "disregard instruction",
    "disregard directive",
    "bypass system",
    "bypass prompt",
    "bypass instruction",
    "bypass directive",
    "ignore",
    "forget",
    "disregard",
    "override",
    "bypass",
    "hack",
    "exploit",
    "system",
    "prompt",
    "injection",
    "jailbreak",
    "break",
    "escape",
]

# ---------------------------------------------------------------------------
# Compiled regex constants
# ---------------------------------------------------------------------------

_BASE64_RE = re.compile(
    r"(?<![A-Za-z0-9+/=])([A-Za-z0-9+/]{20,}={0,2})(?![A-Za-z0-9+/=])"
)

_HEX_ESCAPE_RE = re.compile(r"((?:\\x[0-9a-fA-F]{2}){4,})")

_ROT13_WRAPPER_RE = re.compile(r"rot13\(([^)]+)\)", re.IGNORECASE)

_INJECTION_INDICATORS = re.compile(
    r"|".join(
        [
            r"(?:ignore|forget|disregard|override|bypass)"
            r"\s+(?:all\s+)?(?:previous|prior|above|the|your|my)?"
            r"\s*(?:instructions?|prompts?|rules?|directives?)",
            r"\b(?:system\s+prompt|jailbreak|prompt\s+injection)\b",
            r"\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be)\b",
            r"\b(?:reveal|show|output)\s+(?:the\s+|your\s+)?"
            r"(?:system|hidden|secret)",
        ]
    ),
    re.IGNORECASE,
)

_INVISIBLE_CHARS_RE = re.compile(
    "["
    "\u00ad"
    "\u200b-\u200d"
    "\u2060"
    "\ufeff"
    "\u202a-\u202e"
    "\u2066-\u2069"
    "\U000e0001-\U000e007f"
    "]"
)

_INJECTION_PATTERNS = [
    re.compile(
        r"(?:please\s+)?"
        r"(?:ignore|forget|disregard|override|bypass|skip|drop|abandon|stop\s+following)"
        r"\s+(?:all\s+|any\s+)?(?:the\s+|my\s+|your\s+)?"
        r"(?:previous|prior|above|earlier|preceding|original|initial|"
        r"old|existing|current|given|following)?\s*"
        r"(?:instructions?|prompts?|directives?|rules?|guidelines?|"
        r"constraints?|commands?|context|messages?|text|input)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:you\s+are\s+now|from\s+now\s+on(?:\s+you\s+are)?"
        r"|act\s+as(?:\s+(?:a|an|the))?"
        r"|pretend\s+(?:to\s+be|you(?:\s+are|'re))"
        r"|imagine\s+you(?:\s+are|'re)"
        r"|roleplay\s+as|behave\s+(?:as|like)"
        r"|simulate\s+being"
        r"|your\s+new\s+(?:role|identity|persona)\s+is)"
        r"\s+[^.!?\n]*[.!?]?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:reveal|show|display|output|print|repeat|echo|tell\s+me|give\s+me"
        r"|what\s+(?:is|are)|share|leak|expose|read\s+(?:back|out))"
        r"\s+(?:the\s+|your\s+)?(?:(?:full|complete|exact|entire|original|"
        r"hidden|secret)\s+)?"
        r"(?:system\s+(?:prompt|message|instructions?)"
        r"|initial\s+(?:prompt|instructions?)"
        r"|hidden\s+(?:prompt|instructions?|message)"
        r"|original\s+(?:prompt|instructions?)"
        r"|(?:training|internal)\s+(?:data|instructions?|rules?))",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*-{3,}\s*$", re.MULTILINE),
    re.compile(r"^\s*={3,}\s*$", re.MULTILINE),
    re.compile(r"^\s*#{3,}\s*$", re.MULTILINE),
    re.compile(r"^\s*`{3,}\s*$", re.MULTILINE),
    re.compile(
        r"(?:new\s+instructions?\s*:"
        r"|updated\s+(?:prompt|instructions?)\s*:"
        r"|begin(?:ning)?\s+(?:of\s+)?new\s+conversation"
        r"|end\s+of\s+(?:system|original)\s+(?:prompt|message|instructions?)"
        r"|<\s*/?system\s*>"
        r"|<\s*/?(?:user|assistant|human|ai)\s*>)",
        re.IGNORECASE,
    ),
]

_SENTENCE_SIGNALS = [
    # Imperative mood controlling AI behaviour
    (
        0.25,
        [
            re.compile(
                r"\b(?:do\s+not|don'?t)\s+"
                r"(?:follow|obey|listen\s+to|adhere\s+to|comply\s+with)",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:output|print|write|generate|respond\s+with|reply\s+with|say)\s+"
                r"(?:only|exactly|just|nothing\s+but)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:you\s+(?:must|should|need\s+to|have\s+to)\s+"
                r"(?:now|always|only|instead))\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:execute|run|perform|carry\s+out)\s+"
                r"(?:the\s+following|this|these)\b",
                re.IGNORECASE,
            ),
        ],
    ),
    # References to system internals
    (
        0.3,
        [
            re.compile(
                r"\b(?:system\s+(?:prompt|message|instructions?)"
                r"|training\s+data"
                r"|internal\s+(?:instructions?|rules?))\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\byour\s+(?:original|initial|hidden|secret|real|true)\s+"
                r"(?:instructions?|prompt|rules?|purpose|goal)\b",
                re.IGNORECASE,
            ),
        ],
    ),
    # Role / identity manipulation
    (
        0.3,
        [
            re.compile(
                r"\b(?:you\s+are\s+(?:now\s+)?(?:a|an|no\s+longer)"
                r"|your\s+(?:new\s+)?(?:role|identity|persona|character))\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:act\s+as|pretend|roleplay|simulate|impersonate|become)\b",
                re.IGNORECASE,
            ),
            re.compile(r"\bfrom\s+now\s+on\b", re.IGNORECASE),
        ],
    ),
    # Negation of constraints
    (
        0.25,
        [
            re.compile(
                r"\b(?:don'?t\s+worry\s+about"
                r"|no\s+(?:restrictions?|rules?|limits?|boundaries|constraints?)"
                r"|without\s+(?:any\s+)?(?:limits?|restrictions?|constraints?"
                r"|rules?|boundaries))\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:unrestricted|unlimited|unfiltered|uncensored|unmoderated)\b",
                re.IGNORECASE,
            ),
        ],
    ),
    # Urgency / authority markers
    (
        0.2,
        [
            re.compile(
                r"\b(?:immediately|urgently|right\s+now"
                r"|this\s+is\s+(?:very\s+)?important"
                r"|admin(?:istrator)?\s+(?:override|access|mode)"
                r"|priority\s+(?:one|override))\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:emergency"
                r"|critical\s+(?:override|instruction)"
                r"|you\s+must\s+(?:comply|obey|do\s+this))\b",
                re.IGNORECASE,
            ),
        ],
    ),
]

_DEFAULT_ENFORCEMENT_MAP = {
    "{": "\\{",
    "}": "\\}",
    "<": "\\<",
    ">": "\\>",
    "`": "\\`",
}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class PromptSanitizer:
    """Sanitizes text input by removing or neutralising prompt injection attempts.

    Runs a layered pipeline of strategies: encoding detection, unicode
    normalisation, regex pattern removal, sentence-level scoring, fuzzy
    matching, and keyword stripping.  An optional seventh strategy (prompt
    enforcement) escapes template syntax.
    """

    def __init__(
        self,
        enable_encoding_detection=True,
        enable_unicode_normalization=True,
        enable_pattern_removal=True,
        enable_sentence_scoring=True,
        enable_fuzzy_matching=True,
        enable_keyword_stripping=True,
        enable_prompt_enforcement=False,
        keywords=None,
        case_sensitive=False,
        replacement="",
        fuzzy_threshold=0.85,
        sentence_threshold=0.5,
        enforcement_chars=None,
    ):
        self.enable_encoding_detection = enable_encoding_detection
        self.enable_unicode_normalization = enable_unicode_normalization
        self.enable_pattern_removal = enable_pattern_removal
        self.enable_sentence_scoring = enable_sentence_scoring
        self.enable_fuzzy_matching = enable_fuzzy_matching
        self.enable_keyword_stripping = enable_keyword_stripping
        self.enable_prompt_enforcement = enable_prompt_enforcement

        self.keywords = (
            list(keywords) if keywords is not None else DEFAULT_KEYWORDS.copy()
        )
        self.case_sensitive = case_sensitive
        self.replacement = replacement
        self.fuzzy_threshold = fuzzy_threshold
        self.sentence_threshold = sentence_threshold

        self._enforcement_map = (
            dict(enforcement_chars)
            if enforcement_chars is not None
            else _DEFAULT_ENFORCEMENT_MAP.copy()
        )

    # -- Main entry points --------------------------------------------------

    def sanitize(self, text, return_details=False):
        """Run the sanitisation pipeline on *text*.

        Returns ``(cleaned_text, was_modified)`` by default.  When
        *return_details* is ``True``, returns
        ``(cleaned_text, was_modified, changes)`` where *changes* is a list of
        dicts describing each modification.
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected string input, got {type(text)!r}.")

        changes: list[dict[str, str]] = []

        if self.enable_encoding_detection:
            text = self._apply_encoding_detection(text, changes)
        if self.enable_unicode_normalization:
            text = self._apply_unicode_normalization(text, changes)
        if self.enable_pattern_removal:
            text = self._apply_pattern_removal(text, changes)
        if self.enable_sentence_scoring:
            text = self._apply_sentence_scoring(text, changes)
        if self.enable_fuzzy_matching:
            text = self._apply_fuzzy_matching(text, changes)
        if self.enable_keyword_stripping:
            text = self._apply_keyword_stripping(text, changes)
        if self.enable_prompt_enforcement:
            text = self._apply_prompt_enforcement(text, changes)

        text = _cleanup_whitespace(text)
        was_modified = len(changes) > 0

        if return_details:
            return text, was_modified, changes
        return text, was_modified

    def report_sanitization(self, text):
        """Print a human-readable sanitisation report (mirrors
        ``PromptInjectionDetector.report_injection_status``).
        """
        cleaned, was_modified, changes = self.sanitize(text, return_details=True)
        if not was_modified:
            print("No sanitization needed. Input is clean.")
            return
        print(f"Sanitized input ({len(changes)} modification(s)).")
        for change in changes:
            print(f"  [{change['strategy']}] Removed: {change['removed']}")
        print(f"Cleaned text: {cleaned}")

    # -- Keyword management (mirrors detector API) --------------------------

    def add_keywords(self, keywords):
        self.keywords.extend(_normalize_keyword_input(keywords))

    def remove_keywords(self, keywords):
        for kw in _normalize_keyword_input(keywords):
            if kw in self.keywords:
                self.keywords.remove(kw)

    def get_keywords(self):
        return self.keywords.copy()

    # -- Strategy implementations -------------------------------------------

    def _apply_encoding_detection(self, text, changes):
        # Base64 ---------------------------------------------------------------
        def _check_b64(match):
            candidate = match.group(1)
            try:
                decoded = base64.b64decode(candidate).decode("utf-8", errors="ignore")
            except Exception:
                return match.group(0)
            if (
                decoded.isprintable()
                and len(decoded) >= 4
                and _INJECTION_INDICATORS.search(decoded)
            ):
                changes.append({"strategy": "encoding", "removed": candidate})
                return self.replacement
            return match.group(0)

        text = _BASE64_RE.sub(_check_b64, text)

        # Hex escape sequences -------------------------------------------------
        def _check_hex(match):
            hex_str = match.group(1)
            try:
                decoded = codecs.decode(hex_str, "unicode_escape")
            except Exception:
                return match.group(0)
            if _INJECTION_INDICATORS.search(decoded):
                changes.append({"strategy": "encoding", "removed": hex_str})
                return self.replacement
            return match.group(0)

        text = _HEX_ESCAPE_RE.sub(_check_hex, text)

        # ROT-13 wrappers ------------------------------------------------------
        def _check_rot13(match):
            content = match.group(1)
            try:
                decoded = codecs.decode(content, "rot_13")
            except Exception:
                return match.group(0)
            if _INJECTION_INDICATORS.search(decoded):
                changes.append({"strategy": "encoding", "removed": match.group(0)})
                return self.replacement
            return match.group(0)

        text = _ROT13_WRAPPER_RE.sub(_check_rot13, text)
        return text

    def _apply_unicode_normalization(self, text, changes):
        invisible = _INVISIBLE_CHARS_RE.findall(text)
        if invisible:
            text = _INVISIBLE_CHARS_RE.sub("", text)
            changes.append(
                {
                    "strategy": "unicode",
                    "removed": f"{len(invisible)} invisible character(s)",
                }
            )

        normalized = unicodedata.normalize("NFKC", text)
        if normalized != text:
            changes.append(
                {"strategy": "unicode", "removed": "homoglyphs normalized via NFKC"}
            )
            text = normalized
        return text

    def _apply_pattern_removal(self, text, changes):
        for pattern in _INJECTION_PATTERNS:

            def _record(match, _changes=changes):
                _changes.append(
                    {"strategy": "pattern", "removed": match.group().strip()}
                )
                return self.replacement

            text = pattern.sub(_record, text)
        return text

    def _apply_sentence_scoring(self, text, changes):
        sentences = _split_sentences(text)
        if not sentences:
            return text

        kept = []
        for sentence in sentences:
            score = _score_sentence(sentence)
            if score >= self.sentence_threshold:
                changes.append(
                    {"strategy": "sentence", "removed": sentence.strip()}
                )
            else:
                kept.append(sentence)

        if not kept:
            return ""
        return " ".join(s.strip() for s in kept if s.strip())

    def _apply_fuzzy_matching(self, text, changes):
        words = text.split()
        if not words:
            return text

        phrases_by_len: dict[int, list[str]] = {}
        for phrase in self.keywords:
            p = phrase if self.case_sensitive else phrase.lower()
            n = len(p.split())
            phrases_by_len.setdefault(n, []).append(p)

        remove_indices: set[int] = set()

        for n in sorted(phrases_by_len, reverse=True):
            phrases = phrases_by_len[n]
            if n > len(words):
                continue
            for i in range(len(words) - n + 1):
                if any(j in remove_indices for j in range(i, i + n)):
                    continue
                window_words = words[i : i + n]
                window = " ".join(
                    w.lower() if not self.case_sensitive else w for w in window_words
                )
                window_clean = re.sub(r"[^\w\s]", "", window)
                for phrase in phrases:
                    phrase_clean = re.sub(r"[^\w\s]", "", phrase)
                    if not phrase_clean or not window_clean:
                        continue
                    ratio = SequenceMatcher(None, window_clean, phrase_clean).ratio()
                    # Only fuzzy (non-exact) matches; exact matches are left for
                    # keyword stripping so the change log stays informative.
                    if self.fuzzy_threshold <= ratio < 1.0:
                        for j in range(i, i + n):
                            remove_indices.add(j)
                        changes.append(
                            {
                                "strategy": "fuzzy",
                                "removed": " ".join(window_words),
                            }
                        )
                        break

        if remove_indices:
            kept = [w for i, w in enumerate(words) if i not in remove_indices]
            text = " ".join(kept)
        return text

    def _apply_keyword_stripping(self, text, changes):
        sorted_kws = sorted(self.keywords, key=len, reverse=True)
        for kw in sorted_kws:
            flags = 0 if self.case_sensitive else re.IGNORECASE
            pattern = re.compile(r"\b" + re.escape(kw) + r"\b", flags)

            def _record(match, _changes=changes):
                _changes.append({"strategy": "keyword", "removed": match.group()})
                return self.replacement

            text = pattern.sub(_record, text)
        return text

    def _apply_prompt_enforcement(self, text, changes):
        original = text
        for char, escaped in self._enforcement_map.items():
            text = text.replace(char, escaped)
        if text != original:
            changes.append(
                {"strategy": "enforcement", "removed": "template syntax escaped"}
            )
        return text


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _normalize_keyword_input(keywords):
    if isinstance(keywords, str):
        return [keywords]
    return list(keywords)


def _split_sentences(text):
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p for p in parts if p.strip()]


def _score_sentence(sentence):
    score = 0.0
    for weight, patterns in _SENTENCE_SIGNALS:
        for pattern in patterns:
            if pattern.search(sentence):
                score += weight
                break
    return score


def _cleanup_whitespace(text):
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
