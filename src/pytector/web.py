"""Fetch and screen web pages for indirect prompt injection.

:class:`WebPageScanner` turns a URL or HTML into plain text, chunks long
documents, and screens them with :class:`~pytector.guard.ToolOutputGuard`.
Uses only the standard library (``urllib`` + ``html.parser``).
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .guard import (
    ACTION_ALLOW,
    ACTION_BLOCK,
    ACTION_REDACT,
    GuardDecision,
    PromptInjectionBlockedError,
    ToolOutputGuard,
)

_SKIP_TAGS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "iframe",
        "object",
        "embed",
        "head",
    }
)

_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "br",
        "li",
        "tr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "section",
        "article",
        "header",
        "footer",
        "aside",
        "blockquote",
        "pre",
        "ul",
        "ol",
        "table",
        "thead",
        "tbody",
        "tfoot",
        "hr",
    }
)

_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_NEWLINE_RE = re.compile(r"\n{3,}")

DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_BYTES = 1_000_000
DEFAULT_USER_AGENT = "pytector-WebPageScanner/0.4 (+https://github.com/MaxMLang/pytector)"


class _HTMLTextExtractor(HTMLParser):
    """Stdlib HTML → visible text extractor."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        del attrs
        lower = tag.lower()
        if lower in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if lower in _BLOCK_TAGS or lower == "br":
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lower = tag.lower()
        if lower in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if lower in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data:
            self._parts.append(data)

    def get_text(self) -> str:
        joined = "".join(self._parts)
        joined = _WHITESPACE_RE.sub(" ", joined)
        joined = _NEWLINE_RE.sub("\n\n", joined)
        joined = re.sub(r" *\n *", "\n", joined)
        return joined.strip()


def extract_text_from_html(html: str) -> str:
    """Strip scripts/styles and return visible text from HTML."""
    if not isinstance(html, str):
        raise TypeError(f"Expected HTML string, got {type(html)!r}.")
    parser = _HTMLTextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        stripped = re.sub(
            r"(?is)<(script|style|noscript|template).*?>.*?</\1>",
            " ",
            html,
        )
        stripped = re.sub(r"(?s)<[^>]+>", " ", stripped)
        return _WHITESPACE_RE.sub(" ", stripped).strip()
    return parser.get_text()


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """Split *text* into overlapping windows for long-document screening."""
    if not isinstance(text, str):
        raise TypeError(f"Expected string, got {type(text)!r}.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1.")
    if overlap < 0:
        raise ValueError("overlap must be >= 0.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    if not text:
        return [""]
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    step = chunk_size - overlap
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        if end >= length:
            break
        start += step
    return chunks


class WebPageScanner:
    """Fetch/extract web content and screen it via :class:`ToolOutputGuard`.

    Example::

        from pytector import WebPageScanner

        scanner = WebPageScanner(threshold=0.7)
        decision = scanner.scan_url("https://example.com/page")
        if decision.action == "block":
            raise SystemExit(decision.reasons)
        context = decision.content
    """

    def __init__(
        self,
        guard: Optional[ToolOutputGuard] = None,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_bytes: int = DEFAULT_MAX_BYTES,
        user_agent: str = DEFAULT_USER_AGENT,
        tool_name: str = "web",
        detector=None,
        model_name_or_url: str = "deberta",
        threshold: float = 0.5,
        use_groq: bool = False,
        api_key: Optional[str] = None,
        groq_model: str = "openai/gpt-oss-safeguard-20b",
        detector_kwargs: Optional[Dict[str, Any]] = None,
        sanitizer=None,
        on_detection: str = ACTION_BLOCK,
        raise_on_block: bool = False,
        block_on_api_error: bool = True,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        if timeout <= 0:
            raise ValueError("timeout must be > 0.")
        if max_bytes < 1:
            raise ValueError("max_bytes must be >= 1.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.user_agent = user_agent
        self.tool_name = tool_name

        if guard is None:
            guard = ToolOutputGuard(
                detector=detector,
                model_name_or_url=model_name_or_url,
                threshold=threshold,
                use_groq=use_groq,
                api_key=api_key,
                groq_model=groq_model,
                detector_kwargs=detector_kwargs,
                sanitizer=sanitizer,
                on_detection=on_detection,
                raise_on_block=raise_on_block,
                block_on_api_error=block_on_api_error,
            )
        self.guard = guard

    @staticmethod
    def _validate_url(url: str) -> str:
        if not isinstance(url, str) or not url.strip():
            raise ValueError("url must be a non-empty string.")
        parsed = urlparse(url.strip())
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"Only http/https URLs are supported, got scheme={parsed.scheme!r}."
            )
        if not parsed.netloc:
            raise ValueError(f"URL is missing a host: {url!r}.")
        return url.strip()

    def fetch_url(self, url: str) -> Tuple[str, Dict[str, Any]]:
        """Download *url* and return ``(body_text, fetch_metadata)``."""
        safe_url = self._validate_url(url)
        request = Request(
            safe_url,
            headers={"User-Agent": self.user_agent, "Accept": "text/html,*/*"},
            method="GET",
        )
        metadata: Dict[str, Any] = {"url": safe_url}
        try:
            with urlopen(request, timeout=self.timeout) as response:
                content_type = response.headers.get("Content-Type", "")
                metadata["content_type"] = content_type
                metadata["status"] = getattr(response, "status", None) or response.getcode()
                raw = response.read(self.max_bytes + 1)
        except URLError:
            raise
        except OSError as exc:
            raise URLError(f"Failed to fetch {safe_url}: {exc}") from exc

        truncated = len(raw) > self.max_bytes
        if truncated:
            raw = raw[: self.max_bytes]
        metadata["truncated"] = truncated
        metadata["bytes_read"] = len(raw)

        charset = "utf-8"
        content_type = metadata.get("content_type") or ""
        if "charset=" in content_type.lower():
            charset = (
                content_type.split("charset=", 1)[1]
                .split(";")[0]
                .strip()
                .strip("\"'")
            )
        body = raw.decode(charset or "utf-8", errors="replace")
        return body, metadata

    def scan_text(
        self,
        text: str,
        *,
        tool_name: Optional[str] = None,
        source_metadata: Optional[Dict[str, Any]] = None,
    ) -> GuardDecision:
        """Chunk *text* and screen each window via the underlying guard."""
        if not isinstance(text, str):
            raise TypeError(f"Expected string, got {type(text)!r}.")

        name = tool_name if tool_name is not None else self.tool_name
        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        return self._aggregate_chunk_decisions(
            text,
            chunks,
            tool_name=name,
            source_metadata=source_metadata or {},
        )

    def scan_html(
        self,
        html: str,
        *,
        tool_name: Optional[str] = None,
        source_metadata: Optional[Dict[str, Any]] = None,
    ) -> GuardDecision:
        """Extract visible text from *html*, then :meth:`scan_text`."""
        text = extract_text_from_html(html)
        meta = dict(source_metadata or {})
        meta["extracted_chars"] = len(text)
        meta["html_chars"] = len(html)
        return self.scan_text(text, tool_name=tool_name, source_metadata=meta)

    def scan_url(self, url: str, *, tool_name: Optional[str] = None) -> GuardDecision:
        """Fetch *url*, extract text if HTML-like, and screen the result."""
        body, fetch_meta = self.fetch_url(url)
        content_type = (fetch_meta.get("content_type") or "").lower()
        head = body.lstrip()[:64].lower()
        looks_like_html = (
            "html" in content_type
            or head.startswith("<!doctype html")
            or head.startswith("<html")
            or "<" in body[:500]
        )
        if looks_like_html:
            return self.scan_html(
                body, tool_name=tool_name, source_metadata=fetch_meta
            )
        fetch_meta["extracted_chars"] = len(body)
        return self.scan_text(
            body, tool_name=tool_name, source_metadata=fetch_meta
        )

    def _aggregate_chunk_decisions(
        self,
        original_text: str,
        chunks: Sequence[str],
        *,
        tool_name: str,
        source_metadata: Dict[str, Any],
    ) -> GuardDecision:
        previous_raise = self.guard.raise_on_block
        self.guard.raise_on_block = False
        try:
            chunk_decisions = [
                self.guard.scan_text(chunk, tool_name=tool_name) for chunk in chunks
            ]
        finally:
            self.guard.raise_on_block = previous_raise

        any_injection = any(d.is_injection for d in chunk_decisions)
        scores = [d.score for d in chunk_decisions if d.score is not None]
        max_score = max(scores) if scores else None

        reasons: List[str] = []
        flagged_indexes = [
            i for i, d in enumerate(chunk_decisions) if d.is_injection
        ]
        for i in flagged_indexes:
            reasons.extend(f"chunk[{i}]: {r}" for r in chunk_decisions[i].reasons)

        metadata: Dict[str, Any] = {
            "chunk_count": len(chunks),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "flagged_chunks": flagged_indexes,
            **source_metadata,
        }
        if chunk_decisions:
            for key in ("backend", "threshold", "api_error"):
                if key in chunk_decisions[0].metadata:
                    metadata[key] = chunk_decisions[0].metadata[key]

        if not any_injection:
            decision = GuardDecision(
                action=ACTION_ALLOW,
                is_injection=False,
                original_content=original_text,
                content=original_text,
                score=max_score,
                tool_name=tool_name,
                reasons=reasons,
                metadata=metadata,
            )
        elif self.guard.on_detection == ACTION_REDACT:
            cleaned_parts = [
                d.content if d.content is not None else d.original_content
                for d in chunk_decisions
            ]
            content = self._stitch_redacted_chunks(cleaned_parts)
            reasons.append(
                f"Redacted {len(flagged_indexes)} of {len(chunks)} chunk(s)."
            )
            metadata["sanitizer_modified"] = any(
                d.metadata.get("sanitizer_modified") for d in chunk_decisions
            )
            decision = GuardDecision(
                action=ACTION_REDACT,
                is_injection=True,
                original_content=original_text,
                content=content,
                score=max_score,
                tool_name=tool_name,
                reasons=reasons,
                metadata=metadata,
            )
        else:
            reasons.append(
                f"Blocked: {len(flagged_indexes)} of {len(chunks)} chunk(s) "
                "flagged as prompt injection."
            )
            decision = GuardDecision(
                action=ACTION_BLOCK,
                is_injection=True,
                original_content=original_text,
                content=None,
                score=max_score,
                tool_name=tool_name,
                reasons=reasons,
                metadata=metadata,
            )

        if decision.blocked and previous_raise:
            detail = "; ".join(decision.reasons) or "prompt injection detected"
            source = f" from tool {tool_name!r}" if tool_name else ""
            raise PromptInjectionBlockedError(
                f"Security alert: blocked untrusted content{source}. {detail}"
            )
        return decision

    def _stitch_redacted_chunks(self, cleaned_chunks: Sequence[str]) -> str:
        """Join overlapping cleaned chunks without duplicating the overlap."""
        if not cleaned_chunks:
            return ""
        if len(cleaned_chunks) == 1:
            return cleaned_chunks[0]

        parts = [cleaned_chunks[0]]
        for cleaned in cleaned_chunks[1:]:
            if self.chunk_overlap and len(cleaned) > self.chunk_overlap:
                parts.append(cleaned[self.chunk_overlap :])
            else:
                parts.append(cleaned)
        return "".join(parts)
