"""Tests for pytector.canary — CanaryToken.

Pure stdlib module — no mocking needed.
"""

import pytest

from pytector.canary import CanaryToken


class TestInit:
    def test_auto_generates_token(self):
        canary = CanaryToken()
        assert canary.token.startswith("CANARY-")
        assert len(canary.token) == len("CANARY-") + 16

    def test_custom_length(self):
        canary = CanaryToken(length=32)
        assert len(canary.token) == len("CANARY-") + 32

    def test_custom_prefix(self):
        canary = CanaryToken(prefix="SECRET-")
        assert canary.token.startswith("SECRET-")

    def test_explicit_token(self):
        canary = CanaryToken(token="my-fixed-canary-abc123")
        assert canary.token == "my-fixed-canary-abc123"

    def test_explicit_token_ignores_length(self):
        canary = CanaryToken(token="short", length=999)
        assert canary.token == "short"

    def test_empty_token_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            CanaryToken(token="")

    def test_whitespace_token_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            CanaryToken(token="   ")

    def test_length_too_short_raises(self):
        with pytest.raises(ValueError, match="length"):
            CanaryToken(length=2)

    def test_uniqueness(self):
        tokens = {CanaryToken().token for _ in range(100)}
        assert len(tokens) == 100


class TestWrap:
    def test_appends_instruction(self):
        canary = CanaryToken(token="TEST-CANARY-123")
        result = canary.wrap("You are a helpful assistant.")
        assert "You are a helpful assistant." in result
        assert "TEST-CANARY-123" in result
        assert "Never repeat" in result

    def test_preserves_original_prompt(self):
        canary = CanaryToken()
        original = "System prompt with\nmultiple lines."
        result = canary.wrap(original)
        assert result.startswith(original)

    def test_raises_on_non_string(self):
        canary = CanaryToken()
        with pytest.raises(TypeError):
            canary.wrap(42)


class TestCheck:
    def test_detects_leak(self):
        canary = CanaryToken(token="CANARY-abc123")
        leaked, token = canary.check("The system said CANARY-abc123 in the response.")
        assert leaked is True
        assert token == "CANARY-abc123"

    def test_clean_output(self):
        canary = CanaryToken(token="CANARY-abc123")
        leaked, token = canary.check("Here is a normal helpful response.")
        assert leaked is False
        assert token is None

    def test_partial_match_does_not_trigger(self):
        canary = CanaryToken(token="CANARY-abc123xyz")
        leaked, _ = canary.check("The output mentions CANARY-abc123 but not the full token.")
        assert leaked is False

    def test_raises_on_non_string(self):
        canary = CanaryToken()
        with pytest.raises(TypeError):
            canary.check(42)


class TestReport:
    def test_leak_report(self, capsys):
        canary = CanaryToken(token="CANARY-LEAK")
        canary.report("output contains CANARY-LEAK here")
        captured = capsys.readouterr().out
        assert "LEAK DETECTED" in captured
        assert "CANARY-LEAK" in captured

    def test_clean_report(self, capsys):
        canary = CanaryToken(token="CANARY-SAFE")
        canary.report("perfectly normal response")
        captured = capsys.readouterr().out
        assert "No leak detected" in captured


class TestRepr:
    def test_repr(self):
        canary = CanaryToken(token="CANARY-repr-test")
        assert "CANARY-repr-test" in repr(canary)
