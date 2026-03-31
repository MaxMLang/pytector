"""Tests for pytector.regex_scanner — RegexScanner.

Pure stdlib module — no mocking needed.
"""

import pytest

from pytector.regex_scanner import DEFAULT_PATTERNS, RegexScanner


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestInit:
    def test_defaults_loaded(self):
        scanner = RegexScanner()
        patterns = scanner.get_patterns()
        assert "EMAIL" in patterns
        assert "SSN" in patterns
        assert "CREDIT_CARD" in patterns

    def test_use_defaults_false(self):
        scanner = RegexScanner(use_defaults=False)
        assert scanner.get_patterns() == {}

    def test_custom_patterns_merge(self):
        scanner = RegexScanner(patterns={"CUSTOM": r"\bfoo\b"})
        patterns = scanner.get_patterns()
        assert "CUSTOM" in patterns
        assert "EMAIL" in patterns

    def test_custom_patterns_override(self):
        custom_email = r"[a-z]+@[a-z]+\.com"
        scanner = RegexScanner(patterns={"EMAIL": custom_email})
        assert scanner.get_patterns()["EMAIL"] == custom_email

    def test_custom_only(self):
        scanner = RegexScanner(patterns={"ONLY": r"\d+"}, use_defaults=False)
        assert list(scanner.get_patterns().keys()) == ["ONLY"]


# ---------------------------------------------------------------------------
# Default pattern matching
# ---------------------------------------------------------------------------

class TestDefaultPatterns:
    def test_email(self):
        scanner = RegexScanner()
        has, matches = scanner.scan("Contact user@example.com please")
        assert has is True
        assert any(m["pattern_name"] == "EMAIL" for m in matches)
        assert any(m["match"] == "user@example.com" for m in matches)

    def test_phone_us(self):
        scanner = RegexScanner()
        has, matches = scanner.scan("Call me at (555) 123-4567.")
        assert has is True
        assert any(m["pattern_name"] == "PHONE" for m in matches)

    def test_ssn(self):
        scanner = RegexScanner()
        has, matches = scanner.scan("My SSN is 123-45-6789")
        assert has is True
        assert any(m["pattern_name"] == "SSN" for m in matches)
        assert any(m["match"] == "123-45-6789" for m in matches)

    def test_credit_card_visa(self):
        scanner = RegexScanner()
        has, matches = scanner.scan("Card: 4111 1111 1111 1111")
        assert has is True
        assert any(m["pattern_name"] == "CREDIT_CARD" for m in matches)

    def test_ip_address(self):
        scanner = RegexScanner()
        has, matches = scanner.scan("Server at 192.168.1.100")
        assert has is True
        assert any(m["pattern_name"] == "IP_ADDRESS" for m in matches)
        assert any(m["match"] == "192.168.1.100" for m in matches)

    def test_api_key(self):
        scanner = RegexScanner()
        has, matches = scanner.scan("Key: sk-live-abcdef1234567890")
        assert has is True
        assert any(m["pattern_name"] == "API_KEY" for m in matches)

    def test_jwt_token(self):
        scanner = RegexScanner()
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123_signature"
        has, matches = scanner.scan(f"Token: {jwt}")
        assert has is True
        assert any(m["pattern_name"] == "JWT_TOKEN" for m in matches)

    def test_clean_text(self):
        scanner = RegexScanner()
        has, matches = scanner.scan("Just a normal sentence with nothing sensitive.")
        assert has is False
        assert matches == []


# ---------------------------------------------------------------------------
# Scan output shape
# ---------------------------------------------------------------------------

class TestScanShape:
    def test_match_dict_keys(self):
        scanner = RegexScanner()
        _, matches = scanner.scan("user@example.com")
        for m in matches:
            assert set(m.keys()) == {"pattern_name", "match", "start", "end"}

    def test_matches_sorted_by_start(self):
        scanner = RegexScanner()
        text = "SSN 123-45-6789 email user@example.com"
        _, matches = scanner.scan(text)
        starts = [m["start"] for m in matches]
        assert starts == sorted(starts)

    def test_raises_on_non_string(self):
        scanner = RegexScanner()
        with pytest.raises(TypeError):
            scanner.scan(42)


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

class TestRedact:
    def test_basic_redact(self):
        scanner = RegexScanner()
        result = scanner.redact("Email: user@example.com")
        assert "user@example.com" not in result
        assert "[REDACTED]" in result

    def test_custom_replacement(self):
        scanner = RegexScanner()
        result = scanner.redact("Email: user@example.com", replacement="***")
        assert "***" in result
        assert "[REDACTED]" not in result

    def test_multiple_redactions(self):
        scanner = RegexScanner()
        text = "SSN: 123-45-6789, email: a@b.com"
        result = scanner.redact(text)
        assert "123-45-6789" not in result
        assert "a@b.com" not in result
        assert result.count("[REDACTED]") >= 2

    def test_clean_text_unchanged(self):
        scanner = RegexScanner()
        text = "Nothing special here."
        assert scanner.redact(text) == text


# ---------------------------------------------------------------------------
# Pattern management
# ---------------------------------------------------------------------------

class TestPatternManagement:
    def test_add_pattern(self):
        scanner = RegexScanner(use_defaults=False)
        scanner.add_pattern("DIGITS", r"\d+")
        assert "DIGITS" in scanner.get_patterns()
        has, _ = scanner.scan("There are 42 items")
        assert has is True

    def test_remove_pattern(self):
        scanner = RegexScanner()
        scanner.remove_pattern("EMAIL")
        assert "EMAIL" not in scanner.get_patterns()
        has, matches = scanner.scan("user@example.com")
        assert not any(m["pattern_name"] == "EMAIL" for m in matches)

    def test_remove_nonexistent_is_noop(self):
        scanner = RegexScanner()
        scanner.remove_pattern("DOES_NOT_EXIST")

    def test_overwrite_pattern(self):
        scanner = RegexScanner()
        scanner.add_pattern("EMAIL", r"NOPE")
        has, _ = scanner.scan("user@example.com")
        assert has is False

    def test_get_patterns_is_copy(self):
        scanner = RegexScanner()
        patterns = scanner.get_patterns()
        patterns["INJECTED"] = r".*"
        assert "INJECTED" not in scanner.get_patterns()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

class TestReport:
    def test_report_with_matches(self, capsys):
        scanner = RegexScanner()
        scanner.report("Contact user@example.com")
        captured = capsys.readouterr().out
        assert "Regex scan" in captured
        assert "EMAIL" in captured

    def test_report_no_matches(self, capsys):
        scanner = RegexScanner()
        scanner.report("Clean text")
        assert "No matches found" in capsys.readouterr().out
