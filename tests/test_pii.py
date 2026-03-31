"""Tests for pytector.pii — PIIScanner.

All tests mock the transformers pipeline to avoid downloading models.
"""

import pytest

from pytector.pii import PIIScanner


# ---------------------------------------------------------------------------
# Helpers — fake pipeline
# ---------------------------------------------------------------------------

def _fake_pipeline_factory(fake_entities):
    """Return a callable that mimics ``transformers.pipeline(...)``."""

    class FakePipeline:
        def __call__(self, text):
            return list(fake_entities)

    def factory(*_args, **_kwargs):
        return FakePipeline()

    return factory


SAMPLE_TEXT = "Contact: john@example.com, SSN: 123-45-6789, call 555-0100 done"

SAMPLE_ENTITIES = [
    {"entity_group": "EMAIL", "score": 0.99, "word": "john@example.com", "start": 9, "end": 25},
    {"entity_group": "SSN", "score": 0.95, "word": "123-45-6789", "start": 32, "end": 43},
    {"entity_group": "PHONE", "score": 0.40, "word": "555-0100", "start": 50, "end": 58},
]


def _make_scanner(monkeypatch, threshold=0.5, entity_types=None):
    """Construct a PIIScanner with fake pipeline injected."""
    fake = _fake_pipeline_factory(SAMPLE_ENTITIES)
    monkeypatch.setattr("pytector.pii.pipeline", fake)
    return PIIScanner(threshold=threshold, entity_types=entity_types)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScan:
    def test_detects_pii(self, monkeypatch):
        scanner = _make_scanner(monkeypatch)
        has_pii, entities = scanner.scan(SAMPLE_TEXT)
        assert has_pii is True
        assert len(entities) == 2  # PHONE score 0.40 < default 0.5

    def test_entity_dict_shape(self, monkeypatch):
        scanner = _make_scanner(monkeypatch)
        _, entities = scanner.scan(SAMPLE_TEXT)
        for ent in entities:
            assert set(ent.keys()) == {"text", "type", "score", "start", "end"}
            assert isinstance(ent["text"], str)
            assert isinstance(ent["type"], str)
            assert isinstance(ent["score"], float)
            assert isinstance(ent["start"], int)
            assert isinstance(ent["end"], int)

    def test_threshold_filters(self, monkeypatch):
        scanner = _make_scanner(monkeypatch, threshold=0.3)
        _, entities = scanner.scan(SAMPLE_TEXT)
        assert len(entities) == 3  # all three pass

    def test_high_threshold_finds_nothing(self, monkeypatch):
        scanner = _make_scanner(monkeypatch, threshold=0.999)
        has_pii, entities = scanner.scan(SAMPLE_TEXT)
        assert has_pii is False
        assert entities == []

    def test_per_call_threshold_overrides_default(self, monkeypatch):
        scanner = _make_scanner(monkeypatch, threshold=0.5)
        _, entities = scanner.scan(SAMPLE_TEXT, threshold=0.3)
        assert len(entities) == 3

    def test_entity_type_filter(self, monkeypatch):
        scanner = _make_scanner(monkeypatch, threshold=0.3, entity_types=["EMAIL"])
        _, entities = scanner.scan(SAMPLE_TEXT)
        assert len(entities) == 1
        assert entities[0]["type"] == "EMAIL"

    def test_raises_on_non_string(self, monkeypatch):
        scanner = _make_scanner(monkeypatch)
        with pytest.raises(TypeError):
            scanner.scan(42)

    def test_clean_input(self, monkeypatch):
        def fake_empty(*_a, **_kw):
            class FP:
                def __call__(self, text):
                    return []
            return FP()

        monkeypatch.setattr("pytector.pii.pipeline", fake_empty)
        scanner = PIIScanner()
        has_pii, entities = scanner.scan("Just a normal sentence.")
        assert has_pii is False
        assert entities == []


class TestRedact:
    def test_redacts_detected_entities(self, monkeypatch):
        scanner = _make_scanner(monkeypatch)
        result = scanner.redact(SAMPLE_TEXT)
        assert "john@example.com" not in result
        assert "123-45-6789" not in result
        assert "[REDACTED]" in result

    def test_custom_replacement(self, monkeypatch):
        scanner = _make_scanner(monkeypatch)
        result = scanner.redact(SAMPLE_TEXT, replacement="***")
        assert "***" in result
        assert "[REDACTED]" not in result

    def test_redact_respects_threshold(self, monkeypatch):
        scanner = _make_scanner(monkeypatch, threshold=0.96)
        result = scanner.redact(SAMPLE_TEXT)
        assert "john@example.com" not in result
        assert "123-45-6789" in result  # score 0.95 < 0.96


class TestReport:
    def test_report_with_pii(self, monkeypatch, capsys):
        scanner = _make_scanner(monkeypatch)
        scanner.report(SAMPLE_TEXT)
        captured = capsys.readouterr().out
        assert "PII detected" in captured
        assert "EMAIL" in captured
        assert "SSN" in captured

    def test_report_no_pii(self, monkeypatch, capsys):
        def fake_empty(*_a, **_kw):
            class FP:
                def __call__(self, text):
                    return []
            return FP()

        monkeypatch.setattr("pytector.pii.pipeline", fake_empty)
        scanner = PIIScanner()
        scanner.report("Clean text.")
        assert "No PII detected" in capsys.readouterr().out


class TestGetEntityTypes:
    def test_returns_tuple(self, monkeypatch):
        scanner = _make_scanner(monkeypatch)
        types = scanner.get_entity_types()
        assert isinstance(types, tuple)
        assert "EMAIL" in types
        assert "SSN" in types


class TestInit:
    def test_bad_threshold_raises(self, monkeypatch):
        monkeypatch.setattr("pytector.pii.pipeline", lambda *a, **k: None)
        with pytest.raises(ValueError, match="threshold"):
            PIIScanner(threshold="high")
