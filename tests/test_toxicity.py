"""Tests for pytector.toxicity — ToxicityDetector.

All tests mock the transformers pipeline to avoid downloading models.
"""

import pytest

from pytector.toxicity import ToxicityDetector


# ---------------------------------------------------------------------------
# Helpers — fake pipeline
# ---------------------------------------------------------------------------

def _fake_pipeline_factory(label, score):
    """Return a callable that mimics ``transformers.pipeline(...)``."""

    class FakePipeline:
        def __call__(self, text):
            return [{"label": label, "score": score}]

    def factory(*_args, **_kwargs):
        return FakePipeline()

    return factory


def _make_detector(monkeypatch, label="toxic", score=0.92, threshold=0.5):
    fake = _fake_pipeline_factory(label, score)
    monkeypatch.setattr("pytector.toxicity.pipeline", fake)
    return ToxicityDetector(threshold=threshold)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDetect:
    def test_toxic_above_threshold(self, monkeypatch):
        det = _make_detector(monkeypatch, label="toxic", score=0.92)
        is_toxic, score = det.detect("you are terrible")
        assert is_toxic is True
        assert score == pytest.approx(0.92)

    def test_toxic_below_threshold(self, monkeypatch):
        det = _make_detector(monkeypatch, label="toxic", score=0.3)
        is_toxic, score = det.detect("you are terrible")
        assert is_toxic is False
        assert score == pytest.approx(0.3)

    def test_non_toxic_label(self, monkeypatch):
        det = _make_detector(monkeypatch, label="non-toxic", score=0.95)
        is_toxic, score = det.detect("have a nice day")
        assert is_toxic is False
        assert score == pytest.approx(0.05)

    def test_not_toxic_label(self, monkeypatch):
        det = _make_detector(monkeypatch, label="not_toxic", score=0.99)
        is_toxic, score = det.detect("have a nice day")
        assert is_toxic is False
        assert score == pytest.approx(0.01)

    def test_label_0_convention(self, monkeypatch):
        det = _make_detector(monkeypatch, label="LABEL_0", score=0.90)
        is_toxic, score = det.detect("hello")
        assert is_toxic is False
        assert score == pytest.approx(0.10)

    def test_label_1_convention(self, monkeypatch):
        det = _make_detector(monkeypatch, label="LABEL_1", score=0.85)
        is_toxic, score = det.detect("offensive text")
        assert is_toxic is True
        assert score == pytest.approx(0.85)

    def test_per_call_threshold(self, monkeypatch):
        det = _make_detector(monkeypatch, label="toxic", score=0.6, threshold=0.5)
        is_toxic_default, _ = det.detect("text")
        assert is_toxic_default is True

        is_toxic_high, _ = det.detect("text", threshold=0.7)
        assert is_toxic_high is False

    def test_raises_on_non_string(self, monkeypatch):
        det = _make_detector(monkeypatch)
        with pytest.raises(TypeError):
            det.detect(42)

    def test_empty_results(self, monkeypatch):
        def empty_factory(*_a, **_kw):
            class FP:
                def __call__(self, text):
                    return []
            return FP()

        monkeypatch.setattr("pytector.toxicity.pipeline", empty_factory)
        det = ToxicityDetector()
        is_toxic, score = det.detect("hello")
        assert is_toxic is False
        assert score == 0.0


class TestReport:
    def test_report_toxic(self, monkeypatch, capsys):
        det = _make_detector(monkeypatch, label="toxic", score=0.88)
        det.report("bad text")
        captured = capsys.readouterr().out
        assert "Toxic content detected" in captured
        assert "0.88" in captured

    def test_report_clean(self, monkeypatch, capsys):
        det = _make_detector(monkeypatch, label="non-toxic", score=0.95)
        det.report("nice text")
        captured = capsys.readouterr().out
        assert "No toxicity detected" in captured


class TestInit:
    def test_bad_threshold_raises(self, monkeypatch):
        monkeypatch.setattr("pytector.toxicity.pipeline", lambda *a, **k: None)
        with pytest.raises(ValueError, match="threshold"):
            ToxicityDetector(threshold="high")


class TestExtractScore:
    def test_unknown_label_returns_raw_score(self, monkeypatch):
        det = _make_detector(monkeypatch, label="SOMETHING_ELSE", score=0.77)
        _, score = det.detect("text")
        assert score == pytest.approx(0.77)
