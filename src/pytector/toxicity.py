"""Toxicity detection using transformer sequence-classification models.

Default model: ``citizenlab/distilbert-base-multilingual-cased-toxicity``
(DistilBERT multilingual, F1-micro 0.94, 10 languages).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from transformers import pipeline


class ToxicityDetector:
    """Classify text as toxic or non-toxic.

    Parameters
    ----------
    model_name : str
        A key in :pyattr:`predefined_models` or any Hugging Face model ID /
        local path suitable for ``text-classification``.
    threshold : float
        Score above which text is considered toxic.
    """

    predefined_models: Dict[str, str] = {
        "citizenlab": "citizenlab/distilbert-base-multilingual-cased-toxicity",
    }

    def __init__(
        self,
        model_name: str = "citizenlab",
        threshold: float = 0.5,
    ) -> None:
        if not isinstance(threshold, (int, float)):
            raise ValueError("threshold must be a number.")

        self.default_threshold = float(threshold)

        model_path = self.predefined_models.get(model_name, model_name)
        if os.path.exists(model_name):
            model_path = model_name

        self._pipeline = pipeline(
            "text-classification",
            model=model_path,
        )

    def detect(
        self,
        text: str,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """Detect whether *text* is toxic.

        Returns ``(is_toxic, score)`` mirroring the
        ``PromptInjectionDetector.detect_injection`` return signature.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        threshold = self.default_threshold if threshold is None else float(threshold)

        results: List[Dict[str, Any]] = self._pipeline(text)

        toxic_score = self._extract_toxic_score(results)
        return toxic_score > threshold, toxic_score

    def report(self, text: str, threshold: Optional[float] = None) -> None:
        """Print a human-readable toxicity summary."""
        is_toxic, score = self.detect(text, threshold=threshold)
        if is_toxic:
            print(f"Toxic content detected (score={score:.2f}).")
        else:
            print(f"No toxicity detected (score={score:.2f}).")

    @staticmethod
    def _extract_toxic_score(results: List[Dict[str, Any]]) -> float:
        """Normalise pipeline output into a single toxicity probability.

        The citizenlab model returns ``[{"label": "toxic"|"non-toxic", "score": float}]``.
        Other models may use ``LABEL_1`` / ``LABEL_0`` conventions.
        """
        if not results:
            return 0.0

        result = results[0]
        label = result.get("label", "").lower()
        score = float(result.get("score", 0.0))

        if label in ("toxic", "label_1"):
            return score
        if label in ("non-toxic", "non_toxic", "not_toxic", "label_0"):
            return 1.0 - score
        return score
