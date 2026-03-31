"""PII (Personally Identifiable Information) detection using transformer NER models.

Default model: ``joneauxedgar/pasteproof-pii-detector-v2`` (ModernBERT-base,
F1 0.97, 27 entity types — hosted as v3 weights on HuggingFace).
Requires ``transformers >= 4.48.0`` for ModernBERT support.

Citation
--------
.. code-block:: text

    @model{pasteproof_pii_detector,
      author = {Jonathan Edgar},
      title  = {PasteProof PII Detector},
      year   = {2025},
      publisher = {Hugging Face},
      url    = {https://huggingface.co/joneauxedgar/pasteproof-pii-detector-v2}
    }
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from transformers import pipeline


class PIIScanner:
    """Detect and optionally redact PII entities in text.

    Parameters
    ----------
    model_name : str
        A key in :pyattr:`predefined_models` or any Hugging Face model ID /
        local path suitable for ``token-classification``.
    threshold : float
        Minimum confidence score for an entity to be reported.
    entity_types : list[str] | None
        If provided, only entities whose type is in this list are returned.
        ``None`` means all entity types are returned.
    """

    predefined_models: Dict[str, str] = {
        "pasteproof-v3": "joneauxedgar/pasteproof-pii-detector-v2",
    }

    SUPPORTED_ENTITY_TYPES: Tuple[str, ...] = (
        "CREDIT_CARD", "PCI_PAN", "PCI_TRACK", "PCI_EXPIRY",
        "API_KEY", "AWS_KEY", "PRIVATE_KEY", "PASSWORD",
        "HIPAA_MRN", "HIPAA_ACCOUNT", "HIPAA_DOB",
        "GDPR_PASSPORT", "GDPR_NIN", "GDPR_IBAN",
        "NAME", "FIRST_NAME", "LAST_NAME", "SSN", "DOB", "DRIVER_LICENSE",
        "EMAIL", "PHONE", "IP_ADDRESS",
        "STREET", "CITY", "STATE", "ZIPCODE",
    )

    def __init__(
        self,
        model_name: str = "pasteproof-v3",
        threshold: float = 0.5,
        entity_types: Optional[List[str]] = None,
    ) -> None:
        if not isinstance(threshold, (int, float)):
            raise ValueError("threshold must be a number.")

        self.default_threshold = float(threshold)
        self.entity_types = (
            [e.upper() for e in entity_types] if entity_types else None
        )

        model_path = self.predefined_models.get(model_name, model_name)
        if os.path.exists(model_name):
            model_path = model_name

        self._pipeline = pipeline(
            "token-classification",
            model=model_path,
            aggregation_strategy="simple",
        )

    def scan(
        self,
        text: str,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Scan *text* for PII entities.

        Returns ``(has_pii, entities)`` where each entity dict contains
        ``text``, ``type``, ``score``, ``start``, and ``end``.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        threshold = self.default_threshold if threshold is None else float(threshold)

        raw_entities = self._pipeline(text)
        entities: List[Dict[str, Any]] = []
        for ent in raw_entities:
            score = float(ent["score"])
            if score < threshold:
                continue
            entity_type = ent["entity_group"].upper()
            if self.entity_types and entity_type not in self.entity_types:
                continue
            entities.append({
                "text": ent["word"].strip(),
                "type": entity_type,
                "score": score,
                "start": int(ent["start"]),
                "end": int(ent["end"]),
            })

        return len(entities) > 0, entities

    def redact(
        self,
        text: str,
        threshold: Optional[float] = None,
        replacement: str = "[REDACTED]",
    ) -> str:
        """Return a copy of *text* with detected PII replaced by *replacement*.

        Entities are replaced right-to-left so character offsets stay valid.
        """
        _, entities = self.scan(text, threshold=threshold)
        entities_sorted = sorted(entities, key=lambda e: e["start"], reverse=True)
        result = text
        for ent in entities_sorted:
            result = result[: ent["start"]] + replacement + result[ent["end"] :]
        return result

    def report(self, text: str, threshold: Optional[float] = None) -> None:
        """Print a human-readable PII scan summary."""
        has_pii, entities = self.scan(text, threshold=threshold)
        if not has_pii:
            print("No PII detected.")
            return
        print(f"PII detected — {len(entities)} entit{'y' if len(entities) == 1 else 'ies'} found:")
        for ent in entities:
            print(
                f"  [{ent['type']}] \"{ent['text']}\" "
                f"(score={ent['score']:.2f}, pos={ent['start']}:{ent['end']})"
            )

    def get_entity_types(self) -> Tuple[str, ...]:
        """Return the tuple of entity types supported by the default model."""
        return self.SUPPORTED_ENTITY_TYPES
