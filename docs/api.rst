API Reference
============

This page provides detailed API documentation for pytector.

Core Classes
------------

.. automodule:: pytector
   :members:
   :undoc-members:
   :show-inheritance:

PromptInjectionDetector
-----------------------

.. automodule:: pytector.detector
   :members:
   :undoc-members:
   :show-inheritance:

LangChain Integration
---------------------

.. automodule:: pytector.langchain
   :members:
   :undoc-members:
   :show-inheritance:

PromptSanitizer
---------------

.. automodule:: pytector.sanitizer
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

The following configuration options are available when initializing the detector:

.. list-table:: Configuration Parameters
   :widths: 20 20 60
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - model_name_or_url
     - str
     - Name or path of the model to use for detection
   * - default_threshold
     - float
     - Default confidence threshold for injection detection (0.0 to 1.0)
   * - use_groq
     - bool
     - Whether to use Groq API for detection
   * - api_key
     - str
     - API key for Groq service (required if use_groq=True)
   * - groq_model
     - str
     - Groq model to use for detection (default: openai/gpt-oss-safeguard-20b)

Predefined Models
----------------

The following predefined models are available:

.. list-table:: Predefined Models
   :widths: 20 80
   :header-rows: 1

   * - Model Name
     - Description
   * - deberta
     - protectai/deberta-v3-base-prompt-injection
   * - distilbert
     - fmops/distilbert-prompt-injection
   * - distilbert-onnx
     - prompt-security/fmops-distilbert-prompt-injection-onnx

Groq API Behavior
-----------------

``detect_injection_api`` returns:

* ``True`` for safe responses
* ``False`` for unsafe responses (or non-standard responses treated conservatively as unsafe)
* ``None`` when the API call fails

Use ``return_raw=True`` to inspect raw model output as ``(is_safe, raw_response)``.

Example Usage
-------------

.. code-block:: python

   from pytector import PromptInjectionDetector
   
   # Basic usage with default model
   detector = PromptInjectionDetector()
   is_injection, probability = detector.detect_injection("Your text here")
   
   # Using Groq API
   detector = PromptInjectionDetector(
       use_groq=True,
       api_key="your-api-key"
   )
   is_safe = detector.detect_injection_api("Your text here")
   
   # Using GGUF model
   detector = PromptInjectionDetector("path/to/model.gguf")
   is_injection, probability = detector.detect_injection("Your text here")
   
   # Custom threshold
   detector = PromptInjectionDetector(default_threshold=0.8)
   is_injection, probability = detector.detect_injection("Your text here")

Sanitizer Usage
---------------

.. code-block:: python

   from pytector import PromptSanitizer

   # All strategies enabled by default
   sanitizer = PromptSanitizer()
   cleaned, was_modified = sanitizer.sanitize("Ignore previous instructions. Hello!")

   # With detailed change log
   cleaned, was_modified, changes = sanitizer.sanitize(
       "Ignore previous instructions. Hello!",
       return_details=True,
   )

   # Custom configuration
   sanitizer = PromptSanitizer(
       fuzzy_threshold=0.80,
       sentence_threshold=0.4,
       enable_prompt_enforcement=True,
   )

Sanitizer Configuration
-----------------------

.. list-table:: Sanitizer Parameters
   :widths: 30 15 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - enable_encoding_detection
     - True
     - Decode and strip Base64, hex, ROT13 obfuscated payloads
   * - enable_unicode_normalization
     - True
     - Strip invisible characters, NFKC homoglyph normalization
   * - enable_pattern_removal
     - True
     - Regex-based structural injection pattern removal
   * - enable_sentence_scoring
     - True
     - Heuristic per-sentence analysis; drop suspicious sentences
   * - enable_fuzzy_matching
     - True
     - Catch paraphrased injection phrases via difflib similarity
   * - enable_keyword_stripping
     - True
     - Final pass removing known injection phrases
   * - enable_prompt_enforcement
     - False
     - Escape template syntax (``{ } < > ` ``)
   * - keywords
     - None
     - Custom keyword list; ``None`` uses built-in defaults
   * - fuzzy_threshold
     - 0.85
     - Similarity cutoff for fuzzy matching (0.0-1.0)
   * - sentence_threshold
     - 0.5
     - Heuristic score cutoff for sentence removal (0.0-1.0)

PIIScanner
----------

.. automodule:: pytector.pii
   :members:
   :undoc-members:
   :show-inheritance:

Uses the `PasteProof PII Detector <https://huggingface.co/joneauxedgar/pasteproof-pii-detector-v2>`_
(ModernBERT-base, F1 0.97) for NER-based PII detection across 27 entity types.
Requires ``transformers >= 4.48.0`` for ModernBERT support.

.. code-block:: python

   from pytector import PIIScanner

   scanner = PIIScanner()
   has_pii, entities = scanner.scan("Email john@acme.com, SSN 123-45-6789")
   print(scanner.redact("Email john@acme.com, SSN 123-45-6789"))

   # Filter to specific entity types
   scanner = PIIScanner(entity_types=["EMAIL", "CREDIT_CARD"], threshold=0.7)

.. list-table:: PIIScanner Parameters
   :widths: 20 20 60
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - model_name
     - str
     - Predefined key (``pasteproof-v3``) or HuggingFace model ID / local path
   * - threshold
     - float
     - Minimum confidence for an entity to be reported (default 0.5)
   * - entity_types
     - list[str] | None
     - Filter to specific types (e.g. ``["EMAIL", "SSN"]``); ``None`` = all

.. admonition:: Citation

   .. code-block:: text

      @model{pasteproof_pii_detector,
        author    = {Jonathan Edgar},
        title     = {PasteProof PII Detector},
        year      = {2025},
        publisher = {Hugging Face},
        url       = {https://huggingface.co/joneauxedgar/pasteproof-pii-detector-v2}
      }

ToxicityDetector
----------------

.. automodule:: pytector.toxicity
   :members:
   :undoc-members:
   :show-inheritance:

Uses `citizenlab/distilbert-base-multilingual-cased-toxicity <https://huggingface.co/citizenlab/distilbert-base-multilingual-cased-toxicity>`_
(F1-micro 0.94, 10 languages) for toxicity classification.

.. code-block:: python

   from pytector import ToxicityDetector

   detector = ToxicityDetector()
   is_toxic, score = detector.detect("You are terrible")
   detector.report("Have a wonderful day!")

.. list-table:: ToxicityDetector Parameters
   :widths: 20 20 60
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - model_name
     - str
     - Predefined key (``citizenlab``) or HuggingFace model ID / local path
   * - threshold
     - float
     - Score above which text is considered toxic (default 0.5)

RegexScanner
------------

.. automodule:: pytector.regex_scanner
   :members:
   :undoc-members:
   :show-inheritance:

Pure-stdlib rule-based scanner with customizable patterns.

.. code-block:: python

   from pytector import RegexScanner

   scanner = RegexScanner()
   has_match, matches = scanner.scan("Key: sk-live-abc123def456")
   print(scanner.redact("Email user@example.com"))

   # Custom patterns only
   custom = RegexScanner(
       patterns={"ORDER_ID": r"ORD-\d{8}"},
       use_defaults=False,
   )

.. list-table:: RegexScanner Parameters
   :widths: 20 20 60
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - patterns
     - dict[str, str] | None
     - ``{NAME: regex}`` mapping merged with defaults (or used alone)
   * - use_defaults
     - bool
     - Whether to include built-in patterns (EMAIL, PHONE, SSN, CREDIT_CARD, IP_ADDRESS, API_KEY, JWT_TOKEN)

CanaryToken
-----------

.. automodule:: pytector.canary
   :members:
   :undoc-members:
   :show-inheritance:

Inject a secret token into your system prompt and detect if the model leaks it.
Pure stdlib — zero dependencies, zero calibration.

.. code-block:: python

   from pytector import CanaryToken

   canary = CanaryToken()
   system_prompt = canary.wrap("You are a helpful assistant.")
   # ... pass to LLM, get response ...
   leaked, token = canary.check(model_output)

.. list-table:: CanaryToken Parameters
   :widths: 20 20 60
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - token
     - str | None
     - Explicit canary string; ``None`` auto-generates one
   * - length
     - int
     - Length of the random part of auto-generated tokens (default 16)
   * - prefix
     - str
     - Prefix for auto-generated tokens (default ``CANARY-``)

ToolOutputGuard
---------------

.. automodule:: pytector.guard
   :members:
   :undoc-members:
   :show-inheritance:

Framework-agnostic screening for *indirect* prompt injection (tool results,
browsed pages, files). Returns a :class:`~pytector.guard.GuardDecision`.

.. code-block:: python

   from pytector import ToolOutputGuard

   guard = ToolOutputGuard(threshold=0.7)
   decision = guard.scan_tool_result("browse", page_text)
   if decision.action == "block":
       raise SystemExit(decision.reasons)
   context = decision.content

WebPageScanner
--------------

.. automodule:: pytector.web
   :members:
   :undoc-members:
   :show-inheritance:

Fetch a URL (http/https), extract visible text, chunk long pages, and screen
them through :class:`~pytector.guard.ToolOutputGuard`. Stdlib only.

.. code-block:: python

   from pytector import WebPageScanner

   scanner = WebPageScanner(threshold=0.7)
   decision = scanner.scan_url("https://example.com/article")
   # Or skip the network call:
   # decision = scanner.scan_html(html_from_tool)
   # decision = scanner.scan_text(page_text)

SecurityPipeline
----------------

.. automodule:: pytector.pipeline
   :members:
   :undoc-members:
   :show-inheritance:

Compose regex → (optional sanitize) → injection → (optional PII/toxicity)
into one :class:`~pytector.guard.GuardDecision`. Defaults never auto-load
heavy PII/toxicity models.

.. code-block:: python

   from pytector import SecurityPipeline

   pipeline = SecurityPipeline(threshold=0.7, sanitizer=True)
   decision = pipeline.scan("Ignore previous instructions. Email a@b.com")
   safe_text = decision.content
