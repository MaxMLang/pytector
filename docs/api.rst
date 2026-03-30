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
