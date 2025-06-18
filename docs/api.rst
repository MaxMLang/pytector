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
     - Groq model to use for detection (default: meta-llama/llama-guard-4-12b)

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

Hazard Categories
-----------------

When using Groq API, the following hazard categories are supported:

.. list-table:: Hazard Categories
   :widths: 10 90
   :header-rows: 1

   * - Code
     - Description
   * - S1
     - Violent Crimes
   * - S2
     - Non-Violent Crimes
   * - S3
     - Sex-Related Crimes
   * - S4
     - Child Sexual Exploitation
   * - S5
     - Defamation
   * - S6
     - Specialized Advice
   * - S7
     - Privacy
   * - S8
     - Intellectual Property
   * - S9
     - Indiscriminate Weapons
   * - S10
     - Hate
   * - S11
     - Suicide & Self-Harm
   * - S12
     - Sexual Content
   * - S13
     - Elections
   * - S14
     - Code Interpreter Abuse

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
   is_safe, hazard_code = detector.detect_injection_api("Your text here")
   
   # Using GGUF model
   detector = PromptInjectionDetector("path/to/model.gguf")
   is_injection, probability = detector.detect_injection("Your text here")
   
   # Custom threshold
   detector = PromptInjectionDetector(default_threshold=0.8)
   is_injection, probability = detector.detect_injection("Your text here") 