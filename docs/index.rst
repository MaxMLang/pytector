Welcome to pytector's documentation!
=====================================

**pytector** is a Python package for detecting prompt injections in text using Open-Source Large Language Models (LLMs), designed to provide immediate security controls beyond foundation model defaults.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples
   contributing

Features
--------

* **Prompt Injection Detection**: Uses open-source language models for prompt injection detection
* **Content Safety**: Support for Groq-hosted safeguard models for safety detection
* **Keyword-Based Blocking**: Restrictive keyword filtering for immediate security control
* **Multiple Model Backends**: Support for Hugging Face Transformers and GGUF models
* **Rapid Deployment**: Designed for quick integration into projects needing immediate security layers
* **Configurable**: Customizable detection parameters, thresholds, and security policies

Quick Start
----------

Install the package:

.. code-block:: bash

   pip install pytector

Basic usage:

.. code-block:: python

   from pytector import PromptInjectionDetector
   
   detector = PromptInjectionDetector()
   is_injection, probability = detector.detect_injection("Hello, how are you?")
   print(f"Injection detected: {is_injection}")

For more detailed information, see the :doc:`quickstart` guide.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 
