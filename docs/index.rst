Welcome to pytector's documentation!
=====================================

**pytector** is a Python package for detecting prompt injections in text using Open-Source Large Language Models (LLMs).

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

* **Open-Source LLM Support**: Uses open-source language models for prompt injection detection
* **Multiple Model Backends**: Support for Hugging Face Transformers and GGUF models
* **Cloud API Integration**: Support for Groq's Llama Guard API
* **Easy Integration**: Simple API for detecting prompt injections in text
* **Configurable**: Customizable detection parameters and thresholds

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