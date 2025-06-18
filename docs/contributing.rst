Contributing to pytector
========================

Thank you for your interest in contributing to pytector! This document provides guidelines and information for contributors.

Getting Started
--------------

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a virtual environment** for development
4. **Install in development mode**

.. code-block:: bash

   git clone https://github.com/YOUR_USERNAME/pytector.git
   cd pytector
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .[test]

Development Setup
----------------

Install development dependencies:

.. code-block:: bash

   pip install -e .[test]
   pip install -r docs/requirements.txt

Run tests:

.. code-block:: bash

   pytest tests/

Build documentation locally:

.. code-block:: bash

   cd docs
   make html

Code Style
----------

* Follow PEP 8 style guidelines
* Use type hints for function parameters and return values
* Write docstrings for all public functions and classes
* Keep functions focused and concise

Example of good code style:

.. code-block:: python

   from typing import List, Optional
   
   def detect_injection(text: str, threshold: float = 0.5) -> DetectionResult:
       """
       Detect prompt injection in the given text.
       
       Args:
           text: The text to analyze
           threshold: Confidence threshold for detection
           
       Returns:
           DetectionResult with injection status and confidence
           
       Raises:
           ValueError: If text is empty or invalid
       """
       if not text.strip():
           raise ValueError("Text cannot be empty")
       
       # Implementation here
       pass

Testing
-------

* Write tests for all new functionality
* Ensure existing tests pass
* Aim for good test coverage
* Use descriptive test names

Example test:

.. code-block:: python

   def test_detect_injection_with_normal_text():
       """Test that normal text is not flagged as injection."""
       detector = PromptInjectionDetector()
       result = detector.detect("Hello, how are you?")
       assert not result.is_injection
       assert 0 <= result.confidence <= 1

Documentation
-------------

* Update documentation for any API changes
* Add examples for new features
* Keep docstrings up to date
* Test documentation builds locally

Pull Request Process
-------------------

1. **Create a feature branch** from main
2. **Make your changes** following the guidelines above
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Run tests** and ensure they pass
6. **Submit a pull request** with a clear description

Pull Request Guidelines
----------------------

* Use a clear, descriptive title
* Provide a detailed description of changes
* Reference any related issues
* Include examples if adding new features
* Ensure all CI checks pass

Example PR description:

.. code-block:: markdown

   ## Description
   
   Added support for custom model loading with improved error handling.
   
   ## Changes
   
   - Added `load_custom_model()` method
   - Improved error messages for model loading failures
   - Added validation for model file paths
   
   ## Testing
   
   - Added unit tests for new functionality
   - All existing tests pass
   - Tested with sample models
   
   ## Documentation
   
   - Updated API documentation
   - Added usage examples
   - Updated installation guide

Issue Reporting
---------------

When reporting issues, please include:

* Python version
* Operating system
* pytector version
* Steps to reproduce
* Expected vs actual behavior
* Error messages (if any)

Example issue:

.. code-block:: markdown

   **Environment:**
   - Python 3.11
   - Ubuntu 22.04
   - pytector 0.1.2
   
   **Issue:**
   When using GGUF models, detection fails with "Model not found" error.
   
   **Steps to reproduce:**
   1. Install pytector with GGUF support
   2. Try to load a GGUF model
   3. See error message
   
   **Expected behavior:**
   Model should load successfully
   
   **Actual behavior:**
   Raises ModelLoadError

Getting Help
-----------

* Check existing issues and pull requests
* Join discussions in GitHub issues
* Review the documentation
* Ask questions in GitHub discussions

Thank you for contributing to pytector! 