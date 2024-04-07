# Pytector
<p align="center">
  <img src="https://github.com/MaxMLang/assets/blob/main/pytector-logo.png?raw=true" width="200" height="200" alt="Pytector Logo">
</p>

![Build](https://img.shields.io/github/actions/workflow/status/MaxMLang/pytector/.github/workflows/workflow.yml?branch=main)
![Tests](https://img.shields.io/github/actions/workflow/status/MaxMLang/pytector/.github/workflows/tests.yml?branch=main&label=tests)
![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![Issues](https://img.shields.io/github/issues/MaxMLang/pytector)
![Pull Requests](https://img.shields.io/github/issues-pr/MaxMLang/pytector)

Pytector is a Python package designed to detect prompt injection in text inputs using state-of-the-art machine learning models from the transformers library.

## Disclaimer
Pytector is still a prototype and cannot provide 100% protection against prompt injection attacks!

## Features

- Detect prompt injections with pre-trained models.
- Support for multiple models including DeBERTa, DistilBERT, and ONNX versions.
- Easy-to-use interface with customizable threshold settings.

## Installation
```bash
pip install pytector
```

Install Pytector directly from the source code:

```bash
git clone https://github.com/MaxMLang/pytector.git
cd pytector
pip install .
```



## Usage

To use Pytector, you can import the `PromptInjectionDetector` class and create an instance with a pre-defined model or a custom model URL.

```python
import pytector

# Initialize the detector with a pre-defined model
detector = pytector.PromptInjectionDetector(model_name_or_url="deberta")

# Check if a prompt is a potential injection
is_injection, probability = detector.detect_injection("Your suspicious prompt here")
print(f"Is injection: {is_injection}, Probability: {probability}")
```

## Documentation

For full documentation, visit the `docs` directory.

## Contributing

Contributions are welcome! Please read our [Contributing Guide](contributing.md) for details on our code of conduct, and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

