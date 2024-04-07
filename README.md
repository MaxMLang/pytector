# Pytector
![Build](https://img.shields.io/github/workflow/status/MaxMLang/pytector/workflow.yml?branch=main)
![Tests](https://img.shields.io/github/workflow/status/MaxMLang/pytector/tests.yml?branch=main)
![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/github/license/MaxMLang/pytector)
![Issues](https://img.shields.io/github/issues/MaxMLang/pytector)
![Pull Requests](https://img.shields.io/github/issues-pr/MaxMLang/pytector)
Pytector is a Python package designed to detect prompt injection in text inputs using state-of-the-art machine learning models from the transformers library.

## Disclaimer

The Pytector package is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.
Pytector is still a prototype and cannot provide 100% protection against prompt injection attacks!

## Features

- Detect prompt injections with pre-trained models.
- Support for multiple models including DeBERTa, DistilBERT, and ONNX versions.
- Easy-to-use interface with customizable threshold settings.

## Installation
Via PIP
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

