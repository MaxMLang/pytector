# Pytector

Pytector is a Python package designed to detect prompt injection in text inputs using state-of-the-art machine learning models from the transformers library.

## Features

- Detect prompt injections with pre-trained models.
- Support for multiple models including DeBERTa, DistilBERT, and ONNX versions.
- Easy-to-use interface with customizable threshold settings.

## Installation

Install Pytector directly from the source code:

```bash
git clone https://github.com/yourusername/pytector.git
cd pytector
pip install .
```

## Usage

To use Pytector, you can import the `PromptInjectionDetector` class and create an instance with a pre-defined model or a custom model URL.

```python
from pytector.detector import PromptInjectionDetector

# Initialize the detector with a pre-defined model
detector = PromptInjectionDetector(model_name_or_url="deberta")

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

