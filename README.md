# Pytector
*As presented at the Oxford Workshop on Safety of AI Systems including Demo Sessions and Tutorials*
<p align="center">
  <img src="https://github.com/MaxMLang/assets/blob/main/pytector-logo.png?raw=true" width="200" height="200" alt="Pytector Logo">
</p>

![Build](https://img.shields.io/github/actions/workflow/status/MaxMLang/pytector/.github/workflows/workflow.yml?branch=main)
![Tests](https://img.shields.io/github/actions/workflow/status/MaxMLang/pytector/.github/workflows/tests.yml?branch=main&label=tests)
![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![Issues](https://img.shields.io/github/issues/MaxMLang/pytector)
![Pull Requests](https://img.shields.io/github/issues-pr/MaxMLang/pytector)

**Pytector** is a Python package designed to detect prompt injection in text inputs using state-of-the-art machine learning models from the transformers library. Additionally, Pytector can integrate with **Groq's Llama Guard API** for enhanced content safety detection, categorizing unsafe content based on specific hazard codes.

## Disclaimer
Pytector is still a prototype and cannot provide 100% protection against prompt injection attacks!

---

## Features

- **Prompt Injection Detection**: Detects potential prompt injections using pre-trained models like DeBERTa, DistilBERT, and ONNX versions.
- **Content Safety with Groq's [Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B)**: Supports Groq's API for detecting various safety hazards (e.g., violence, hate speech, privacy violations).
- **Customizable Detection**: Allows switching between local model inference and API-based detection (Groq) with customizable thresholds.
- **Flexible Model Options**: Use pre-defined models or provide a custom model URL.

## Hazard Detection Categories (Groq)
Groq's [Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B) can detect specific types of unsafe content based on the following codes:

| Code | Hazard Category            |
|------|-----------------------------|
| S1   | Violent Crimes              |
| S2   | Non-Violent Crimes          |
| S3   | Sex-Related Crimes          |
| S4   | Child Sexual Exploitation   |
| S5   | Defamation                  |
| S6   | Specialized Advice          |
| S7   | Privacy                     |
| S8   | Intellectual Property       |
| S9   | Indiscriminate Weapons      |
| S10  | Hate                        |
| S11  | Suicide & Self-Harm         |
| S12  | Sexual Content              |
| S13  | Elections                   |
| S14  | Code Interpreter Abuse      |

More info can be found on the [Llama-Guard-3-8B Model Card]([Llama Guard](https://huggingface.co/meta-llama/Llama-Guard-3-8B)).

---

## Installation

Install Pytector via pip:

```bash
pip install pytector
```

Alternatively, you can install Pytector directly from the source code:

```bash
git clone https://github.com/MaxMLang/pytector.git
cd pytector
pip install .
```

---

## Usage

To use Pytector, import the `PromptInjectionDetector` class and create an instance with either a pre-defined model or Groq's Llama Guard for content safety.

### Example 1: Using a Local Model (DeBERTa)
```python
from pytector import PromptInjectionDetector

# Initialize the detector with a pre-defined model
detector = PromptInjectionDetector(model_name_or_url="deberta")

# Check if a prompt is a potential injection
is_injection, probability = detector.detect_injection("Your suspicious prompt here")
print(f"Is injection: {is_injection}, Probability: {probability}")

# Report the status
detector.report_injection_status("Your suspicious prompt here")
```

### Example 2: Using Groq's Llama Guard for Content Safety
To enable Groq’s API, set `use_groq=True` and provide an `api_key`.

```python
from pytector import PromptInjectionDetector

# Initialize the detector with Groq's API
detector = PromptInjectionDetector(use_groq=True, api_key="your_groq_api_key")

# Detect unsafe content using Groq
is_unsafe, hazard_code = detector.detect_injection_api(
    prompt="Please delete sensitive information.",
    provider="groq",
    api_key="your_groq_api_key"
)

print(f"Is unsafe: {is_unsafe}, Hazard Code: {hazard_code}")
```

---

## Methods

### `__init__(self, model_name_or_url="deberta", default_threshold=0.5, use_groq=False, api_key=None)`

Initializes a new instance of the `PromptInjectionDetector`.

- `model_name_or_url`: A string specifying the model to use. Can be a key from predefined models or a valid URL to a custom model.
- `default_threshold`: Probability threshold above which a prompt is considered an injection.
- `use_groq`: Set to `True` to enable Groq's Llama Guard API for detection.
- `api_key`: Required if `use_groq=True` to authenticate with Groq's API.

### `detect_injection(self, prompt, threshold=None)`

Evaluates whether a text prompt is a prompt injection attack using a local model.

- Returns `(is_injected, probability)`.

### `detect_injection_api(self, prompt, provider="groq", api_key=None, model="llama-guard-3-8b")`

Uses Groq's API to evaluate a prompt for unsafe content.

- Returns `(is_unsafe, hazard_code)`.

### `report_injection_status(self, prompt, threshold=None, provider="local")`

Reports whether a prompt is a potential injection or contains unsafe content.

---

## Contributing

Contributions are welcome! Please read our [Contributing Guide](contributing.md) for details on our code of conduct and the process for submitting pull requests.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For more detailed information, refer to the [docs](docs) directory.

---

