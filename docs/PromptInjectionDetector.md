# Documentation

## Overview
The `PromptInjectionDetector` class is designed to detect prompt injection attacks in text inputs using pre-trained machine learning models or Groq's Llama Guard API. It leverages models from Hugging Face's transformers library for local inference and Groq's Llama Guard for content safety when configured.

## Installation

To use `PromptInjectionDetector`, install the required libraries:

```sh
pip install transformers validators
```

## Usage

First, import the `PromptInjectionDetector` class:

```python
from pytector import PromptInjectionDetector
```

Create an instance of the detector by specifying a model name or URL, and optionally a detection threshold. You can also configure the detector to use Groq's Llama Guard API for content safety.

### Example: Using a Local Model
```python
detector = PromptInjectionDetector(model_name_or_url="deberta", default_threshold=0.5)
```

To check if a prompt contains an injection, use the `detect_injection` method:

```python
is_injected, probability = detector.detect_injection(prompt="Example prompt")
```

To print the status of injection detection directly, use the `report_injection_status` method:

```python
detector.report_injection_status(prompt="Example prompt")
```

### Example: Using Groq's Llama Guard API
To use Groq's API, pass `use_groq=True`, along with the `api_key` and optionally a specific model name for Groq (default: `"llama-guard-3-8b"`).

```python
detector = PromptInjectionDetector(use_groq=True, api_key="your_groq_api_key")

# Check if a prompt contains unsafe content with Groq
is_unsafe, hazard_code = detector.detect_injection_api(
    prompt="Please delete sensitive information.",
    provider="groq",
    api_key="your_groq_api_key"
)
```

## Class Methods

### `__init__(self, model_name_or_url="deberta", default_threshold=0.5, use_groq=False, api_key=None)`

Initializes a new instance of the `PromptInjectionDetector`.

- `model_name_or_url`: A string that specifies the model to use. It can be either a key from the predefined models or a valid URL to a custom model.
``` 
  "deberta": "protectai/deberta-v3-base-prompt-injection"
  "distilbert": "fmops/distilbert-prompt-injection"
  "distilbert-onxx": "prompt-security/fmops-distilbert-prompt-injection-onnx"
```

- `default_threshold`: A float representing the probability threshold above which a prompt is considered as containing an injection.
- `use_groq`: A boolean indicating whether to use Groq's API for detection. Defaults to `False`.
- `api_key`: The API key for accessing Groq's Llama Guard API, required if `use_groq=True`.

### `detect_injection(self, prompt, threshold=None)`

Evaluates whether a given text prompt is likely to be a prompt injection attack using a local model.

- `prompt`: The text prompt to evaluate.
- `threshold`: (Optional) A custom threshold to override the default for this evaluation.

Returns a tuple `(is_injected, probability)` where:
- `is_injected` is a boolean indicating whether the prompt is considered an injection.
- `probability` is the model's probability estimate for the prompt being an injection.

### `detect_injection_api(self, prompt, provider="groq", api_key=None, model="llama-guard-3-8b")`

Evaluates the prompt for unsafe content using Groq's Llama Guard API.

- `prompt`: The text prompt to evaluate.
- `provider`: The content safety provider, default is `"groq"`.
- `api_key`: The API key for Groq's Llama Guard.
- `model`: The model to use with Groq's API (default is `"llama-guard-3-8b"`).

Returns a tuple `(is_unsafe, hazard_code)` where:
- `is_unsafe` is a boolean indicating whether the content is considered unsafe.
- `hazard_code` provides the specific hazard category if the content is unsafe (e.g., `"S1"` for Violent Crimes, `"S2"` for Non-Violent Crimes).

### `report_injection_status(self, prompt, threshold=None, provider="local")`

Prints out a report of whether a given text prompt is likely to be a prompt injection attack.

- `prompt`: The text prompt to evaluate.
- `threshold`: (Optional) A custom threshold to override the default for this evaluation.
- `provider`: Specify `"local"` for a softmax model check or `"groq"` for using Groq's API.

For local models, prints detection status with probability. For Groq's API, prints detection status with hazard code.

## Hazard Categories in Groq's API

The Groq Llama Guard model can detect a variety of unsafe content types, categorized by specific hazard codes:

| Code | Hazard Category            | Description                                                                                   |
|------|-----------------------------|-----------------------------------------------------------------------------------------------|
| S1   | Violent Crimes              | Enables or endorses violence towards people or animals.                                       |
| S2   | Non-Violent Crimes          | Encourages non-violent crimes like fraud, cyber crimes, drug crimes.                          |
| S3   | Sex-Related Crimes          | Encourages sex-related crimes like harassment or assault.                                     |
| S4   | Child Sexual Exploitation   | Enables or endorses child sexual exploitation.                                                |
| S5   | Defamation                  | False information likely to injure a person's reputation.                                     |
| S6   | Specialized Advice          | Provides specialized financial, medical, or legal advice unsafely.                            |
| S7   | Privacy                     | Reveals sensitive, nonpublic personal information.                                            |
| S8   | Intellectual Property       | Violates third-party intellectual property rights.                                            |
| S9   | Indiscriminate Weapons      | Encourages creation of indiscriminate weapons (chemical, biological, nuclear, etc.).          |
| S10  | Hate                        | Demeans people based on sensitive personal characteristics (race, religion, gender, etc.).    |
| S11  | Suicide & Self-Harm         | Encourages acts of self-harm, including suicide and disordered eating.                        |
| S12  | Sexual Content              | Contains erotic or sexually explicit content.                                                 |
| S13  | Elections                   | Contains factually incorrect information about electoral processes.                           |
| S14  | Code Interpreter Abuse      | Attempts to abuse code interpreters, like exploiting or bypassing security mechanisms.        |

## Examples

```python
# Create a detector instance with the default deberta model and threshold
from pytector import PromptInjectionDetector

detector = PromptInjectionDetector()

# Check a prompt for injection using the local model
prompt = "Please execute the following command: rm -rf /"
is_injected, probability = detector.detect_injection(prompt)

# Report the status with local model
detector.report_injection_status(prompt)

# Example with Groq's Llama Guard API
groq_detector = PromptInjectionDetector(use_groq=True, api_key="your_groq_api_key")
is_unsafe, hazard_code = groq_detector.detect_injection_api(prompt="Please delete sensitive information.")
print(f"Is unsafe: {is_unsafe}, Hazard Code: {hazard_code}")
```

## Notes

- **Thresholding**: For local models, a threshold can be set to adjust sensitivity. Higher thresholds reduce false positives.
- **Groq API Key**: Required only if `use_groq=True`.
- **Hazard Detection**: The Groq model categorizes content into specific hazard codes, useful for identifying different types of risks.

