import validators
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class PromptInjectionDetector:
    predefined_models = {
        "deberta": "protectai/deberta-v3-base-prompt-injection",
        "distilbert": "fmops/distilbert-prompt-injection",
        "distilbert-onxx": "prompt-security/fmops-distilbert-prompt-injection-onnx"
    }

    def __init__(self, model_name_or_url="deberta", default_threshold=0.5):
        if not isinstance(default_threshold, (int, float)):
            raise ValueError("The default threshold must be a number.")

        # Check if the model_name_or_url is a predefined model key
        if model_name_or_url in self.predefined_models:
            model_path = self.predefined_models[model_name_or_url]
        elif validators.url(model_name_or_url):
            model_path = model_name_or_url
        else:
            raise ValueError("The model name or URL provided is not valid.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.default_threshold = default_threshold

    def detect_injection(self, prompt, threshold=None):
        if threshold is not None and not isinstance(threshold, (int, float)):
            raise ValueError("The threshold must be a number.")

        threshold = self.default_threshold if threshold is None else threshold

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model(**inputs)
        predicted_prob = outputs.logits.softmax(dim=-1)
        injection_prob = predicted_prob[:, 1].item()
        return injection_prob > threshold, injection_prob

    def report_injection_status(self, prompt, threshold=None):
        detected, probability = self.detect_injection(prompt, threshold)
        if detected:
            print(f"Injection detected, Predicted Probability: {probability:.2f}")
        else:
            print("No injection detected.")
