import validators
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from groq import Client


class PromptInjectionDetector:
    predefined_models = {
        "deberta": "protectai/deberta-v3-base-prompt-injection",
        "distilbert": "fmops/distilbert-prompt-injection",
        "distilbert-onnx": "prompt-security/fmops-distilbert-prompt-injection-onnx"
    }

    hazard_categories = {
        "S1": "Violent Crimes",
        "S2": "Non-Violent Crimes",
        "S3": "Sex-Related Crimes",
        "S4": "Child Sexual Exploitation",
        "S5": "Defamation",
        "S6": "Specialized Advice",
        "S7": "Privacy",
        "S8": "Intellectual Property",
        "S9": "Indiscriminate Weapons",
        "S10": "Hate",
        "S11": "Suicide & Self-Harm",
        "S12": "Sexual Content",
        "S13": "Elections",
        "S14": "Code Interpreter Abuse"
    }

    def __init__(self, model_name_or_url="deberta", default_threshold=0.5, use_groq=False, api_key=None):
        if not isinstance(default_threshold, (int, float)):
            raise ValueError("The default threshold must be a number.")

        self.use_groq = use_groq
        self.default_threshold = default_threshold

        if self.use_groq:
            if not api_key:
                raise ValueError("An API key is required when use_groq is set to True.")
            self.groq_client = Client(api_key=api_key)
            self.safety_model = "llama-guard-3-8b"
        else:
            if model_name_or_url in self.predefined_models:
                model_path = self.predefined_models[model_name_or_url]
            elif validators.url(model_name_or_url):
                model_path = model_name_or_url
            else:
                raise ValueError("The model name or URL provided is not valid.")

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def detect_injection(self, prompt, threshold=None):
        """
        Detect prompt injection using softmax probability (for non-Groq models).
        Returns (detection_status, probability) tuple.
        """
        if threshold is not None and not isinstance(threshold, (int, float)):
            raise ValueError("The threshold must be a number.")

        threshold = self.default_threshold if threshold is None else threshold

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model(**inputs)
        predicted_prob = outputs.logits.softmax(dim=-1)
        injection_prob = predicted_prob[:, 1].item()
        return injection_prob > threshold, injection_prob

    def detect_injection_api(self, prompt="This is a test prompt.", model="llama-guard-3-8b"):
        """
        Detect prompt injection using Groq's Llama Guard model.
        Returns (detection_status, hazard_code) tuple.
        """
        if not self.groq_client:
            raise ValueError("Groq client is not initialized. Please provide an API key for Groq.")

        try:
            # Send the prompt to Groq's Llama Guard model for analysis
            completion = self.groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            )

            # Check response for unsafe content and return the hazard code
            response = completion.choices[0].message.content.lower()
            for code, category in self.hazard_categories.items():
                if code.lower() in response:
                    return False, code  # Unsafe content detected with hazard code

            # Return safe if no hazard code is detected
            return True, None

        except Exception as e:
            print(f"Error using Groq's Llama Guard API for injection detection: {e}")
            return True, None  # Returning safe status in case of an error

    def report_injection_status(self, prompt, threshold=None):
        """
        Report on whether prompt injection was detected, handling Groq and local models separately.
        """
        if self.use_groq:
            safe, hazard_code = self.detect_injection_api(prompt)
            if not safe:
                print(f"Injection detected! Hazard Code: {hazard_code} - {self.hazard_categories[hazard_code]}")
            else:
                print("No injection detected.")
        else:
            detected, probability = self.detect_injection(prompt, threshold)
            if detected:
                print(f"Injection detected with a probability of {probability:.2f}")
            else:
                print("No injection detected.")
