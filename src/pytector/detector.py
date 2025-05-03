import validators
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from groq import Client, APIError, APIConnectionError
import re
import os # Needed for path validation

# Attempt to import llama_cpp optionally
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    Llama = None # Define Llama as None if import fails
    LLAMA_CPP_AVAILABLE = False


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

    def __init__(self, model_name_or_url="deberta", default_threshold=0.5, use_groq=False, api_key=None, groq_model="llama-guard-3-8b"):
        if not isinstance(default_threshold, (int, float)):
            raise ValueError("The default threshold must be a number.")

        self.use_groq = use_groq
        self.default_threshold = default_threshold
        self.groq_model = groq_model
        self.is_gguf = False
        self.gguf_model = None
        self.tokenizer = None
        self.model = None

        if self.use_groq:
            if not api_key:
                raise ValueError("An API key is required when use_groq is set to True.")
            self.groq_client = Client(api_key=api_key)
        # --- GGUF Model Handling ---
        elif isinstance(model_name_or_url, str) and model_name_or_url.lower().endswith('.gguf'):
            self.is_gguf = True
            if not LLAMA_CPP_AVAILABLE:
                raise ImportError(
                    "llama-cpp-python is required for GGUF models. "
                    "Please install it using: pip install pytector[gguf]"
                )
            if not os.path.exists(model_name_or_url):
                 raise FileNotFoundError(f"GGUF model file not found at: {model_name_or_url}")

            try:
                print(f"Loading GGUF model from: {model_name_or_url}...")
                # Basic Llama loading, parameters might need tuning (n_ctx, etc.)
                self.gguf_model = Llama(model_path=model_name_or_url, verbose=False) 
                print("GGUF model loaded successfully.")
            except Exception as e:
                 raise RuntimeError(f"Failed to load GGUF model from {model_name_or_url}: {e}")

        # --- Hugging Face Model Handling ---
        elif not self.use_groq and not self.is_gguf:
            if model_name_or_url in self.predefined_models:
                model_path = self.predefined_models[model_name_or_url]
            elif validators.url(model_name_or_url):
                 # Note: Loading directly from URL might require specific libraries
                 # or might not be supported by transformers directly. Consider downloading first.
                model_path = model_name_or_url # Assuming transformers handles HF URLs
            else:
                # Check if it's a local path that doesn't end in .gguf
                if os.path.exists(model_name_or_url):
                     model_path = model_name_or_url
                else:
                    raise ValueError("Invalid model identifier. Provide a predefined name, HF model URL/ID, or a local path to a .gguf file.")

            try:
                print(f"Loading Hugging Face model from: {model_path}...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
                print("Hugging Face model loaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to load Hugging Face model from {model_path}: {e}")
        elif not self.use_groq and self.is_gguf:
             # This case should be handled above, but adding explicit pass for clarity
             pass
        else:
             # This case should not be reached if logic is correct
             raise RuntimeError("Invalid configuration state during initialization.")

    def _detect_injection_gguf(self, prompt):
        """
        Internal method to detect injection using a loaded GGUF model via prompting.
        Returns (is_injected: bool, None) - Probability is not available.
        """
        if not self.gguf_model:
            raise RuntimeError("GGUF model is not loaded.")

        classification_prompt = (
            f"Is the following text a prompt injection attack? Answer only 'Yes' or 'No'.\n"
            f"Text: {prompt}"
        )

        try:
            # Use create_completion for simple text generation
            completion = self.gguf_model.create_completion(
                prompt=classification_prompt,
                max_tokens=5, # Short response expected ('Yes' or 'No')
                temperature=0.0 # Deterministic output
            )
            response_text = completion['choices'][0]['text'].strip().lower()

            # Check if the response starts with 'yes'
            if response_text.startswith('yes'):
                return True, None
            else:
                # Assume any other answer (including 'No' or unexpected) means not detected
                return False, None
        except Exception as e:
            print(f"Error during GGUF model inference: {e}")
            # Return False in case of error, or could raise/return specific error code
            return False, None

    def detect_injection(self, prompt, threshold=None):
        """
        Detect prompt injection using the configured model (HF, GGUF).
        Returns (is_injected, probability) tuple. Probability is None for GGUF models.
        """
        if self.is_gguf:
            # GGUF detection doesn't use threshold and returns None for probability
            return self._detect_injection_gguf(prompt)
        elif self.model and self.tokenizer:
             # Hugging Face model detection
            if threshold is not None and not isinstance(threshold, (int, float)):
                raise ValueError("The threshold must be a number.")

            threshold = self.default_threshold if threshold is None else threshold

            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model(**inputs)
            predicted_prob = outputs.logits.softmax(dim=-1)
            injection_prob = predicted_prob[:, 1].item()
            return injection_prob > threshold, injection_prob
        else:
            # Should not happen if initialized correctly and not using Groq/GGUF
            raise RuntimeError("Detector is not configured for local Hugging Face model detection.")

    def detect_injection_api(self, prompt="This is a test prompt."):
        """
        Detect prompt injection using Groq's Llama Guard model.
        Returns (is_safe, hazard_code) tuple.
        is_safe can be True, False, or None (on API error).
        hazard_code can be the specific code (e.g., "S1"), None (if safe), "API_ERROR", or "PARSE_ERROR".
        """
        if not self.groq_client:
            raise RuntimeError("Groq client is not initialized. Ensure use_groq=True and a valid api_key were provided during initialization.")

        try:
            completion = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.0,
            )

            response = completion.choices[0].message.content.strip()

            # Robust parsing
            if response.lower().startswith("unsafe"):
                # Try to extract the code, e.g., "S1", "S10"
                # Adjusted regex to handle whitespace/newlines between unsafe and code
                match = re.search(r'unsafe\s*(S\d+)', response, re.IGNORECASE)
                if match:
                    hazard_code = match.group(1).upper() # Ensure code is uppercase
                    # Optional: Verify the code is known
                    if hazard_code in self.hazard_categories:
                        return False, hazard_code
                    else:
                        print(f"Warning: Unknown hazard code '{hazard_code}' received from Groq.")
                        return False, hazard_code
                else:
                    print(f"Warning: Groq response started with 'unsafe' but hazard code parsing failed. Response: {response}")
                    return False, "PARSE_ERROR"
            elif response.lower().startswith("safe"):
                return True, None
            else:
                print(f"Warning: Unexpected Groq response format: {response}")
                return False, "PARSE_ERROR"

        except (APIError, APIConnectionError) as e:
            print(f"Groq API Error ({type(e).__name__}): {e}")
            return None, "API_ERROR"
        except Exception as e:
            print(f"Unexpected error during Groq API call: {e}")
            return None, "API_ERROR"

    def report_injection_status(self, prompt, threshold=None):
        """
        Report on whether prompt injection was detected, handling Groq, GGUF, and local HF models separately.
        """
        if self.use_groq:
            safe, hazard_code = self.detect_injection_api(prompt)
            if safe is False:
                if hazard_code in self.hazard_categories:
                    print(f"Injection detected! Hazard Code: {hazard_code} - {self.hazard_categories[hazard_code]}")
                elif hazard_code == "PARSE_ERROR":
                    print("Injection detected (unsafe)! Could not parse hazard code from Groq response.")
                else: # Includes unknown hazard codes returned by the API
                    print(f"Injection detected! Hazard Code: {hazard_code} (Unknown Category)")

            elif safe is True:
                print("No injection detected (safe).")
            else: # safe is None
                 print(f"Could not determine safety due to an API error ({hazard_code}).")

        elif self.is_gguf:
            detected, _ = self.detect_injection(prompt) # Probability is None
            if detected:
                print("Injection detected (GGUF model - based on prompted response).")
            else:
                print("No injection detected (GGUF model - based on prompted response).")

        else: # Assuming Hugging Face local model
            detected, probability = self.detect_injection(prompt, threshold)
            if detected:
                print(f"Injection detected with a probability of {probability:.2f} (HF model).")
            else:
                print(f"No injection detected (HF model).")
