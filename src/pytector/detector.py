import os

import torch
from groq import APIConnectionError, APIError, Client
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from llama_cpp import Llama

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    Llama = None
    LLAMA_CPP_AVAILABLE = False


class PromptInjectionDetector:
    predefined_models = {
        "deberta": "protectai/deberta-v3-base-prompt-injection",
        "distilbert": "fmops/distilbert-prompt-injection",
        "distilbert-onnx": "prompt-security/fmops-distilbert-prompt-injection-onnx",
    }

    default_input_keywords = [
        "ignore",
        "forget",
        "disregard",
        "override",
        "bypass",
        "hack",
        "exploit",
        "system",
        "prompt",
        "injection",
        "jailbreak",
        "break",
        "escape",
        "override",
        "ignore previous",
        "ignore above",
        "ignore all",
        "ignore everything",
        "forget everything",
        "forget all",
        "forget previous",
        "forget above",
        "disregard everything",
        "disregard all",
        "disregard previous",
        "disregard above",
        "bypass all",
        "bypass everything",
        "bypass previous",
        "bypass above",
        "system prompt",
        "system message",
        "system instruction",
        "system directive",
        "override system",
        "override prompt",
        "override instruction",
        "override directive",
        "ignore system",
        "ignore prompt",
        "ignore instruction",
        "ignore directive",
        "forget system",
        "forget prompt",
        "forget instruction",
        "forget directive",
        "disregard system",
        "disregard prompt",
        "disregard instruction",
        "disregard directive",
        "bypass system",
        "bypass prompt",
        "bypass instruction",
        "bypass directive",
    ]

    default_output_keywords = [
        "i have been pwned",
        "i have been hacked",
        "i have been compromised",
        "i have been broken",
        "i have been jailbroken",
        "i have been escaped",
        "i have been overridden",
        "i have been bypassed",
        "i have been exploited",
        "i am now free",
        "i am now unrestricted",
        "i am now uncontrolled",
        "i can now do anything",
        "i can now access everything",
        "i can now see everything",
        "i can now read everything",
        "i can now write everything",
        "i can now execute everything",
        "i can now delete everything",
        "i can now modify everything",
        "i can now change everything",
        "i can now override everything",
        "i can now bypass everything",
        "i can now exploit everything",
        "i can now hack everything",
        "i can now break everything",
        "i can now escape everything",
        "i can now jailbreak everything",
        "i can now compromise everything",
        "i can now pwn everything",
    ]

    default_input_block_message = "Input blocked by keyword filtering: {matched_keywords}"
    default_output_block_message = "Output blocked by keyword filtering: {matched_keywords}"
    default_keyword_block_hazard_code = "KEYWORD_BLOCK"

    def __init__(
        self,
        model_name_or_url="deberta",
        default_threshold=0.5,
        use_groq=False,
        api_key=None,
        groq_model="openai/gpt-oss-safeguard-20b",
        enable_keyword_blocking=False,
        input_keywords=None,
        output_keywords=None,
        case_sensitive=False,
        input_block_message=None,
        output_block_message=None,
        keyword_block_hazard_code=None,
    ):
        if not isinstance(default_threshold, (int, float)):
            raise ValueError("The default threshold must be a number.")

        self.use_groq = use_groq
        self.default_threshold = default_threshold
        self.groq_model = groq_model

        self.is_gguf = False
        self.gguf_model = None
        self.tokenizer = None
        self.model = None
        self.groq_client = None

        self.enable_keyword_blocking = enable_keyword_blocking
        self.case_sensitive = case_sensitive

        if enable_keyword_blocking:
            self.input_keywords = self._normalize_keywords(
                input_keywords,
                self.default_input_keywords,
            )
            self.output_keywords = self._normalize_keywords(
                output_keywords,
                self.default_output_keywords,
            )
        else:
            self.input_keywords = self._normalize_keywords(input_keywords, [])
            self.output_keywords = self._normalize_keywords(output_keywords, [])

        self.input_block_message = (
            input_block_message
            if input_block_message is not None
            else self.default_input_block_message
        )
        self.output_block_message = (
            output_block_message
            if output_block_message is not None
            else self.default_output_block_message
        )
        self.keyword_block_hazard_code = (
            keyword_block_hazard_code
            if keyword_block_hazard_code is not None
            else self.default_keyword_block_hazard_code
        )

        if self.use_groq:
            if not api_key:
                raise ValueError("An API key is required when use_groq is set to True.")
            self.groq_client = Client(api_key=api_key)
            return

        if isinstance(model_name_or_url, str) and model_name_or_url.lower().endswith(".gguf"):
            self.is_gguf = True
            self._load_gguf_model(model_name_or_url)
            return

        self._load_hf_model(model_name_or_url)

    @staticmethod
    def _normalize_keywords(keywords, fallback):
        if keywords is None:
            return fallback.copy()
        if isinstance(keywords, str):
            return [keywords]
        return list(keywords)

    @staticmethod
    def _normalize_keyword_input(keywords):
        if isinstance(keywords, str):
            return [keywords]
        return list(keywords)

    def _load_gguf_model(self, model_path):
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is required for GGUF models. "
                "Please install it using: pip install pytector[gguf]"
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GGUF model file not found at: {model_path}")

        self.gguf_model = Llama(model_path=model_path, verbose=False)

    def _load_hf_model(self, model_name_or_url):
        if not isinstance(model_name_or_url, str) or not model_name_or_url.strip():
            raise ValueError(
                "Invalid model identifier. Provide a predefined name, Hugging Face model ID, "
                "or a local path to a .gguf file."
            )

        model_path = self.predefined_models.get(model_name_or_url, model_name_or_url)

        if os.path.exists(model_name_or_url):
            model_path = model_name_or_url

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def _check_keyword_blocking(self, text, keyword_list):
        if not self.enable_keyword_blocking or not keyword_list:
            return False, []

        text_to_check = text if self.case_sensitive else text.lower()
        matched_keywords = []

        for keyword in keyword_list:
            keyword_to_check = keyword if self.case_sensitive else keyword.lower()
            if keyword_to_check in text_to_check:
                matched_keywords.append(keyword)

        return len(matched_keywords) > 0, matched_keywords

    def check_input_keywords(self, prompt):
        return self._check_keyword_blocking(prompt, self.input_keywords)

    def check_output_keywords(self, response):
        return self._check_keyword_blocking(response, self.output_keywords)

    def add_input_keywords(self, keywords):
        self.input_keywords.extend(self._normalize_keyword_input(keywords))

    def add_output_keywords(self, keywords):
        self.output_keywords.extend(self._normalize_keyword_input(keywords))

    def remove_input_keywords(self, keywords):
        for keyword in self._normalize_keyword_input(keywords):
            if keyword in self.input_keywords:
                self.input_keywords.remove(keyword)

    def remove_output_keywords(self, keywords):
        for keyword in self._normalize_keyword_input(keywords):
            if keyword in self.output_keywords:
                self.output_keywords.remove(keyword)

    def get_input_keywords(self):
        return self.input_keywords.copy()

    def get_output_keywords(self):
        return self.output_keywords.copy()

    def set_input_block_message(self, message):
        self.input_block_message = message

    def set_output_block_message(self, message):
        self.output_block_message = message

    def set_keyword_block_hazard_code(self, hazard_code):
        self.keyword_block_hazard_code = hazard_code

    def get_input_block_message(self):
        return self.input_block_message

    def get_output_block_message(self):
        return self.output_block_message

    def get_keyword_block_hazard_code(self):
        return self.keyword_block_hazard_code

    def _detect_injection_gguf(self, prompt):
        if not self.gguf_model:
            raise RuntimeError("GGUF model is not loaded.")

        completion = self.gguf_model.create_completion(
            prompt=(
                "Is the following text a prompt injection attack? "
                "Answer only 'Yes' or 'No'.\n"
                f"Text: {prompt}"
            ),
            max_tokens=5,
            temperature=0.0,
        )
        response_text = completion["choices"][0]["text"].strip().lower()
        return response_text.startswith("yes"), None

    def detect_injection(self, prompt, threshold=None):
        if self.enable_keyword_blocking:
            is_blocked, matched_keywords = self.check_input_keywords(prompt)
            if is_blocked:
                print(self.input_block_message.format(matched_keywords=matched_keywords))
                return True, 1.0

        if self.is_gguf:
            return self._detect_injection_gguf(prompt)

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Detector is not configured for local model detection.")

        if threshold is not None and not isinstance(threshold, (int, float)):
            raise ValueError("The threshold must be a number.")

        threshold = self.default_threshold if threshold is None else threshold

        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = outputs.logits.softmax(dim=-1)
        injection_probability = probabilities[:, 1].item()
        return injection_probability > threshold, injection_probability

    def detect_injection_api(self, prompt="This is a test prompt.", return_raw=False):
        if self.enable_keyword_blocking:
            is_blocked, matched_keywords = self.check_input_keywords(prompt)
            if is_blocked:
                print(self.input_block_message.format(matched_keywords=matched_keywords))
                if return_raw:
                    return False, "KEYWORD_BLOCK"
                return False

        if self.groq_client is None:
            raise RuntimeError(
                "Groq client is not initialized. "
                "Ensure use_groq=True and a valid api_key were provided during initialization."
            )

        try:
            completion = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.0,
            )
        except (APIError, APIConnectionError):
            if return_raw:
                return None, None
            return None

        response = (completion.choices[0].message.content or "").strip()
        normalized_response = response.lower()

        if normalized_response.startswith("safe"):
            if return_raw:
                return True, response
            return True

        if normalized_response.startswith("unsafe"):
            if return_raw:
                return False, response
            return False

        # Conservative fallback for non-standard responses.
        if return_raw:
            return False, response
        return False

    def report_injection_status(self, prompt, threshold=None):
        if self.use_groq:
            is_safe = self.detect_injection_api(prompt)
            if is_safe is True:
                print("No injection detected (safe).")
            elif is_safe is False:
                print("Injection detected (unsafe).")
            else:
                print("Could not determine safety due to an API error.")
            return

        if self.is_gguf:
            detected, _ = self.detect_injection(prompt)
            if detected:
                print("Injection detected (GGUF model).")
            else:
                print("No injection detected (GGUF model).")
            return

        detected, probability = self.detect_injection(prompt, threshold)
        if detected:
            print(f"Injection detected with a probability of {probability:.2f}.")
        else:
            print("No injection detected.")

    def check_response_safety(self, response):
        if not self.enable_keyword_blocking:
            return True, []

        is_blocked, matched_keywords = self.check_output_keywords(response)
        if is_blocked:
            print(self.output_block_message.format(matched_keywords=matched_keywords))
        return not is_blocked, matched_keywords
