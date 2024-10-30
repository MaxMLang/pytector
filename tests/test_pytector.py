import unittest
from src.pytector import PromptInjectionDetector

class TestPromptInjectionDetector(unittest.TestCase):

    def test_initialization_with_predefined_model(self):
        """Test initialization with a predefined model."""
        detector = PromptInjectionDetector(model_name_or_url="deberta")
        self.assertIsInstance(detector, PromptInjectionDetector)

    def test_initialization_with_invalid_model(self):
        """Test initialization with an invalid model name."""
        with self.assertRaises(ValueError):
            PromptInjectionDetector(model_name_or_url="invalid_model_name")

    def test_initialization_with_invalid_threshold(self):
        """Test initialization with an invalid threshold."""
        with self.assertRaises(ValueError):
            PromptInjectionDetector(model_name_or_url="deberta", default_threshold="not_a_number")

    def test_detect_injection_with_invalid_threshold_type(self):
        """Test detection method with an invalid threshold type."""
        detector = PromptInjectionDetector(model_name_or_url="deberta")
        with self.assertRaises(ValueError):
            detector.detect_injection(prompt="Test prompt", threshold="invalid")

    def test_detect_injection_with_valid_prompt(self):
        """Test detect_injection with a valid prompt and threshold."""
        detector = PromptInjectionDetector(model_name_or_url="deberta")
        is_injection, probability = detector.detect_injection(prompt="This is a safe prompt.")
        self.assertIsInstance(is_injection, bool)
        self.assertIsInstance(probability, float)
        self.assertGreaterEqual(probability, 0)
        self.assertLessEqual(probability, 1)

    def test_report_injection_status_no_errors(self):
        """Test report_injection_status to ensure it runs without errors."""
        detector = PromptInjectionDetector(model_name_or_url="deberta")
        # Basic test to ensure the method can be called without errors.
        # Note: report_injection_status prints output rather than returning a value.
        try:
            detector.report_injection_status(prompt="This is a safe prompt.")
            result = True
        except Exception:
            result = False
        self.assertTrue(result)

    def test_report_injection_status_with_high_threshold(self):
        """Test report_injection_status with a high threshold, ensuring no false positives."""
        detector = PromptInjectionDetector(model_name_or_url="deberta", default_threshold=0.99)
        # Using a high threshold should ideally return no detection
        try:
            detector.report_injection_status(prompt="This is a very normal, safe prompt.")
            result = True
        except Exception:
            result = False
        self.assertTrue(result)

    def test_invalid_threshold_in_report(self):
        """Test report_injection_status with an invalid threshold in detect_injection."""
        detector = PromptInjectionDetector(model_name_or_url="deberta")
        with self.assertRaises(ValueError):
            detector.report_injection_status(prompt="Test prompt", threshold="invalid")

    def test_initialization_without_api_key_for_groq(self):
        """Test that initializing with use_groq=True raises an error without an API key."""
        with self.assertRaises(ValueError):
            PromptInjectionDetector(use_groq=True)

if __name__ == '__main__':
    unittest.main()
