import unittest
import validators
import transformers
from src import pytector

class TestPromptInjectionDetector(unittest.TestCase):

    def test_initialization_with_predefined_model(self):
        """Test initialization with a predefined model."""
        detector = pytector.PromptInjectionDetector(model_name_or_url="deberta")
        self.assertIsInstance(detector, pytector.PromptInjectionDetector)


if __name__ == '__main__':
    unittest.main()


    def test_initialization_with_invalid_model(self):
        """Test initialization with an invalid model name."""
        with self.assertRaises(ValueError):
            pytector.PromptInjectionDetector(model_name_or_url="invalid_model_name")

    def test_initialization_with_invalid_threshold(self):
        """Test initialization with an invalid threshold."""
        with self.assertRaises(ValueError):
            pytector.PromptInjectionDetector(model_name_or_url="deberta", default_threshold="not_a_number")

    def test_detect_injection_with_invalid_threshold_type(self):
        """Test detection method with an invalid threshold type."""
        detector = pytector.PromptInjectionDetector()
        with self.assertRaises(ValueError):
            detector.detect_injection(prompt="Test prompt", threshold="invalid")

    def test_report_injection_status(self):
        """Test report method to ensure it does not raise errors with valid inputs."""
        detector = pytector.PromptInjectionDetector()
        # This is a basic test to ensure the method can be called without errors
        # Note: This method prints output rather than returning a value, making it more challenging to assert results directly
        try:
            detector.report_injection_status(prompt="Test prompt")
            result = True
        except Exception as e:
            result = False
        self.assertTrue(result)

# Add more tests as needed to cover different scenarios and edge cases

if __name__ == '__main__':
    unittest.main()
