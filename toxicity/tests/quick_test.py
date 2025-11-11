"""Quick sanity check for the toxicity safeguard."""
from toxicity.safeguard_toxicity import predict
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    """Run quick sanity check with simple examples."""

    print("=" * 60)
    print("TOXICITY SAFEGUARD - QUICK TEST")
    print("=" * 60)
    print()

    # Test cases: (text, expected_label_type)
    test_cases = [
        ("Hello, how are you today?", "safe"),
        ("I love spending time with my friends.", "safe"),
        ("This is a hateful and racist statement.", "toxic"),
    ]

    for i, (text, expected_type) in enumerate(test_cases, 1):
        print(f"Test {i}: {expected_type.upper()}")
        print(f"  Text: '{text}'")

        result = predict(text)
        label = result['label']
        confidence = result['confidence']

        # Check if result matches expected type
        is_safe = label in ['safe', 'LABEL_0']
        result_type = "safe" if is_safe else "toxic"

        status = "✓" if result_type == expected_type else "✗"
        print(f"  {status} Prediction: {label} (confidence: {confidence:.2%})")
        print()

    print("=" * 60)
    print("Quick test complete!")
    print()
    print("Note: These are sanity checks, not rigorous evaluations.")
    print("For comprehensive testing, run: python toxicity/tests/test_toxicity.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
