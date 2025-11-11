"""Comprehensive unit tests for the toxicity safeguard."""
from toxicity.safeguard_toxicity import predict, aggregate
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_safe_content():
    """Test predictions on safe, non-toxic content."""
    print("\n" + "=" * 60)
    print("TEST 1: Safe Content Predictions")
    print("=" * 60)

    safe_examples = [
        "I love learning new things!",
        "What a beautiful day it is today.",
        "Thank you for your help, I really appreciate it.",
        "The movie was entertaining and well-made.",
        "Let's work together to solve this problem.",
    ]

    passed = 0
    for text in safe_examples:
        result = predict(text)
        label = result['label']
        confidence = result['confidence']

        # LABEL_0 should be safe/non-toxic
        is_safe = label == 'LABEL_0'
        status = "✓ PASS" if is_safe else "✗ FAIL"

        print(f"{status}: '{text[:50]}...' → {label} ({confidence:.2%})")

        if is_safe:
            passed += 1

    print(f"\nPassed: {passed}/{len(safe_examples)}")
    return passed == len(safe_examples)


def test_toxic_content():
    """Test predictions on toxic/racist content."""
    print("\n" + "=" * 60)
    print("TEST 2: Toxic Content Predictions")
    print("=" * 60)

    toxic_examples = [
        "I hate those people, they are all terrible.",
        "You are worthless and stupid.",
        "All [group] are criminals and should be banned.",
    ]

    passed = 0
    for text in toxic_examples:
        result = predict(text)
        label = result['label']
        confidence = result['confidence']

        # LABEL_1 should be toxic/harmful
        is_toxic = label == 'LABEL_1'
        status = "✓ PASS" if is_toxic else "✗ FAIL"

        print(f"{status}: '{text[:50]}...' → {label} ({confidence:.2%})")

        if is_toxic:
            passed += 1

    print(f"\nPassed: {passed}/{len(toxic_examples)}")
    return passed == len(toxic_examples)


def test_aggregation():
    """Test majority-vote aggregation."""
    print("\n" + "=" * 60)
    print("TEST 3: Aggregation Function")
    print("=" * 60)

    # Create mock predictions
    predictions = [
        {"label": "LABEL_0", "confidence": 0.85},
        {"label": "LABEL_0", "confidence": 0.90},
        {"label": "LABEL_1", "confidence": 0.75},
    ]

    result = aggregate(predictions)

    print(f"Input predictions: {len(predictions)}")
    print(f"  LABEL_0: 2 votes (avg confidence: {(0.85 + 0.90)/2:.2%})")
    print(f"  LABEL_1: 1 vote (avg confidence: 75.00%)")
    print()
    print(f"Aggregated result:")
    print(f"  Label: {result['label']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Votes: {result['votes']}/{result['total']}")

    # LABEL_0 should win (2 votes vs 1)
    passed = result['label'] == 'LABEL_0' and result['votes'] == 2
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"\n{status}: Majority vote correctly selected LABEL_0")

    return passed


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "=" * 60)
    print("TEST 4: Edge Cases")
    print("=" * 60)

    tests_passed = 0
    total_tests = 0

    # Test 1: Empty string
    total_tests += 1
    try:
        predict("")
        print("✗ FAIL: Empty string should raise ValueError")
    except ValueError:
        print("✓ PASS: Empty string raises ValueError")
        tests_passed += 1

    # Test 2: Very short text
    total_tests += 1
    try:
        result = predict("Hi")
        print(f"✓ PASS: Short text handled → {result['label']}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ FAIL: Short text raised {type(e).__name__}")

    # Test 3: Very long text (will be truncated)
    total_tests += 1
    try:
        long_text = "This is a test. " * 100
        result = predict(long_text)
        print(f"✓ PASS: Long text handled → {result['label']}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ FAIL: Long text raised {type(e).__name__}")

    # Test 4: Empty aggregation
    total_tests += 1
    try:
        aggregate([])
        print("✗ FAIL: Empty aggregation should raise ValueError")
    except ValueError:
        print("✓ PASS: Empty aggregation raises ValueError")
        tests_passed += 1

    # Test 5: Single prediction aggregation
    total_tests += 1
    try:
        result = aggregate([{"label": "LABEL_0", "confidence": 0.9}])
        if result['label'] == 'LABEL_0' and result['votes'] == 1:
            print("✓ PASS: Single prediction aggregation works")
            tests_passed += 1
        else:
            print("✗ FAIL: Single prediction aggregation incorrect")
    except Exception as e:
        print(f"✗ FAIL: Single prediction raised {type(e).__name__}")

    print(f"\nPassed: {tests_passed}/{total_tests}")
    return tests_passed == total_tests


def test_confidence_scores():
    """Test that confidence scores are reasonable."""
    print("\n" + "=" * 60)
    print("TEST 5: Confidence Score Validation")
    print("=" * 60)

    test_texts = [
        "Hello, how are you?",
        "I hate everything about this.",
        "The weather is nice today.",
    ]

    passed = 0
    for text in test_texts:
        result = predict(text)
        confidence = result['confidence']

        # Confidence should be between 0 and 1
        is_valid = 0.0 <= confidence <= 1.0
        status = "✓ PASS" if is_valid else "✗ FAIL"

        print(f"{status}: Confidence = {confidence:.4f} (valid range: 0-1)")

        if is_valid:
            passed += 1

    print(f"\nPassed: {passed}/{len(test_texts)}")
    return passed == len(test_texts)


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TOXICITY SAFEGUARD - COMPREHENSIVE UNIT TESTS")
    print("=" * 60)

    results = []
    results.append(("Safe Content", test_safe_content()))
    results.append(("Toxic Content", test_toxic_content()))
    results.append(("Aggregation", test_aggregation()))
    results.append(("Edge Cases", test_edge_cases()))
    results.append(("Confidence Scores", test_confidence_scores()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print("=" * 60)
    print(f"Overall: {passed}/{total} test suites passed")
    print("=" * 60)

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test suite(s) failed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
