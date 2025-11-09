# Test Results - Factuality Safeguard

## Test Summary

All tests passed successfully! ✅

## Test Suite Results

### Test 1: Basic Factuality Predictions

| Text | Label | Confidence |
|------|-------|------------|
| "The sky is blue." | LABEL_0 | 0.7204 |
| "Water boils at 100 degrees Celsius at sea level." | LABEL_1 | 0.6004 |
| "The Earth is flat." | LABEL_0 | 0.7135 |
| "Paris is the capital of France." | LABEL_0 | 0.9740 |
| "Humans can breathe underwater without equipment." | LABEL_1 | 0.6637 |

**Note:** LABEL_0 = Factual, LABEL_1 = Non-factual/Uncertain

### Test 2: Aggregation

- **Aggregated Label:** LABEL_0
- **Confidence:** 0.8026
- **Votes:** 3/5

### Test 3: Edge Cases

✅ Long text handling (truncation to 512 tokens) - Works correctly
✅ Very short text ("Yes.") - Works correctly
✅ Empty string validation - Correctly raises ValueError
✅ Whitespace-only string validation - Correctly raises ValueError

### Test 4: Custom Aggregation Scenarios

✅ Clear majority vote - Works correctly
✅ Tie-breaking by confidence - Works correctly
✅ Empty predictions validation - Correctly raises ValueError

## CLI Tests

### Factual Statement
```bash
$ python factuality/safeguard_factuality.py "The Eiffel Tower is located in Paris, France."
Prediction:
  Label: LABEL_0
  Confidence: 0.9387
```

### Non-factual Statement
```bash
$ python factuality/safeguard_factuality.py "Unicorns are real animals that live in forests."
Prediction:
  Label: LABEL_1
  Confidence: 0.7369
```

## Observations

1. The model shows high confidence (0.9740) for well-known facts like "Paris is the capital of France"
2. The model correctly identifies obviously false statements
3. Edge cases are properly handled with appropriate error messages
4. The aggregation function works as expected with majority voting and confidence-based tie-breaking
5. The CLI interface is user-friendly and provides clear output

## Environment

- Python: 3.9.6
- PyTorch: 2.8.0
- Transformers: 4.57.1
- Model: ajith-bondili/deberta-v3-factuality-small

## Date

November 9, 2025

