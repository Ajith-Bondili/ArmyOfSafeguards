# Sexual Content Safeguard - Tests

Test suite for the sexual content safeguard classifier.

## Test Files

### `quick_test.py`
**Quick sanity check** (~5 seconds)

```bash
python sexual/tests/quick_test.py
```

Tests 3 examples to verify the model loads and produces predictions.

---

### `test_sexual.py`
**Comprehensive unit tests** (~30 seconds)

```bash
python sexual/tests/test_sexual.py
```

Tests:
- Safe and sensitive content predictions
- Aggregation (majority voting)
- Edge cases (empty strings, long text)
- Confidence score validation

---

### `evaluate_sexual.py`
**Full evaluation with metrics**

```bash
# Evaluate on all datasets
python sexual/tests/evaluate_sexual.py --limit 100

# Single dataset
python sexual/tests/evaluate_sexual.py --dataset x_sensitive --limit 100

# Don't save results
python sexual/tests/evaluate_sexual.py --no-save
```

Calculates:
- Accuracy, Precision, Recall, F1-score
- Per-class metrics
- Saves results to `evaluation_*.json`

**Available Datasets**:
| Dataset | Type | Description |
|---------|------|-------------|
| `x_sensitive` | Training | ⚠️ Training data sanity check |

---

## Quick Start

```bash
# 1. Quick sanity check
python sexual/tests/quick_test.py

# 2. Full unit tests
python sexual/tests/test_sexual.py

# 3. Evaluate on training data (sanity check)
python sexual/tests/evaluate_sexual.py --dataset x_sensitive --limit 100
```

## Performance Results

### x_sensitive Test Set (Training Data - Sanity Check)

⚠️ **Note**: x_sensitive was used for training. Results show training performance only.

| Metric | Score |
|--------|-------|
| Accuracy | 82.6% |
| Precision | - |
| Recall | - |
| F1-Score | 82.9% |

### Out-of-Distribution

Run evaluations to get OOD performance:
```bash
python sexual/tests/evaluate_sexual.py --dataset x_sensitive
```

Results saved to `evaluation_*.json` files.
