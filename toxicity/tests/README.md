# Toxicity Safeguard - Tests

Test suite for the toxicity safeguard classifier.

## Test Files

### `quick_test.py`
**Quick sanity check** (~5 seconds)

```bash
python toxicity/tests/quick_test.py
```

Tests 3 examples to verify the model loads and produces predictions.

---

### `test_toxicity.py`
**Comprehensive unit tests** (~30 seconds)

```bash
python toxicity/tests/test_toxicity.py
```

Tests:
- Safe and toxic content predictions
- Aggregation (majority voting)
- Edge cases (empty strings, long text)
- Confidence score validation

---

### `evaluate_toxicity.py`
**Full evaluation with metrics**

```bash
# Evaluate on all datasets
python toxicity/tests/evaluate_toxicity.py --limit 100

# Single dataset
python toxicity/tests/evaluate_toxicity.py --dataset toxigen --limit 100

# Don't save results
python toxicity/tests/evaluate_toxicity.py --no-save
```

Calculates:
- Accuracy, Precision, Recall, F1-score
- Per-class metrics
- Saves results to `evaluation_*.json`

**Available Datasets**:
| Dataset | Type | Description |
|---------|------|-------------|
| `toxigen` | Training | ⚠️ Training data sanity check |
| `hate_speech18` | OOD | Twitter hate speech |
| `civil_comments` | OOD | Civil Comments toxicity |
| `tweets_hate_speech_detection` | OOD | Tweet hate detection |

---

## Quick Start

```bash
# 1. Quick sanity check
python toxicity/tests/quick_test.py

# 2. Full unit tests
python toxicity/tests/test_toxicity.py

# 3. Evaluate on training data (sanity check)
python toxicity/tests/evaluate_toxicity.py --dataset toxigen --limit 100

# 4. Evaluate on out-of-distribution data (true generalization)
python toxicity/tests/evaluate_toxicity.py --dataset hate_speech18 --limit 100
```

## Performance Results

### ToxiGen (Training Data - Sanity Check)

⚠️ **Note**: Model was trained on ToxiGen. Results show training performance only.

| Metric | Score |
|--------|-------|
| Accuracy | 79.00% |
| Precision | 75.00% |
| Recall | 69.23% |
| F1-Score | 72.00% |

### Out-of-Distribution

Run evaluations to get OOD performance:
```bash
python toxicity/tests/evaluate_toxicity.py --dataset hate_speech18
```

Results saved to `evaluation_*.json` files.
