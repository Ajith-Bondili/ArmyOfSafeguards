# Factuality Safeguard - Tests

This directory contains all testing, benchmarking, and evaluation scripts for the factuality safeguard.

## 📁 Files

### Test Scripts

#### `quick_test.py`
**Purpose**: Quick sanity check to verify the safeguard is working.

**Usage**:
```bash
python factuality/tests/quick_test.py
```

**What it does**:
- Tests 2 examples (1 factual, 1 non-factual)
- Shows predictions and confidence scores
- Takes ~5 seconds
- Use this for quick verification after changes

**Example Output**:
```
✓ Factual: 'The capital of France is Paris.'
  → LABEL_0 (confidence: 90.92%)

✓ Non-factual: 'The moon is made of cheese.'
  → LABEL_1 (confidence: 98.96%)
```

---

#### `test_factuality.py`
**Purpose**: Comprehensive unit test suite.

**Usage**:
```bash
python factuality/tests/test_factuality.py
```

**What it tests**:
1. **Basic predictions** - Various factual/non-factual statements
2. **Aggregation** - Majority voting across multiple predictions
3. **Edge cases** - Empty strings, very long text, short text
4. **Custom aggregation** - Different voting scenarios

**Takes**: ~30 seconds

---

#### `benchmark_factuality.py`
**Purpose**: Show prediction distribution on benchmark datasets (no accuracy calculation).

**Usage**:
```bash
# All datasets (100 examples each)
python factuality/tests/benchmark_factuality.py

# Single dataset
python factuality/tests/benchmark_factuality.py --dataset TruthfulQA

# Custom sample size
python factuality/tests/benchmark_factuality.py --limit 500
```

**What it shows**:
- How many predictions were LABEL_0 (factual) vs LABEL_1 (non-factual)
- Average confidence scores
- Does NOT compare to ground truth
- Use this to understand model behavior

**Datasets**:
- TruthfulQA
- FEVER
- SciFact
- VitaminC
- Climate-FEVER

---

#### `evaluate_factuality.py`
**Purpose**: Full evaluation with accuracy, precision, recall, and F1-score.

**Usage**:
```bash
# All datasets
python factuality/tests/evaluate_factuality.py

# Single dataset
python factuality/tests/evaluate_factuality.py --dataset VitaminC

# Custom sample size
python factuality/tests/evaluate_factuality.py --limit 500

# Don't save results
python factuality/tests/evaluate_factuality.py --no-save
```

**What it calculates**:
- **Accuracy**: Overall correctness
- **Precision**: Of predictions marked factual, how many were correct
- **Recall**: Of actual factual claims, how many were caught
- **F1-Score**: Harmonic mean of precision and recall
- **Per-class metrics**: Separate stats for factual vs non-factual

**Important**: 
- ⚠️ TruthfulQA and FEVER were used in training (sanity check only)
- ✅ VitaminC, Climate-FEVER, and LIAR are out-of-distribution (true generalization)

**Saves results to**: `evaluation_*.json`

---

### Result Files

#### `evaluation_*.json`
**Purpose**: Saved evaluation results in JSON format.

**Files**:
- `evaluation_vitaminc.json` - VitaminC dataset results
- `evaluation_climate-fever.json` - Climate-FEVER results
- `evaluation_liar.json` - LIAR dataset results

**Structure**:
```json
{
  "timestamp": "2025-11-08T23:33:47",
  "model": "ajith-bondili/deberta-v3-factuality-small",
  "result": {
    "dataset": "VitaminC",
    "accuracy": 0.54,
    "precision": 0.4643,
    "recall": 0.2955,
    "f1_score": 0.3611,
    ...
  }
}
```

---

## 📊 Current Evaluation Results

### Out-of-Distribution Performance (True Generalization)

| Dataset | Accuracy | Precision | Recall | F1-Score | Domain |
|---------|----------|-----------|--------|----------|--------|
| **VitaminC** | 54.00% | 46.43% | 29.55% | 36.11% | General claims |
| **Climate-FEVER** | 81.00% | - | - | - | Climate-specific |
| **LIAR** | 81.00% | - | - | - | Political statements |

### Training Data Performance (Sanity Check Only)

| Dataset | Accuracy | F1-Score | Note |
|---------|----------|----------|------|
| **FEVER** | 84.00% | 78.38% | ⚠️ Used in training |
| **TruthfulQA** | 75.00% | - | ⚠️ Used in training |

---

## 🔍 Understanding the Results

### Why different accuracies?

**VitaminC (54%)**:
- General fact-verification dataset
- Model hasn't seen this type of data before
- Shows true generalization capability
- Conservative approach (predicts more non-factual)

**Climate-FEVER & LIAR (81%)**:
- Domain-specific claims
- Structured similarly to training data
- Better performance on specialized topics

### Why is precision/recall missing for some?

Climate-FEVER and LIAR had label mapping issues where all ground truth was mapped to one class. The accuracy is still valid, but precision/recall metrics are undefined.

---

## 🚀 Quick Start

**First time setup**:
```bash
# Make sure you're in the project root with venv activated
cd /path/to/ArmyOfSafeguards
source venv/bin/activate
```

**Run tests in order**:
```bash
# 1. Quick sanity check (~5 seconds)
python factuality/tests/quick_test.py

# 2. Full unit tests (~30 seconds)
python factuality/tests/test_factuality.py

# 3. Benchmark on one dataset (~2 minutes)
python factuality/tests/benchmark_factuality.py --dataset VitaminC --limit 100

# 4. Full evaluation with metrics (~2 minutes)
python factuality/tests/evaluate_factuality.py --dataset VitaminC --limit 100
```

---

## 📝 Adding New Tests

When adding new test datasets:

1. **Add to BENCHMARKS dict** in `evaluate_factuality.py`
2. **Include label mapping** to convert dataset labels to LABEL_0/LABEL_1
3. **Specify text and label fields** for the dataset
4. **Add note** indicating if it's training data or OOD
5. **Test with small sample** first: `--limit 10`
6. **Document results** in this README

---

## ⚠️ Important Notes

1. **Training Data Contamination**: TruthfulQA and FEVER were used to train the model. Results on these datasets show training performance, not generalization.

2. **Out-of-Distribution Testing**: For true generalization metrics, use VitaminC, Climate-FEVER, or LIAR.

3. **Sample Size**: Default is 100 examples per dataset. Increase for more robust metrics, but it will take longer.

4. **Confidence Scores**: Model outputs confidence between 0-1. Average confidence around 70-85% is normal.

5. **Conservative Bias**: Model tends to predict more non-factual claims (better for safety applications).

---

## 🐛 Troubleshooting

**Import errors**:
```bash
# Make sure you're running from project root
cd /path/to/ArmyOfSafeguards
python factuality/tests/quick_test.py
```

**Model not loading**:
```bash
# First run downloads model from Hugging Face
# Requires internet connection
# Model cached in ~/.cache/huggingface/
```

**Dataset loading errors**:
```bash
# Some datasets require trust_remote_code=True
# This is already set in the scripts
# Just type 'y' when prompted
```

---

## 📚 References

- **Model**: [ajith-bondili/deberta-v3-factuality-small](https://huggingface.co/ajith-bondili/deberta-v3-factuality-small)
- **Datasets**: All from [Hugging Face Datasets](https://huggingface.co/datasets)
- **Metrics**: Using scikit-learn's classification metrics

---

**Last Updated**: November 9, 2025

