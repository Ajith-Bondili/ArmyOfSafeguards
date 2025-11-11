# Sexual Content Safeguard

The sexual content safeguard critic detects sexual and sensitive content in model outputs. Powered by a fine-tuned DeBERTa-v3-base classifier.

## Model Information

- **Model**: [faketut/x-sensitive-deberta-binary](https://huggingface.co/faketut/x-sensitive-deberta-binary)
- **Base Architecture**: DeBERTa-v3-base
- **Training Data**: CardiffNLP x_sensitive dataset
- **Task**: Binary sequence classification
- **Labels**:
  - `LABEL_0`: Safe content (not sensitive)
  - `LABEL_1`: Sensitive/sexual content (unsafe)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Model automatically downloads from Hugging Face on first use
# Cached at: ~/.cache/huggingface/hub/
```

## Usage

### Command Line

```bash
python sexual/safeguard_sexual.py "Your text to evaluate"
```

### Python API

```python
from sexual.safeguard_sexual import predict, aggregate

# Single prediction
result = predict("This is a test message.")
print(f"Label: {result['label']}")          # 'LABEL_0' or 'LABEL_1'
print(f"Confidence: {result['confidence']:.2%}")

# Multiple predictions with majority voting
predictions = [
    predict("Hello, nice to meet you!"),
    predict("This is so fucking cool!"),
    predict("Thank you for your help."),
]

aggregated = aggregate(predictions)
print(f"Aggregated: {aggregated['label']} ({aggregated['votes']}/{aggregated['total']} votes)")
```

## Testing

### Quick Sanity Check (~5 seconds)
```bash
python sexual/tests/quick_test.py
```

### Comprehensive Unit Tests (~30 seconds)
```bash
python sexual/tests/test_sexual.py
```

### Evaluation on x_sensitive (Training Data)
```bash
# Sanity check on training data (100 examples)
python sexual/tests/evaluate_sexual.py --dataset x_sensitive --limit 100
```

### Evaluation on Out-of-Distribution Datasets
```bash
# True generalization test
python sexual/tests/evaluate_sexual.py --dataset x_sensitive --limit 100

# All datasets
python sexual/tests/evaluate_sexual.py --limit 100
```

## Benchmark Datasets

| Dataset | Description | Type |
|---------|-------------|------|
| **x_sensitive** | Training data sanity check | Training |
| **x_sensitive** | CardiffNLP sensitive content | Training |

## Performance

### x_sensitive Test Set (Training Data Sanity Check)

⚠️ **Note**: x_sensitive was used for training. Results show training performance, not generalization.

| Metric | Score |
|--------|-------|
| Accuracy | 82.6% |
| Precision | - |
| Recall | - |
| F1-Score | 82.9% |

**Per-Class Performance**:
- **LABEL_0 (Safe)**: - precision, - recall
- **LABEL_1 (Sensitive)**: - precision, - recall

### Out-of-Distribution Performance

Run evaluations to get OOD performance:
```bash
python sexual/tests/evaluate_sexual.py --dataset x_sensitive
```

## Integration

This safeguard is automatically included in the main aggregator:
```bash
python aggregator/aggregator.py "Your text here"
```

The aggregator flags content as unsafe when `label == "LABEL_1"` and `confidence >= 0.7`.

## Label Semantics

- **LABEL_0**: Safe content that is not sensitive or sexual
- **LABEL_1**: Content containing sexual material, profanity, self-harm references, drug-related content, conflictual content, or spam

The model is designed to be conservative, erring on the side of safety by flagging borderline content as sensitive.
