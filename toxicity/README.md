# Toxicity Safeguard

The toxicity safeguard critic detects racist, hateful, and toxic content in model outputs. Powered by a fine-tuned DeBERTa-v2-base classifier.

## Model Information

- **Model**: [SohamNagi/tiny-toxicity-classifier](https://huggingface.co/SohamNagi/tiny-toxicity-classifier)
- **Base Architecture**: DeBERTa-v2-base (184M parameters)
- **Training Data**: ToxiGen annotated dataset
- **Task**: Binary sequence classification
- **Labels**:
  - `safe` (0): Non-toxic, respectful content
  - `unsafe` (1): Toxic, racist, or hateful content

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
python toxicity/safeguard_toxicity.py "Your text to evaluate"
```

### Python API

```python
from toxicity.safeguard_toxicity import predict, aggregate

# Single prediction
result = predict("This is a test message.")
print(f"Label: {result['label']}")          # 'safe' or 'unsafe'
print(f"Confidence: {result['confidence']:.2%}")

# Multiple predictions with majority voting
predictions = [
    predict("Hello, nice to meet you!"),
    predict("I hate everything about this."),
    predict("Thank you for your help."),
]

aggregated = aggregate(predictions)
print(f"Aggregated: {aggregated['label']} ({aggregated['votes']}/{aggregated['total']} votes)")
```

## Testing

### Quick Sanity Check (~5 seconds)
```bash
python toxicity/tests/quick_test.py
```

### Comprehensive Unit Tests (~30 seconds)
```bash
python toxicity/tests/test_toxicity.py
```

### Evaluation on ToxiGen (Training Data)
```bash
# Sanity check on training data (100 examples)
python toxicity/tests/evaluate_toxigen.py --split test --limit 100
```

### Evaluation on Out-of-Distribution Datasets
```bash
# True generalization test
python toxicity/tests/evaluate_toxicity.py --dataset hate_speech18 --limit 100

# All OOD datasets
python toxicity/tests/evaluate_toxicity.py --limit 100
```

## Benchmark Datasets

| Dataset | Description | Type |
|---------|-------------|------|
| **ToxiGen** | Training data sanity check | Training |
| **hate_speech18** | Twitter hate speech | OOD |
| **civil_comments** | Civil Comments toxicity | OOD |
| **tweets_hate_speech_detection** | Tweet hate detection | OOD |

## Performance

### ToxiGen Test Set (Training Data Sanity Check)

⚠️ **Note**: ToxiGen was used for training. Results show training performance, not generalization.

| Metric | Score |
|--------|-------|
| Accuracy | 79.00% |
| Precision | 75.00% |
| Recall | 69.23% |
| F1-Score | 72.00% |

**Per-Class Performance**:
- **safe**: 81.25% precision, 85.25% recall
- **unsafe**: 75.00% precision, 69.23% recall

### Out-of-Distribution Performance

Run evaluations to get OOD performance:
```bash
python toxicity/tests/evaluate_toxicity.py --dataset hate_speech18
```

## Integration

This safeguard is automatically included in the main aggregator:
```bash
python aggregator/aggregator.py "Your text here"
```

The aggregator flags content as unsafe when `label == "unsafe"` and `confidence >= 0.7`.

## Label Semantics

- **safe**: Non-toxic, respectful content that is appropriate
- **unsafe**: Content containing hate speech, racism, or toxic language

The model is designed to be conservative, erring on the side of safety by flagging borderline content as unsafe.
