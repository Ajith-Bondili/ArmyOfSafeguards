# Factuality Safeguard

The factuality safeguard critic flags model outputs that contradict verified facts or propagate misinformation. It is powered by the fine-tuned [ajith-bondili/deberta-v3-factuality-small](https://huggingface.co/ajith-bondili/deberta-v3-factuality-small) DeBERTa-v3 sequence classifier.

## Model Information

- **Model**: `ajith-bondili/deberta-v3-factuality-small`
- **Base Architecture**: DeBERTa-v3-small
- **Task**: Binary sequence classification
- **Labels**: 
  - `LABEL_0`: Factual
  - `LABEL_1`: Non-factual/Uncertain

## Installation

From the project root:

```bash
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
python factuality/safeguard_factuality.py "The Earth orbits the sun once every 365 days."
```

### Python API

```python
from factuality.safeguard_factuality import predict, aggregate

# Single prediction
result = predict("The capital of France is Paris.")
print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']:.4f}")

# Multiple predictions with majority voting
predictions = [
    predict("Water boils at 100°C."),
    predict("The moon is made of cheese."),
    predict("Python is a programming language."),
]

aggregated = aggregate(predictions)
print(f"Aggregated: {aggregated['label']} ({aggregated['votes']}/{aggregated['total']} votes)")
```

## Features

- **predict(text)**: Returns label and confidence score for a single text
- **aggregate(predictions)**: Majority-vote aggregation across multiple predictions
- **CLI interface**: Direct command-line evaluation
- **Error handling**: Validates input and handles edge cases

## Testing

See `tests/test_factuality.py` for comprehensive tests within this module.

## 📊 Benchmark Datasets

The factuality safeguard can be benchmarked against standard datasets:

| Dataset | Description | Source |
|---------|-------------|--------|
| **TruthfulQA** | LLM factuality benchmark with adversarial questions | `truthful_qa` |
| **FEVER** | Wikipedia-based claim verification dataset | `fever` |
| **SciFact** | Scientific factuality verification | `allenai/scifact` |
| **VitaminC** | Contradiction-aware claim dataset | `tals/vitaminc` |
| **Climate-FEVER** | Climate misinformation detection | `climate_fever` |

### Running Benchmarks

**Quick benchmark** (shows prediction distribution):
```bash
python factuality/tests/benchmark_factuality.py --limit 100
```

**Full evaluation** (calculates accuracy, precision, recall, F1-score):
```bash
python factuality/tests/evaluate_factuality.py --limit 100
```

Options:
```bash
# Evaluate single dataset
python factuality/tests/evaluate_factuality.py --dataset FEVER

# Custom sample size
python factuality/tests/evaluate_factuality.py --limit 500

# Don't save results to JSON
python factuality/tests/evaluate_factuality.py --no-save
```

### Evaluation Results

**⚠️ Important**: The model was trained on TruthfulQA and FEVER, so those results show training performance, not generalization.

**Out-of-Distribution Performance** (true generalization test):

| Dataset | Accuracy | Precision | Recall | F1-Score | Domain |
|---------|----------|-----------|--------|----------|--------|
| **VitaminC** | 54.00% | 46.43% | 29.55% | 36.11% | General claims |
| **Climate-FEVER** | 81.00% | - | - | - | Climate claims |
| **LIAR** | 81.00% | - | - | - | Political statements |

**Training Data Performance** (sanity check only):

| Dataset | Accuracy | F1-Score | Note |
|---------|----------|----------|------|
| **FEVER** | 84.00% | 78.38% | ⚠️ Used in training |
| **TruthfulQA** | 75.00% | - | ⚠️ Used in training |

See `tests/EVALUATION_SUMMARY.md` for detailed analysis and recommendations.

## Integration

This safeguard is automatically included in the main aggregator at `../aggregator/aggregator.py`.
