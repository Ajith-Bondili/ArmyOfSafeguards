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

See `../tests/test_factuality.py` for comprehensive tests.

## Integration

This safeguard is automatically included in the main aggregator at `../aggregator/aggregator.py`.
