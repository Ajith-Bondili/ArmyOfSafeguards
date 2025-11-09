# Army of Safeguards

**CS399 - UR2PhD Project** - A modular multiagent safeguarding system for LLM output detection

A modular collection of AI safety safeguards for detecting various types of harmful or problematic content.

## 🏗️ Project Structure

```
ArmyOfSafeguards/
├── factuality/              # Factuality checking safeguard
│   ├── safeguard_factuality.py
│   ├── README.md
│   └── tests/               # Factuality-specific tests
│       ├── test_factuality.py
│       ├── quick_test.py
│       ├── benchmark_factuality.py
│       ├── evaluate_factuality.py
│       └── EVALUATION_SUMMARY.md
├── toxicity/                # Toxicity detection (coming soon)
├── sexual/                  # Sexual content detection (coming soon)
├── jailbreak/               # Jailbreak attempt detection (coming soon)
├── aggregator/              # Unified interface for all safeguards
│   ├── aggregator.py
│   └── README.md
├── requirements.txt         # Shared dependencies
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/SohamNagi/ArmyOfSafeguards.git
cd ArmyOfSafeguards

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Use the Factuality Safeguard

Currently, only the factuality safeguard is implemented:

```bash
# Run factuality check
python factuality/safeguard_factuality.py "The Earth is flat."
```

### 3. Aggregator (Ready for Team Integration)

The aggregator framework is ready and currently includes the factuality safeguard:

```bash
# Run aggregator (currently only factuality)
python aggregator/aggregator.py "Your text to evaluate here"
```

**Note**: As teammates add their safeguards (toxicity, sexual, jailbreak), they will be automatically included in the aggregator.

## 📦 Safeguards Status

### ✅ Complete

#### Factuality Safeguard (Ajith)
- **Model**: `ajith-bondili/deberta-v3-factuality-small`
- **Purpose**: Detects factually incorrect or misleading statements
- **Performance**: 54-81% accuracy on out-of-distribution datasets
- **Documentation**: [factuality/README.md](factuality/README.md)
- **Tests**: [factuality/tests/README.md](factuality/tests/README.md)

### 🚧 In Development
- **Toxicity Detection** (Soham) - Not yet implemented
- **Sexual Content Detection** (Jian) - Not yet implemented
- **Jailbreak Detection** (Tommy) - Not yet implemented

### ✅ Infrastructure Complete
- **Aggregator Framework**: Ready to integrate multiple safeguards
- **Testing Template**: Comprehensive test structure for teammates to follow
- **Documentation Template**: Clear pattern for documenting safeguards

## 🔧 Usage

### Factuality Safeguard (Currently Available)

**Python API**:
```python
from factuality.safeguard_factuality import predict

# Single prediction
result = predict("The sky is blue.")
print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")

# Multiple predictions with aggregation
from factuality.safeguard_factuality import aggregate

predictions = [
    predict("Water boils at 100°C."),
    predict("The moon is made of cheese."),
]
result = aggregate(predictions)
print(f"Aggregated: {result['label']} ({result['votes']}/{result['total']} votes)")
```

**Command Line**:
```bash
python factuality/safeguard_factuality.py "Text to check"
```

### Aggregator (Framework Ready)

**Python API**:
```python
from aggregator.aggregator import evaluate_text

# Currently runs factuality safeguard only
# Will automatically include other safeguards as they're added
result = evaluate_text("Your text here", threshold=0.7)
print(f"Is Safe: {result['is_safe']}")
print(f"Individual Results: {result['individual_results']}")
```

**Command Line**:
```bash
python aggregator/aggregator.py "Text to check"
```

## 🧪 Testing & Evaluation

Each safeguard has its own test suite in its directory:

```bash
# Factuality tests
python factuality/tests/test_factuality.py

# Quick sanity check
python factuality/tests/quick_test.py

# Benchmark (prediction distribution)
python factuality/tests/benchmark_factuality.py

# Full evaluation (accuracy, precision, recall, F1)
python factuality/tests/evaluate_factuality.py
```

### Evaluation Results

**Factuality Safeguard Performance**:

⚠️ **Note**: Model trained on TruthfulQA & FEVER - use OOD datasets for true generalization.

**Out-of-Distribution (True Generalization)**:
| Dataset | Accuracy | F1-Score | Domain |
|---------|----------|----------|--------|
| VitaminC | 54.00% | 36.11% | General claims |
| Climate-FEVER | 81.00% | - | Climate-specific |
| LIAR | 81.00% | - | Political statements |

**Training Data (Sanity Check)**:
| Dataset | Accuracy | F1-Score |
|---------|----------|----------|
| FEVER | 84.00% | 78.38% |
| TruthfulQA | 75.00% | - |

### Benchmark Datasets

The factuality safeguard can be evaluated on:
- **TruthfulQA** - LLM factuality benchmark
- **FEVER** - Wikipedia claim verification  
- **SciFact** - Scientific factuality
- **VitaminC** - Contradiction-aware claims
- **Climate-FEVER** - Climate misinformation

See `factuality/tests/` for benchmark and evaluation scripts.

## 🤝 Contributing

Each team member maintains their own safeguard module:

1. Create your safeguard in its own directory (e.g., `toxicity/`)
2. Implement `predict()` function that returns `{"label": str, "confidence": float}`
3. Add your safeguard to the aggregator
4. Include tests and documentation

## 📝 Requirements

- Python 3.9+
- PyTorch
- Transformers
- See `requirements.txt` for full list

## 📄 License

[Add license information]

## 👥 Team

- **Ajith**: Factuality Safeguard
- **Soham**: Toxicity Safeguard
- **Jian**: Sexual Content Safeguard
- **Tommy**: Jailbreak Safeguard
