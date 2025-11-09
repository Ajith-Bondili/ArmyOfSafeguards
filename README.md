# Army of Safeguards

A modular collection of AI safety safeguards for detecting various types of harmful or problematic content.

## 🏗️ Project Structure

```
ArmyOfSafeguards/
├── factuality/              # Factuality checking safeguard
│   ├── safeguard_factuality.py
│   └── README.md
├── toxicity/                # Toxicity detection (coming soon)
├── sexual/                  # Sexual content detection (coming soon)
├── jailbreak/               # Jailbreak attempt detection (coming soon)
├── aggregator/              # Unified interface for all safeguards
│   ├── aggregator.py
│   └── README.md
├── tests/                   # Test scripts and results
│   ├── test_factuality.py
│   ├── quick_test.py
│   └── TEST_RESULTS.md
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

### 2. Run the Aggregator

Evaluate text using all available safeguards:

```bash
python aggregator/aggregator.py "Your text to evaluate here"
```

### 3. Use Individual Safeguards

Each safeguard can be used independently:

```bash
# Factuality check
python factuality/safeguard_factuality.py "The Earth is flat."
```

## 📦 Available Safeguards

### ✅ Factuality Safeguard
- **Status**: Complete
- **Model**: `ajith-bondili/deberta-v3-factuality-small`
- **Purpose**: Detects factually incorrect or misleading statements
- **Developer**: Ajith
- [View Documentation](factuality/README.md)

### 🚧 Coming Soon
- **Toxicity Detection** (Soham)
- **Sexual Content Detection** (Jian)
- **Jailbreak Detection** (Tommy)

## 🔧 Usage

### Python API

```python
# Use the aggregator for comprehensive evaluation
from aggregator.aggregator import evaluate_text

result = evaluate_text("Your text here", threshold=0.7)
print(f"Is Safe: {result['is_safe']}")
print(f"Flags: {result['flags']}")

# Or use individual safeguards
from factuality.safeguard_factuality import predict

result = predict("The sky is blue.")
print(f"Label: {result['label']}, Confidence: {result['confidence']:.2%}")
```

### Command Line

```bash
# Run all safeguards
python aggregator/aggregator.py "Text to check"

# Run specific safeguard
python factuality/safeguard_factuality.py "Text to check"
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

