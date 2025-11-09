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

## 🧪 Testing

```bash
# Run all tests
python tests/test_factuality.py

# Quick sanity check
python tests/quick_test.py
```

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

