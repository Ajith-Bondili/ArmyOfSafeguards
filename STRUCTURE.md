# Project Structure

This document explains the organization of the Army of Safeguards repository.

## Directory Layout

```
ArmyOfSafeguards/
├── factuality/              # Factuality checking safeguard (Ajith)
│   ├── safeguard_factuality.py
│   ├── README.md
│   └── tests/               # Factuality-specific tests
│       ├── test_factuality.py
│       ├── quick_test.py
│       └── benchmark_factuality.py
├── toxicity/                # Toxicity detection (Soham) - Coming soon
│   ├── safeguard_toxicity.py
│   ├── README.md
│   └── tests/               # Toxicity-specific tests
├── sexual/                  # Sexual content detection (Jian) - Coming soon
│   ├── safeguard_sexual.py
│   ├── README.md
│   └── tests/               # Sexual content-specific tests
├── jailbreak/               # Jailbreak detection (Tommy) - Coming soon
│   ├── safeguard_jailbreak.py
│   ├── README.md
│   └── tests/               # Jailbreak-specific tests
├── aggregator/              # Unified interface for all safeguards
│   ├── aggregator.py
│   └── README.md
├── requirements.txt         # Shared dependencies
├── .gitignore              # Git ignore rules (includes venv/)
├── README.md               # Main project documentation
└── STRUCTURE.md            # This file
```

## Design Principles

### 1. Modularity
Each safeguard is self-contained in its own directory with:
- A main Python file with `predict()` function
- A README documenting usage and model info
- A `tests/` subdirectory with module-specific tests
- Independent from other safeguards

### 2. Standardized Interface
All safeguards follow the same pattern:
```python
def predict(text: str) -> Dict[str, float]:
    """
    Returns:
        {"label": str, "confidence": float}
    """
```

### 3. Aggregator Pattern
The `aggregator/` module:
- Imports all available safeguards
- Runs them on input text
- Combines results with configurable logic
- Provides both CLI and Python API

### 4. Testing
The `tests/` directory contains:
- Unit tests for individual safeguards
- Integration tests
- Test results documentation
- Quick sanity checks

## Adding a New Safeguard

1. **Create directory**: `mkdir your_safeguard/`

2. **Implement safeguard**: Create `your_safeguard/safeguard_your_name.py`
   ```python
   def predict(text: str) -> Dict[str, float]:
       # Your implementation
       return {"label": "...", "confidence": 0.xx}
   ```

3. **Add to aggregator**: Edit `aggregator/aggregator.py`
   ```python
   from your_safeguard.safeguard_your_name import predict as your_check
   results['your_safeguard'] = your_check(text)
   ```

4. **Document**: Create `your_safeguard/README.md`

5. **Test**: Add tests in `your_safeguard/tests/`

## Running the System

### Individual Safeguard
```bash
python factuality/safeguard_factuality.py "Text to check"
```

### All Safeguards (Aggregator)
```bash
python aggregator/aggregator.py "Text to check"
```

### Tests
```bash
# Factuality tests
python factuality/tests/test_factuality.py
python factuality/tests/quick_test.py
python factuality/tests/benchmark_factuality.py
```

## Import Paths

All modules use path manipulation to enable imports from the project root:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

This allows running scripts from any directory while maintaining clean imports.

## Virtual Environment

- **Location**: `venv/` (git-ignored)
- **Setup**: `python3 -m venv venv && source venv/bin/activate`
- **Dependencies**: `pip install -r requirements.txt`

The virtual environment is NOT committed to git. Each developer creates their own from `requirements.txt`.

## Git Workflow

1. Each person works on their own branch (e.g., `ajith`, `soham`)
2. Implement your safeguard in its own directory
3. Test thoroughly
4. Push to your branch
5. Create PR for review
6. Merge to main

## Phase 1 Deliverable

The complete system will have:
- ✅ **Factuality safeguard (Ajith) - COMPLETE**
  - ✅ Model trained and deployed on Hugging Face
  - ✅ Standardized `predict()` and `aggregate()` functions
  - ✅ Comprehensive tests and documentation
  - ✅ Benchmark framework (`factuality/tests/benchmark_factuality.py`)
  - ✅ Full evaluation with metrics (`factuality/tests/evaluate_factuality.py`)
  - ✅ Evaluated on TruthfulQA, FEVER, SciFact
  - ✅ **Results: 84% accuracy on FEVER, 78.38% F1-score**
- 🚧 Toxicity safeguard (Soham)
- 🚧 Sexual content safeguard (Jian)
- 🚧 Jailbreak safeguard (Tommy)
- ✅ Aggregator (shared)
- ✅ Tests and documentation

The aggregator will be the main entry point for evaluating text against all safeguards.

